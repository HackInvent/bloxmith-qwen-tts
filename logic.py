"""Bounded complete-file synthesis; publication is separate from cancellable native work."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
from uuid import uuid4
import wave
from .process import QwenError, Cancelled, check, run_process

DEFAULTS = {
    "asset_directory": "",
    "environment_python": "",
    "output_directory": "",
    "device": "cuda:0",
    "precision": "bfloat16",
    "speaker": "Ryan",
    "language": "Auto",
    "instruct": "",
    "format": "ogg_opus",
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1.05,
    "threads": 2,
    "max_audio_sec": 30,
    "max_new_tokens": 1024,
    "max_text_chars": 1000,
    "timeout_sec": 240,
    "memory_limit_mib": 0,
    "min_available_ram_mib": 16384,
    "max_pending": 4
}
SPEAKERS = ("Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee")
LANGUAGES = ("Auto", "English", "French", "German", "Spanish", "Italian", "Portuguese", "Russian", "Chinese", "Japanese", "Korean")


def configuration(raw):
    if not isinstance(raw, Mapping) or set(raw)-set(DEFAULTS)-{"execution","position","runtime_path","runtime_path_label"}:
        raise QwenError("config")
    cfg={**DEFAULTS,**{k:v for k,v in raw.items() if k in DEFAULTS}}
    for key in ("asset_directory","environment_python","output_directory"):
        value=cfg[key]
        try:
            if not isinstance(value,str) or len(value.encode("utf-8"))>4096 or any(ord(c)<32 for c in value):
                raise QwenError("config")
        except UnicodeError:
            raise QwenError("config") from None
    try:
        if not isinstance(cfg["instruct"], str) or len(cfg["instruct"]) > 512 or len(cfg["instruct"].encode("utf-8")) > 2048 or "\0" in cfg["instruct"]:
            raise QwenError("config")
    except UnicodeError:
        raise QwenError("config") from None
    for key,choices in (("device",("cpu","cuda:0")),("precision",("float32","bfloat16")),("format",("wav","ogg_opus")),("speaker",SPEAKERS),("language",LANGUAGES)):
        if not isinstance(cfg[key],str) or cfg[key] not in choices:
            raise QwenError("config")
    for key,low,high in (("top_k",1,500),("threads",1,16),("max_audio_sec",1,90),
            ("max_text_chars",1,2000),("timeout_sec",1,240),("memory_limit_mib",0,262144),
            ("min_available_ram_mib",8192,262144),("max_pending",0,16),("max_new_tokens",16,4096)):
        if type(cfg[key]) is not int or not low<=cfg[key]<=high:
            raise QwenError("config")
    if 0 < cfg["memory_limit_mib"] < 4096:
        raise QwenError("config")
    for key, low, high in (("temperature",.1,2),("top_p",.01,1),("repetition_penalty",.1,2)):
        if type(cfg[key]) not in (int,float) or not math.isfinite(cfg[key]) or not low <= cfg[key] <= high:
            raise QwenError("config")
    return cfg


def json_value(raw, limit):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise QwenError("input")
            out[k]=v
        return out
    if not isinstance(raw,str): return raw
    try:
        if len(raw.encode("utf-8"))>limit: raise QwenError("input")
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(QwenError("input")))
    except (ValueError,UnicodeError,RecursionError):
        raise QwenError("input") from None


def interrupt(raw):
    value=json_value(raw,1024)
    if not isinstance(value,Mapping) or dict(value)!={"action":"interrupt"}:
        raise QwenError("input")
    return {"action":"interrupt"}


def request(raw, cfg):
    correlation=uuid4().hex
    # Plain strings remain speech, even if they look like commands. Correlated
    # JSON must arrive as a mapping, or be decoded according to its input MIME.
    if isinstance(raw,Mapping):
        if set(raw)!={"text","request_id"}: raise QwenError("input")
        correlation=raw["request_id"];raw=raw["text"]
        if not isinstance(correlation,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}",correlation):
            raise QwenError("input")
    try:
        if not isinstance(raw,str) or not raw.strip() or len(raw)>cfg["max_text_chars"] or len(raw.encode("utf-8"))>8192 or "\0" in raw:
            raise QwenError("input")
    except UnicodeError:
        raise QwenError("input") from None
    return {"text":raw.strip(),"request_id":correlation}


def resolve_path(root, value):
    path=Path(value).expanduser()
    return (Path(root)/path).absolute() if not path.is_absolute() else path


@dataclass
class PreparedAudio:
    owner: object
    path: Path
    output_directory: Path
    metadata: dict
    deadline: float

    def close(self):
        self.owner.cleanup()


def audio_info(path, cfg, raw):
    try:
        info=json_value(raw.decode("utf-8") if isinstance(raw,bytes) else raw,8192)
        if set(info)!={"complete","eos","tokens","sample_rate","frames","gain"} or info["complete"] is not True or info["eos"] is not True:
            raise ValueError()
        if type(info["tokens"]) is not int or not 1 <= info["tokens"] <= cfg["max_new_tokens"]:
            raise ValueError()
        if type(info["frames"]) is not int or not 0<info["frames"]<=24000*cfg["max_audio_sec"] or type(info["sample_rate"]) is not int or info["sample_rate"]!=24000:
            raise ValueError()
        if type(info["gain"]) not in (int,float) or not math.isfinite(info["gain"]) or not 0<info["gain"]<=1:
            raise ValueError()
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as source:
            file=os.fstat(source.fileno())
            if not stat.S_ISREG(file.st_mode) or not 44<=file.st_size<=24000*cfg["max_audio_sec"]*2+128:
                raise ValueError()
            with wave.open(source,"rb") as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes())!=(1,2,24000,info["frames"]):
                    raise ValueError()
                if len(wav.readframes(info["frames"]+1))!=info["frames"]*2:
                    raise ValueError()
        return info
    except (ValueError,TypeError,KeyError,OSError,EOFError,wave.Error):
        raise QwenError("audio_invalid") from None


def generate(item, cfg, root, storage, cancel=lambda:False):
    cfg=configuration(cfg);item=request(item,cfg)
    storage=Path(storage)
    if not storage.is_absolute() or storage.is_symlink() or not storage.is_dir(): raise QwenError("storage")
    assets=resolve_path(root,cfg["asset_directory"])
    if not cfg["asset_directory"] or not assets.is_dir() or assets.is_symlink(): raise QwenError("assets_missing")
    python=resolve_path(root,cfg["environment_python"]) if cfg["environment_python"] else Path(sys.executable)
    if not python.is_file() or not os.access(python,os.X_OK): raise QwenError("dependency")
    target=resolve_path(root,cfg["output_directory"]) if cfg["output_directory"] else storage/"audio"
    if target.is_symlink() or target.exists() and not target.is_dir(): raise QwenError("storage")
    started=time.monotonic();deadline=started+cfg["timeout_sec"]
    owner=tempfile.TemporaryDirectory(prefix="qwen-job-",dir=storage);job=Path(owner.name)
    try:
        envelope={"config":cfg,"request":item,"assets":str(assets),"job":str(job)}
        raw=run_process([str(python),"-I","-B",str(Path(__file__).with_name("worker.py"))],directory=job,
            deadline=deadline,cancel=cancel,limit=8192,stdin=json.dumps(envelope,ensure_ascii=True).encode())
        del envelope
        wav=job/"generated.wav";info=audio_info(wav,cfg,raw)
        output=wav
        if cfg["format"]=="ogg_opus":
            output=job/"generated.ogg"
            run_process(["ffmpeg","-v","error","-nostdin","-n","-threads","1","-protocol_whitelist","file,pipe",
                "-format_whitelist","wav","-i",str(wav),"-map_metadata","-1","-ac","1","-ar","48000",
                "-c:a","libopus","-b:a","96000","-f","ogg",str(output)],directory=job,deadline=deadline,cancel=cancel,limit=1024)
        check(deadline,cancel)
        info.update(request_id=item["request_id"],duration_sec=info["frames"]/24000,
            source_frames=info["frames"],
            format=cfg["format"],sample_rate=48000 if cfg["format"]=="ogg_opus" else 24000,channels=1,
            source_sample_rate=24000,elapsed_sec=round(time.monotonic()-started,4),
            model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",model_revision="0c0e3051f131929182e2c023b9537f8b1c68adfe",
            engine_revision="022e286b98fbec7e1e916cb940cdf532cd9f488e",
            settings={k:cfg[k] for k in ("device","precision","speaker","language","temperature","top_k","top_p","repetition_penalty","max_audio_sec","max_new_tokens")})
        del info["frames"]
        return PreparedAudio(owner,output,target,info,deadline)
    except BaseException:
        owner.cleanup()
        raise


def publish(prepared, cancel=lambda:False):
    """Publish after the listener has drained pending interrupts; never overwrite a file."""
    check(prepared.deadline,cancel)
    target=prepared.output_directory
    if target.is_symlink(): raise QwenError("storage")
    target.mkdir(mode=0o700,parents=True,exist_ok=True)
    suffix=".ogg" if prepared.metadata["format"]=="ogg_opus" else ".wav"
    final=target/("qwen-"+uuid4().hex+suffix)
    temporary=None
    try:
        fd=os.open(prepared.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as source:
            size=os.fstat(source.fileno())
            if not stat.S_ISREG(size.st_mode) or not 0<size.st_size<=33554432: raise QwenError("audio_invalid")
            descriptor,temporary=tempfile.mkstemp(prefix=".qwen-publish-",dir=target)
            digest=hashlib.sha256();total=0
            with os.fdopen(descriptor,"wb") as output:
                while chunk:=source.read(65536):
                    check(prepared.deadline,cancel);total+=len(chunk)
                    if total>33554432: raise QwenError("audio_invalid")
                    output.write(chunk);digest.update(chunk)
                output.flush();os.fsync(output.fileno())
            if total!=size.st_size: raise QwenError("audio_invalid")
        check(prepared.deadline,cancel)
        os.link(temporary,final,follow_symlinks=False)
        metadata={**prepared.metadata,"path":str(final),"bytes":total,"sha256":digest.hexdigest(),
            "content_type":"audio/ogg" if suffix==".ogg" else "audio/wav"}
        return metadata
    except OSError:
        raise QwenError("storage") from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
