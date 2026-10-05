"""Portable isolated fixtures. No actual model or personal data is used."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SECRET="[12,34,56,78,90]"


def settings(root):
    root=Path(root);assets=root/"synthetic-assets";assets.mkdir(exist_ok=True)
    executable=root/"synthetic-qwen-python"
    executable.write_text("#!"+sys.executable+"\nimport runpy\nrunpy.run_path("+
        repr(str(Path(__file__).with_name("fixture_worker.py")))+",run_name='__main__')\n",encoding="utf-8")
    executable.chmod(0o700)
    return {"asset_directory":str(assets),"environment_python":str(executable),
        "device":"cpu","timeout_sec":30,"max_pending":1}


def alive(pid):
    try:return Path(f"/proc/{int(pid)}/stat").read_text().split(")",1)[1].strip().split()[0]!="Z"
    except (FileNotFoundError, ProcessLookupError):return False


def wait(predicate,timeout=8):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        time.sleep(.02)
    raise AssertionError("Timed out waiting for synthetic native process state")


def started(cfg,ident):
    path=Path(cfg["asset_directory"])/("started-"+ident+".json")
    wait(path.is_file)
    return json.loads(path.read_text())


def check_file(metadata):
    path=Path(metadata["path"]);raw=path.read_bytes()
    assert metadata["complete"] is True and metadata["eos"] is True
    assert len(raw)==metadata["bytes"] and hashlib.sha256(raw).hexdigest()==metadata["sha256"]
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)]))
    stream=probe["streams"][0]
    assert stream["channels"]==1 and int(stream["sample_rate"])==metadata["sample_rate"]
    assert stream["codec_name"]==("opus" if metadata["format"]=="ogg_opus" else "pcm_s16le")
    decoded=subprocess.check_output(["ffmpeg","-v","error","-nostdin","-i",str(path),"-ac","1","-ar","24000","-f","s16le","pipe:1"])
    assert len(decoded)==24000 and any(decoded),len(decoded)
    assert metadata["duration_sec"]==.5 and metadata["source_frames"]==12000
    assert SECRET not in json.dumps(metadata)
    return path
