"""Owned offline worker for the pinned Qwen3-TTS CustomVoice Python interface."""

from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import resource
import sys
import types
import wave

if __package__:
    from .assets_store import AssetError, prepare
    from .logic import configuration
    from .process import QwenError
else:
    package = types.ModuleType("qwen_worker_owned")
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__] = package
    from qwen_worker_owned.assets_store import AssetError, prepare
    from qwen_worker_owned.logic import configuration
    from qwen_worker_owned.process import QwenError

SDK_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"


def available_ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    try:
        cap = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if cap != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            available = min(available, max(0, int(cap) - used))
    except (OSError, ValueError):
        pass
    return available


def require_dependencies():
    from importlib.metadata import distribution, version
    for name, expected in (("torch", "2.10.0"), ("torchaudio", "2.10.0"),
                           ("transformers", "4.57.3"), ("accelerate", "1.12.0"), ("qwen-tts", "0.1.1")):
        if version(name).split("+")[0] != expected:
            raise QwenError("dependency")
    try:
        origin = json.loads(distribution("qwen-tts").read_text("direct_url.json") or "{}")
        if origin.get("url", "").removesuffix(".git") != "https://github.com/QwenLM/Qwen3-TTS" or origin.get("vcs_info", {}).get("commit_id") != SDK_REVISION:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise QwenError("dependency") from None


def load_model(assets, cfg, torch):
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(str(assets), device_map=cfg["device"],
        dtype=getattr(torch, cfg["precision"]), attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False)
    inner = model.model
    if (inner.tts_model_type, inner.tts_model_size, inner.tokenizer_type) != (
            "custom_voice", "1b7", "qwen3_tts_tokenizer_12hz"):
        raise QwenError("assets_corrupt")
    if cfg["speaker"].lower() not in (model.get_supported_speakers() or ()) or cfg["language"].lower() not in (model.get_supported_languages() or ()):
        raise QwenError("assets_corrupt")
    if inner.config.talker_config.codec_eos_token_id != 2150:
        raise QwenError("assets_corrupt")
    return model


def guard_eos(talker, maximum):
    """Observe the pinned SDK's actual HF token result before it discards EOS.

    The high-level SDK returns audio even on max-token exhaustion. This owned,
    instance-local observer rejects such a result; no upstream file is edited.
    Single-request workers ensure this cannot affect another block's model.
    """
    original = talker.generate
    state = {"called": False, "eos": False, "tokens": 0}

    def generate(*args, **kwargs):
        if state["called"] or kwargs.get("eos_token_id") != 2150 or kwargs.get("max_new_tokens") != maximum:
            raise QwenError("model_failed")
        state["called"] = True
        result = original(*args, **kwargs)
        sequences = getattr(result, "sequences", None)
        if sequences is None or sequences.ndim != 2 or sequences.shape[0] != 1 or not 1 <= sequences.shape[1] <= maximum:
            raise QwenError("model_failed")
        tokens = sequences[0].tolist()
        if any(type(token) is not int for token in tokens):
            raise QwenError("model_failed")
        if tokens[-1] != 2150 or 2150 in tokens[:-1]:
            raise QwenError("audio_limit")
        state.update(eos=True, tokens=len(tokens))
        return result

    talker.generate = generate
    return original, state


def generate_audio(model, text, cfg, job):
    import numpy as np
    # Native tokenizer emits 12.5 frames/sec. One further token allows EOS;
    # the decoded duration is independently enforced without truncation.
    maximum = min(cfg["max_new_tokens"], math.ceil(cfg["max_audio_sec"] * 12.5) + 1)
    talker = model.model.talker
    original, state = guard_eos(talker, maximum)
    try:
        wavs, rate = model.generate_custom_voice(text=text, speaker=cfg["speaker"],
            language=cfg["language"], instruct=cfg["instruct"], non_streaming_mode=True,
            max_new_tokens=maximum, do_sample=True, temperature=cfg["temperature"],
            top_k=cfg["top_k"], top_p=cfg["top_p"], repetition_penalty=cfg["repetition_penalty"],
            subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)
    finally:
        talker.generate = original
    if not state["eos"]:
        raise QwenError("audio_limit")
    if type(rate) is not int or rate != 24000 or not isinstance(wavs, (list, tuple)) or len(wavs) != 1:
        raise QwenError("audio_invalid")
    audio = np.asarray(wavs[0])
    if audio.ndim != 1 or not 0 < audio.size <= rate * cfg["max_audio_sec"] or audio.dtype.kind != "f" or not np.isfinite(audio).all():
        raise QwenError("audio_invalid")
    peak = float(np.abs(audio).max())
    if peak <= 1e-7:
        raise QwenError("audio_invalid")
    gain = min(1., .99 / peak)
    pcm = np.rint(audio * gain * 32767).astype("<i2").tobytes()
    path = Path(job) / "generated.wav"
    with path.open("xb") as target:
        os.chmod(path, 0o600)
        with wave.open(target, "wb") as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(rate); output.writeframes(pcm)
    return {"complete": True, "eos": True, "tokens": state["tokens"], "sample_rate": rate, "frames": int(audio.size), "gain": gain}


def infer(item):
    cfg = configuration(item["config"])
    if cfg["memory_limit_mib"]:
        limit = cfg["memory_limit_mib"] * 1048576
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cpu_limit = cfg["timeout_sec"] * cfg["threads"]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    require_dependencies()
    if available_ram() < cfg["min_available_ram_mib"] * 1048576:
        raise QwenError("resources")
    import torch
    torch.set_num_threads(cfg["threads"]); torch.set_num_interop_threads(1)
    if cfg["device"] == "cuda:0" and (not torch.cuda.is_available() or
            cfg["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported()):
        raise QwenError("device")
    assets = prepare(item["assets"])
    model = load_model(assets, cfg, torch)
    with torch.inference_mode():
        return generate_audio(model, item["request"]["text"], cfg, Path(item["job"]))


def main():
    data = sys.stdin.buffer.read(32769)
    if len(data) > 32768:
        raise QwenError("input")
    with redirect_stdout(sys.stderr):
        result = infer(json.loads(data))
    sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        from importlib.metadata import PackageNotFoundError
        if isinstance(exc, QwenError): code = exc.code
        elif isinstance(exc, AssetError): code = str(exc)
        elif isinstance(exc, (ImportError, PackageNotFoundError)): code = "dependency"
        elif isinstance(exc, MemoryError) or isinstance(exc, RuntimeError) and any(s in str(exc) for s in ("out of memory", "DefaultCPUAllocator", "Cannot allocate memory")):
            code = "resources"
        else: code = "model_failed"
        print(json.dumps({"error_code": code})); raise SystemExit(1) from None
