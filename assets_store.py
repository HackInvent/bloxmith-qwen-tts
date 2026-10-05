"""Explicit pinned public Qwen model preparation; runtime never downloads assets."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class AssetError(ValueError):
    """Public diagnostic code, without paths, credentials or provider responses."""


def manifest():
    return json.loads(Path(__file__).with_name("model_artifacts.json").read_text(encoding="utf-8"))


def safe_path(root, relative):
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise AssetError("assets_missing")
    path = Path(relative)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise AssetError("assets_corrupt")
    target = root
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise AssetError("assets_corrupt")
    return target


def hasher(spec):
    if "sha256" in spec:
        return hashlib.sha256(), spec["sha256"]
    if "git_blob_sha1" in spec:
        # Small upstream files are Git blobs rather than LFS blobs. Hash their
        # object header as well as contents, exactly as in the pinned Git tree.
        digest = hashlib.sha1()
        digest.update(f"blob {spec['bytes']}\0".encode("ascii"))
        return digest, spec["git_blob_sha1"]
    raise AssetError("invalid_manifest")


def verify(root, spec, *, deadline=None):
    path = safe_path(root, spec["path"])
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    except OSError:
        raise AssetError("assets_corrupt") from None
    with os.fdopen(fd, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != spec["bytes"]:
            raise AssetError("assets_corrupt")
        digest, expected = hasher(spec)
        while chunk := source.read(1048576):
            if deadline is not None and time.monotonic() >= deadline:
                raise AssetError("asset_timeout")
            digest.update(chunk)
        after = os.fstat(source.fileno())
        if digest.hexdigest() != expected or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise AssetError("assets_corrupt")
    return True


def inventory(root, *, deadline=None):
    """Read-only model inventory, checksum verification and disk preflight."""
    root = Path(root).absolute()
    specs = manifest()["artifacts"]
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise AssetError("assets_corrupt")
    missing = [spec for spec in specs if not root.exists() or not verify(root, spec, deadline=deadline)]
    probe = root
    while not probe.exists():
        probe = probe.parent
    available = shutil.disk_usage(probe).free
    required = sum(spec["bytes"] for spec in missing)
    return {"ready": not missing, "required_bytes": sum(s["bytes"] for s in specs),
        "missing_bytes": required, "available_bytes": available,
        "enough_disk": available >= required + 1073741824,
        "missing_public": [s["path"] for s in missing if s["access"] == "public"],
        "missing_authorized": [s["path"] for s in missing if s["access"] == "user_prepared_gated"]}


class PinnedRedirects(HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or parsed.username or parsed.password or not (
            host in {"raw.githubusercontent.com", "huggingface.co"} or
            host.endswith(".huggingface.co") or host.endswith(".hf.co")):
            raise AssetError("download_failed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_public(root, *, timeout_sec=3600, progress=lambda path: None):
    """Explicit setup only; refusal gates run before any request or directory creation."""
    if type(timeout_sec) is not int or not 1 <= timeout_sec <= 14400:
        raise AssetError("invalid_timeout")
    root = Path(root).absolute()
    deadline = time.monotonic() + timeout_sec
    state = inventory(root, deadline=deadline)
    if state["missing_authorized"]:
        raise AssetError("authorized_tokenizer_required")
    if state["ready"]:
        return root
    if not state["enough_disk"]:
        raise AssetError("disk_space")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    opener = build_opener(ProxyHandler({}), PinnedRedirects())
    missing = set(state["missing_public"])
    for spec in manifest()["artifacts"]:
        if spec["path"] not in missing:
            continue
        if spec["access"] != "public":
            raise AssetError("authorized_tokenizer_required")
        if time.monotonic() >= deadline:
            raise AssetError("asset_timeout")
        target = safe_path(root, spec["path"])
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        progress(spec["path"])
        with tempfile.TemporaryDirectory(prefix=".qwen-download-", dir=root) as temporary:
            candidate = Path(temporary) / "asset"
            request = Request(spec["url"], headers={"User-Agent": "BloxSmith-Qwen-TTS/0.1.0", "Accept-Encoding": "identity"})
            try:
                with opener.open(request, timeout=min(20, max(.1, deadline-time.monotonic()))) as source, candidate.open("xb") as output:
                    os.chmod(candidate, 0o600)
                    digest, expected = hasher(spec)
                    total = 0
                    while True:
                        if time.monotonic() >= deadline:
                            raise AssetError("asset_timeout")
                        chunk = source.read(min(1048576, spec["bytes"] + 1 - total))
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > spec["bytes"]:
                            raise AssetError("download_integrity")
                        digest.update(chunk); output.write(chunk)
                    output.flush(); os.fsync(output.fileno())
                if total != spec["bytes"] or digest.hexdigest() != expected:
                    raise AssetError("download_integrity")
                try:
                    os.link(candidate, target, follow_symlinks=False)
                except FileExistsError:
                    if not verify(root, spec, deadline=deadline):
                        raise AssetError("assets_corrupt") from None
            except AssetError:
                raise
            except Exception:
                raise AssetError("download_failed") from None
    return root


def prepare(root, job=None):
    """Verify the local model snapshot without downloading or executing remote code."""
    root = Path(root)
    for spec in manifest()["artifacts"]:
        if not verify(root, spec):
            raise AssetError("assets_missing")
    return root
