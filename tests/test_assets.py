"""Offline preparation tests only; these do not qualify real Qwen synthesis."""

import hashlib
import io
import os
from pathlib import Path
import sys
import json
import subprocess
from tempfile import TemporaryDirectory
import time
import types
import unittest
from unittest.mock import patch
from urllib.request import Request

PACKAGE = Path(__file__).resolve().parents[1]
package = types.ModuleType("qwen_assets_test")
package.__path__ = [str(PACKAGE)]
sys.modules[package.__name__] = package
from qwen_assets_test import assets_store as assets


def spec(path, content=b"safe", *, gated=False, git=False):
    digest = hashlib.sha1(f"blob {len(content)}\0".encode()+content) if git else hashlib.sha256(content)
    return {"path":path,"bytes":len(content),"git_blob_sha1" if git else "sha256":digest.hexdigest(),
        "access":"user_prepared_gated" if gated else "public",
        "url":"https://huggingface.co/example/resolve/immutable/file"}


class AssetTests(unittest.TestCase):
    def test_qualification_helper_default_is_read_only(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)/"absent"
            result=subprocess.run([sys.executable,"-B",str(PACKAGE/"setup_model.py"),
                "--directory",str(root)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(result.stdout)["ready"])
            self.assertFalse(root.exists())

    def test_manifest(self):
        model = assets.manifest()
        self.assertEqual(len(model["artifacts"]),11)
        self.assertEqual(sum(s["bytes"] for s in model["artifacts"]),4520159586)
        self.assertEqual(len({s["path"] for s in model["artifacts"]}),11)
        gated = [s for s in model["artifacts"] if s["access"]=="user_prepared_gated"]
        self.assertEqual(len(gated),0)
        for item in model["artifacts"]:
            self.assertGreater(item["bytes"],0)
            self.assertEqual(set(item)&{"sha256","git_blob_sha1"},{"sha256"} if "sha256" in item else {"git_blob_sha1"})
            self.assertTrue(item["url"].startswith("https://"))
            self.assertNotIn("/main/",item["url"])
            self.assertNotIn("/../",item["path"])
        self.assertEqual(gated, [])

    def test_inventory_is_read_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)/"not-created"
            result = assets.inventory(root)
            self.assertFalse(root.exists())
            self.assertFalse(result["ready"])
            self.assertEqual(len(result["missing_public"]),11)
            self.assertEqual(len(result["missing_authorized"]),0)

    def test_integrity_and_unsafe_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); target = root/"asset"
            item = spec("asset"); self.assertFalse(assets.verify(root,item))
            target.write_bytes(b"safe")
            self.assertTrue(assets.verify(root,item))
            self.assertTrue(assets.verify(root,spec("asset",git=True)))
            with self.assertRaisesRegex(assets.AssetError,"asset_timeout"):
                assets.verify(root,item,deadline=time.monotonic()-1)
            target.write_bytes(b"evil")
            with self.assertRaisesRegex(assets.AssetError,"assets_corrupt"):
                assets.verify(root,item)
            linked=root/"linked";linked.symlink_to(target)
            for path in ("linked","../escape","/absolute"):
                with self.assertRaises(assets.AssetError):
                    assets.verify(root,spec(path))
            fifo=root/"pipe";os.mkfifo(fifo)
            with self.assertRaises(assets.AssetError):
                assets.verify(root,spec("pipe"))

    def test_no_network_before_access_and_space_gates(self):
        with TemporaryDirectory() as directory, patch.object(assets,"build_opener") as network:
            root=Path(directory)
            with patch.object(assets,"manifest",return_value={"artifacts":[spec("model")]}), patch.object(
                    assets.shutil,"disk_usage",return_value=types.SimpleNamespace(free=0)):
                with self.assertRaisesRegex(assets.AssetError,"disk_space"):
                    assets.download_public(root)
                self.assertFalse(list(root.iterdir()))
            network.assert_not_called()

    def test_verified_download_and_existing_preservation(self):
        items=[spec("engine/module.py",git=True),spec("tokenizer/config")]
        with TemporaryDirectory() as directory, patch.object(assets,"manifest",return_value={"artifacts":items}):
            root=Path(directory);(root/"tokenizer").mkdir();(root/"tokenizer/config").write_bytes(b"safe")
            opener=types.SimpleNamespace(open=lambda *a,**kw:io.BytesIO(b"safe"))
            with patch.object(assets,"build_opener",return_value=opener) as network:
                assets.download_public(root)
                network.assert_called_once()
            self.assertTrue(assets.inventory(root)["ready"])
            self.assertFalse(list(root.glob(".qwen-download-*")))
            with patch.object(assets,"build_opener") as network, patch.object(assets.shutil,"disk_usage",return_value=types.SimpleNamespace(free=0)):
                assets.download_public(root)
                network.assert_not_called()
            (root/"engine/module.py").write_bytes(b"BAD!")
            with patch.object(assets,"build_opener") as network:
                with self.assertRaisesRegex(assets.AssetError,"assets_corrupt"):
                    assets.download_public(root)
                network.assert_not_called()
            self.assertEqual((root/"engine/module.py").read_bytes(),b"BAD!")

    def test_bad_download_never_published(self):
        for body in (b"bad",b"evil",b"too long"):
            with self.subTest(body=body), TemporaryDirectory() as directory, patch.object(
                    assets,"manifest",return_value={"artifacts":[spec("model")]}):
                root=Path(directory)
                with patch.object(assets,"build_opener",return_value=types.SimpleNamespace(open=lambda *a,**kw:io.BytesIO(body))):
                    with self.assertRaisesRegex(assets.AssetError,"download_integrity"):
                        assets.download_public(root)
                self.assertFalse((root/"model").exists())
                self.assertFalse(list(root.glob(".qwen-download-*")))

    def test_redirect_policy(self):
        redirect=assets.PinnedRedirects();request=Request("https://huggingface.co/asset")
        for target in ("http://huggingface.co/file","https://huggingface.co.evil.invalid/file","https://other.invalid/file","https://user:pass@huggingface.co/file"):
            with self.subTest(target=target), self.assertRaises(assets.AssetError):
                redirect.redirect_request(request,None,302,"",{},target)
        self.assertIsNotNone(redirect.redirect_request(request,None,302,"",{},"https://cdn-lfs.hf.co/file"))


if __name__ == "__main__":
    unittest.main()
