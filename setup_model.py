"""Inspect the pinned local Qwen snapshot or explicitly download its public assets."""

import argparse
import json
from pathlib import Path
import sys
import types

package = types.ModuleType("qwen_setup_owned")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules[package.__name__] = package
from qwen_setup_owned.assets_store import AssetError, download_public, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--download-public", action="store_true",
        help="Explicitly download public pinned weights after a disk-space preflight.")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()
    try:
        if args.download_public:
            download_public(args.directory, timeout_sec=args.timeout_sec, progress=lambda path: print(path, file=sys.stderr))
        print(json.dumps(inventory(args.directory), indent=2))
    except (AssetError, OSError) as exc:
        code = str(exc) if isinstance(exc, AssetError) else "asset_io"
        print(json.dumps({"error": code}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
