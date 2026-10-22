#!/usr/bin/env python3
"""Preflight the interpreter and assets used by external Deep3DFaceRecon."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


_PROBE = r"""
import importlib
import json
import sys

report = {"executable": sys.executable, "python": sys.version.split()[0], "modules": {}}
for name in ("torch", "torchvision", "kornia"):
    try:
        module = importlib.import_module(name)
        report["modules"][name] = {
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
            "path": getattr(module, "__file__", None),
        }
    except Exception as exc:
        report["modules"][name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(report))
sys.exit(0 if all(item["available"] for item in report["modules"].values()) else 2)
"""


def main():
    parser = argparse.ArgumentParser(
        description="Check the exact interpreter and assets before launching Deep3DFaceRecon."
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Interpreter used to launch test.py (defaults to this interpreter).",
    )
    parser.add_argument("--reconstruction-repo", required=True, help="Path to the Deep3DFaceRecon_pytorch checkout.")
    parser.add_argument("--checkpoint", help="Optional model checkpoint path to verify.")
    args = parser.parse_args()

    repo = Path(args.reconstruction_repo).expanduser()
    errors = []
    if not (repo / "test.py").is_file():
        errors.append(f"Missing reconstruction entrypoint: {repo / 'test.py'}")
    if args.checkpoint and not Path(args.checkpoint).expanduser().is_file():
        errors.append(f"Missing model checkpoint: {Path(args.checkpoint).expanduser()}")

    print(f"Interpreter requested: {args.python}")
    try:
        result = subprocess.run(
            [args.python, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: could not probe interpreter: {exc}")
        return 2

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print("Interpreter stderr:")
        print(result.stderr.strip())
    if result.returncode != 0:
        try:
            report = json.loads(result.stdout)
            missing = [name for name, module in report["modules"].items() if not module["available"]]
        except (ValueError, KeyError, TypeError):
            missing = ["torch", "torchvision", "kornia"]
        install_cmd = shlex.join([args.python, "-m", "pip", "install", *missing])
        errors.append(
            "Required imports are missing in the selected interpreter. Install the missing "
            f"packages into that same interpreter, for example: {install_cmd}"
        )

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("PASS: reconstruction entrypoint, requested assets, and torch/torchvision/kornia imports are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
