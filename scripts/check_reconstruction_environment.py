#!/usr/bin/env python3
"""Preflight the exact interpreter and assets used by external Deep3DFaceRecon."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


_REQUIRED_MODULES = ("torch", "torchvision", "kornia", "dominate")
_SAFE_PACKAGE_SPECS = {
    "kornia": "kornia>=0.7",
    "dominate": "dominate>=2.9",
}

_PROBE = r"""
import importlib
import json
import sys

report = {"executable": sys.executable, "python": sys.version.split()[0], "modules": {}}
for name in ("torch", "torchvision", "kornia", "dominate"):
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


def _last_json_line(stdout):
    for line in reversed(stdout.splitlines()):
        try:
            return json.loads(line)
        except (ValueError, TypeError):
            continue
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check the exact interpreter and assets before launching Deep3DFaceRecon."
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Interpreter used to launch test.py (defaults to this interpreter).",
    )
    parser.add_argument(
        "--reconstruction-repo", required=True,
        help="Path to the Deep3DFaceRecon_pytorch checkout.",
    )
    parser.add_argument("--checkpoint", help="Optional model checkpoint path to verify.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run test.py --help after imports pass; this checks the import/CLI path but does not process images.",
    )
    args = parser.parse_args(argv)

    repo = Path(args.reconstruction_repo).expanduser()
    entrypoint = repo / "test.py"
    errors = []
    if not entrypoint.is_file():
        errors.append(f"Missing reconstruction entrypoint: {entrypoint}")
    if args.checkpoint and not Path(args.checkpoint).expanduser().is_file():
        errors.append(f"Missing model checkpoint: {Path(args.checkpoint).expanduser()}")

    print(f"Interpreter requested: {args.python}")
    try:
        result = subprocess.run(
            [args.python, "-c", _PROBE],
            cwd=str(repo) if repo.is_dir() else None,
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
        print("Interpreter stderr (tail):")
        print(result.stderr[-4000:])

    report = _last_json_line(result.stdout)
    module_status = report.get("modules", {}) if isinstance(report, dict) else {}
    missing = [
        name for name in _REQUIRED_MODULES
        if not module_status.get(name, {}).get("available", False)
    ]
    imports_ok = result.returncode == 0 and not missing

    if missing:
        repairable = [name for name in missing if name in _SAFE_PACKAGE_SPECS]
        core = [name for name in missing if name not in _SAFE_PACKAGE_SPECS]
        if repairable:
            specs = [_SAFE_PACKAGE_SPECS[name] for name in repairable]
            install_cmd = shlex.join([args.python, "-m", "pip", "install", *specs])
            errors.append(
                f"Missing helper module(s): {', '.join(repairable)}. "
                f"Install these into the same interpreter: {install_cmd}"
            )
        if core:
            errors.append(
                f"Core PyTorch import(s) missing or broken: {', '.join(core)}. "
                "Do not blindly reinstall torch/torchvision; check that their versions "
                "match the active Python and Colab runtime."
            )
    elif not isinstance(report, dict):
        errors.append("The interpreter probe did not return valid JSON; inspect the interpreter output above.")
    elif result.returncode != 0:
        errors.append(f"The probe returned exit code {result.returncode} even though its report showed no missing modules.")

    smoke_result = None
    if args.smoke_test:
        if imports_ok and entrypoint.is_file():
            try:
                smoke_result = subprocess.run(
                    [args.python, str(entrypoint), "--help"],
                    cwd=str(repo),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"test.py --help smoke test could not complete: {exc}")
            else:
                if smoke_result.stdout.strip():
                    print("test.py --help stdout (tail):")
                    print(smoke_result.stdout[-6000:])
                if smoke_result.stderr.strip():
                    print("test.py --help stderr (tail):")
                    print(smoke_result.stderr[-4000:])
                if smoke_result.returncode != 0:
                    errors.append(
                        f"test.py --help smoke test failed with exit code {smoke_result.returncode}; "
                        "review the output above before retrying reconstruction."
                    )
                else:
                    print("PASS: test.py --help completed without launching inference.")
        else:
            print("SKIP: --smoke-test requires a valid entrypoint and successful imports.")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    checked = "torch/torchvision/kornia/dominate imports"
    if args.smoke_test:
        checked += " and the test.py CLI smoke test"
    print(f"PASS: reconstruction entrypoint, requested assets, and {checked} are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
