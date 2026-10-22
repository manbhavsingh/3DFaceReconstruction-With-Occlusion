import json
import subprocess
import sys

from scripts import check_reconstruction_environment as checker


def _report(**availability):
    modules = {}
    versions = {"torch": "2.0.0", "torchvision": "0.15.0", "kornia": "0.7.0", "dominate": "2.9.0"}
    for name, available in availability.items():
        modules[name] = (
            {"available": True, "version": versions[name], "path": f"/venv/{name}.py"}
            if available
            else {"available": False, "error": "ModuleNotFoundError"}
        )
    return {"executable": sys.executable, "python": "3.11.0", "modules": modules}


def _completed(command, returncode, *, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def test_missing_dominate_suggests_install_into_same_interpreter(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "Deep3DFaceRecon_pytorch"
    repo.mkdir()
    (repo / "test.py").write_text("")

    probe = _report(torch=True, torchvision=True, kornia=True, dominate=False)
    monkeypatch.setattr(
        checker.subprocess, "run",
        lambda command, **kwargs: _completed(command, 2, stdout=json.dumps(probe)),
    )

    result = checker.main(["--python", "/usr/bin/python3", "--reconstruction-repo", str(repo)])
    output = capsys.readouterr().out

    assert result == 1
    assert "Missing helper module(s): dominate" in output
    assert "/usr/bin/python3 -m pip install 'dominate>=2.9'" in output
    assert "pip install torch " not in output


def test_torch_stack_failure_does_not_recommend_blind_reinstall(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "Deep3DFaceRecon_pytorch"
    repo.mkdir()
    (repo / "test.py").write_text("")

    probe = _report(torch=False, torchvision=False, kornia=True, dominate=True)
    monkeypatch.setattr(
        checker.subprocess, "run",
        lambda command, **kwargs: _completed(command, 2, stdout=json.dumps(probe)),
    )

    result = checker.main(["--python", "/usr/bin/python3", "--reconstruction-repo", str(repo)])
    output = capsys.readouterr().out

    assert result == 1
    assert "Core PyTorch import(s) missing or broken: torch, torchvision" in output
    assert "Do not blindly reinstall torch/torchvision" in output
    assert "pip install torch torchvision" not in output


def test_smoke_test_runs_test_py_help_only_after_imports_pass(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "Deep3DFaceRecon_pytorch"
    repo.mkdir()
    entrypoint = repo / "test.py"
    entrypoint.write_text("")

    probe = _report(torch=True, torchvision=True, kornia=True, dominate=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "-c" in command:
            return _completed(command, 0, stdout=json.dumps(probe))
        if "--help" in command:
            return _completed(command, 0, stdout="usage: test.py [options]")
        raise AssertionError(f"Unexpected subprocess command: {command}")

    monkeypatch.setattr(checker.subprocess, "run", fake_run)
    result = checker.main([
        "--python", "/usr/bin/python3",
        "--reconstruction-repo", str(repo),
        "--smoke-test",
    ])
    output = capsys.readouterr().out

    assert result == 0
    assert len(calls) == 2
    assert calls[1][0] == ["/usr/bin/python3", str(entrypoint), "--help"]
    assert calls[1][1]["cwd"] == str(repo)
    assert "without launching inference" in output
    assert "PASS: reconstruction entrypoint" in output


def test_smoke_test_failure_returns_nonzero(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "Deep3DFaceRecon_pytorch"
    repo.mkdir()
    (repo / "test.py").write_text("")

    probe = _report(torch=True, torchvision=True, kornia=True, dominate=True)
    call_count = 0

    def fake_run(command, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _completed(command, 0, stdout=json.dumps(probe))
        return _completed(command, 1, stderr="ModuleNotFoundError: No module named 'another_dependency'")

    monkeypatch.setattr(checker.subprocess, "run", fake_run)
    result = checker.main([
        "--reconstruction-repo", str(repo),
        "--smoke-test",
    ])
    output = capsys.readouterr().out

    assert result == 1
    assert "test.py --help smoke test failed with exit code 1" in output
    assert "another_dependency" in output
