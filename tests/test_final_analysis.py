import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def test_final_analysis_cli_generates_complete_synthetic_smoke_outputs(tmp_path):
    """End-to-end software smoke test; these are mock data, not research findings."""
    repo_root = Path(__file__).resolve().parents[1]
    feature_csv = tmp_path / "synthetic_features.csv"
    output_dir = tmp_path / "analysis_output"

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "generate_final_analysis.py"),
        "--features-csv", str(feature_csv),
        "--output-dir", str(output_dir),
        "--seed-primary", "42",
        "--seed-robustness", "100",
        "--allow-synthetic",
        "--force-resplit",
    ]
    result = subprocess.run(
        command, cwd=repo_root, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (
        f"Analysis command failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    assert feature_csv.is_file()
    assert (output_dir / "final_results.csv").is_file()
    assert (output_dir / "final_experiment_summary.json").is_file()
    assert (output_dir / "FINAL_ANALYSIS.md").is_file()

    with (output_dir / "final_experiment_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["synthetic_data_used"] is True
    assert summary["dataset"].lower().startswith("synthetic")
    assert len(summary["ablation_experiments"]) == 5
    assert len(summary["figure_files"]) == 9

    figures_dir = output_dir / "final_figures"
    figures = sorted(p.name for p in figures_dir.glob("*.png"))
    assert len(figures) == 9
    for prefix in [f"{i:02d}_" for i in range(1, 10)]:
        assert any(name.startswith(prefix) for name in figures), (prefix, figures)

    with (output_dir / "final_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5

    report = (output_dir / "FINAL_ANALYSIS.md").read_text(encoding="utf-8")
    assert "Synthetic software-validation run only" in report
    assert "not an official OULU-NPU protocol" in report

def test_synthetic_provenance_flag(tmp_path):
    """Test that a repeated invocation reads the synthetic flag from an existing CSV."""
    repo_root = Path(__file__).resolve().parents[1]
    feature_csv = tmp_path / "synthetic_features.csv"
    output_dir1 = tmp_path / "analysis_output1"
    output_dir2 = tmp_path / "analysis_output2"

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    # First invocation: generate the synthetic data
    command1 = [
        sys.executable,
        str(repo_root / "scripts" / "generate_final_analysis.py"),
        "--features-csv", str(feature_csv),
        "--output-dir", str(output_dir1),
        "--allow-synthetic"
    ]
    subprocess.run(command1, cwd=repo_root, env=env, check=True, capture_output=True)

    # Second invocation: read the existing synthetic data
    command2 = [
        sys.executable,
        str(repo_root / "scripts" / "generate_final_analysis.py"),
        "--features-csv", str(feature_csv),
        "--output-dir", str(output_dir2)
    ]
    result2 = subprocess.run(
        command2, cwd=repo_root, env=env, capture_output=True, text=True
    )
    assert result2.returncode == 0

    with (output_dir2 / "final_experiment_summary.json").open(encoding="utf-8") as handle:
        summary2 = json.load(handle)

    assert summary2["synthetic_data_used"] is True
    assert summary2["dataset"].lower().startswith("synthetic")
