"""Tests for restart-safe geometry feature extraction checkpoints."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_geometry_extraction.py"
SPEC = importlib.util.spec_from_file_location("run_geometry_extraction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(sample_id="sample-1"):
    return {
        "sample_id": sample_id,
        "depth_std": 0.1,
        "depth_range": 0.4,
        "gradient_mean": 0.6,
        "gradient_std": 0.2,
        "gaussian_curvature_mean": 0.01,
        "gaussian_curvature_std": 0.03,
    }


def test_save_progress_writes_complete_csv_without_temp_files(tmp_path):
    output = tmp_path / "features.csv"
    MODULE.save_progress([_row()], output)
    frame = pd.read_csv(output)
    assert frame.loc[0, "sample_id"] == "sample-1"
    assert frame.loc[0, "gradient_mean"] == pytest.approx(0.6)
    assert list(tmp_path.iterdir()) == [output]


def test_load_cache_rejects_missing_required_columns(tmp_path):
    output = tmp_path / "features.csv"
    pd.DataFrame([{"sample_id": "sample-1"}]).to_csv(output, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        MODULE.load_cache(output)


def test_load_cache_rejects_non_finite_features(tmp_path):
    output = tmp_path / "features.csv"
    row = _row()
    row["depth_std"] = float("nan")
    pd.DataFrame([row]).to_csv(output, index=False)
    with pytest.raises(ValueError, match="non-finite/non-numeric"):
        MODULE.load_cache(output)


def test_load_cache_normalizes_sample_id_extensions(tmp_path):
    output = tmp_path / "features.csv"
    pd.DataFrame([_row("sample-1.obj")]).to_csv(output, index=False)
    cache = MODULE.load_cache(output)
    assert "sample-1" in cache
