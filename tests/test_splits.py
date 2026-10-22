"""
Tests for subject-disjoint dataset splitting, metadata normalization, and split verification.
"""

import numpy as np
import pandas as pd
import pytest

from face_recon_occlusion.data.splits import (
    create_subject_disjoint_split,
    normalize_id,
    summarize_split,
    verify_subject_disjoint,
)
from face_recon_occlusion.evaluation.ablation import run_ablation_experiments


def test_normalize_id():
    """Verify that normalize_id handles various extensions and trailing whitespace."""
    assert normalize_id("sample_01.jpg") == "sample_01"
    assert normalize_id("sample_02.PNG") == "sample_02"
    assert normalize_id(" sample_03.obj ") == "sample_03"
    assert normalize_id("sample_04.mat") == "sample_04"
    assert normalize_id("sample_05") == "sample_05"
    assert normalize_id(np.nan) == ""


def test_verify_subject_disjoint_success():
    """Verify that disjoint subject sets pass without error."""
    train_df = pd.DataFrame({"subject_id": ["sub_01", "sub_02", "sub_03"]})
    test_df = pd.DataFrame({"subject_id": ["sub_04", "sub_05"]})
    assert verify_subject_disjoint(train_df, test_df) is True


def test_verify_subject_disjoint_raises_on_leakage():
    """Verify that any overlapping subject between train and test raises ValueError."""
    train_df = pd.DataFrame({"subject_id": ["sub_01", "sub_02", "sub_03"]})
    test_df = pd.DataFrame({"subject_id": ["sub_03", "sub_04"]})

    with pytest.raises(ValueError, match="Subject identity leak detected"):
        verify_subject_disjoint(train_df, test_df)


def test_create_subject_disjoint_split():
    """Verify that create_subject_disjoint_split strictly partitions subjects and balances classes."""
    n_samples = 100
    subjects = [f"sub_{(i % 20) + 1:02d}" for i in range(n_samples)]
    labels = [1 if (i % 5) != 0 else 0 for i in range(n_samples)]

    df = pd.DataFrame({
        "sample_id": [f"id_{i}" for i in range(n_samples)],
        "subject_id": subjects,
        "label": labels,
    })

    split_df = create_subject_disjoint_split(df, test_ratio=0.25, random_state=42)

    assert "split" in split_df.columns
    train_df = split_df[split_df["split"] == "train"]
    test_df = split_df[split_df["split"] == "test"]

    # Verify zero leakage
    assert verify_subject_disjoint(train_df, test_df) is True

    # Verify test_ratio: with 20 subjects, 25% is 5 subjects in test, 15 in train
    assert test_df["subject_id"].nunique() == 5
    assert train_df["subject_id"].nunique() == 15

    # Verify both classes exist in both splits
    assert set(train_df["label"].unique()) == {0, 1}
    assert set(test_df["label"].unique()) == {0, 1}


def test_shared_split_consistency_across_ablations():
    """
    Verify that all 5 ablation feature groups evaluate on the exact same subject
    partition and test samples without any re-splitting.
    """
    rng = np.random.RandomState(42)
    n_samples = 100
    df = pd.DataFrame({
        "sample_id": [f"s_{i}" for i in range(n_samples)],
        "subject_id": [f"sub_{(i % 20) + 1:02d}" for i in range(n_samples)],
        "label": [1 if (i % 5) != 0 else 0 for i in range(n_samples)],
        "depth_std": rng.uniform(0.08, 0.15, n_samples),
        "depth_range": rng.uniform(0.40, 0.60, n_samples),
        "gradient_mean": rng.uniform(0.45, 0.65, n_samples),
        "gradient_std": rng.uniform(0.40, 0.70, n_samples),
        "gaussian_curvature_mean": rng.uniform(-0.01, 0.01, n_samples),
        "gaussian_curvature_std": rng.uniform(0.03, 0.07, n_samples),
    })

    split_df = create_subject_disjoint_split(df, test_ratio=0.25, random_state=42)
    test_samples_before = sorted(split_df[split_df["split"] == "test"]["sample_id"].tolist())

    res_df = run_ablation_experiments(split_df, random_state=42)

    # 5 experiments must be executed
    assert len(res_df) == 5
    # Total evaluated test samples (tn + fp + fn + tp) must be identical across all experiments
    for _, row in res_df.iterrows():
        total_eval = row["tn"] + row["fp"] + row["fn"] + row["tp"]
        assert total_eval == len(test_samples_before)


def test_create_split_rejects_invalid_ratio():
    df = pd.DataFrame({
        "subject_id": ["a", "a", "b", "b"],
        "label": [0, 1, 0, 1],
    })
    for ratio in (0, 1, -0.1, 1.2, np.nan):
        with pytest.raises(ValueError, match="test_ratio"):
            create_subject_disjoint_split(df, test_ratio=ratio)


def test_create_split_rejects_missing_subject_ids():
    df = pd.DataFrame({
        "subject_id": ["a", None, "b", "b"],
        "label": [0, 1, 0, 1],
    })
    with pytest.raises(ValueError, match="missing or blank subject IDs"):
        create_subject_disjoint_split(df)


def test_create_split_rejects_impossible_class_coverage():
    # Only one subject contains the bona-fide class, so both partitions cannot contain both labels.
    df = pd.DataFrame({
        "subject_id": ["a", "b", "c", "d"],
        "label": [0, 1, 1, 1],
    })
    with pytest.raises(ValueError, match="Could not create a subject-disjoint split"):
        create_subject_disjoint_split(df)


def test_ablation_rejects_subject_leakage():
    rng = np.random.RandomState(8)
    df = pd.DataFrame({
        "subject_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
        "split": ["train", "test", "train", "train", "test", "test", "train", "test"],
        "depth_std": rng.rand(8),
        "depth_range": rng.rand(8),
        "gradient_mean": rng.rand(8),
        "gradient_std": rng.rand(8),
        "gaussian_curvature_mean": rng.rand(8),
        "gaussian_curvature_std": rng.rand(8),
    })
    with pytest.raises(ValueError, match="Subject identity leak detected"):
        run_ablation_experiments(df)


def test_ablation_rejects_non_finite_features():
    rng = np.random.RandomState(4)
    df = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(8)],
        "subject_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
        "split": ["train", "train", "train", "train", "test", "test", "test", "test"],
        "depth_std": rng.rand(8),
        "depth_range": rng.rand(8),
        "gradient_mean": rng.rand(8),
        "gradient_std": rng.rand(8),
        "gaussian_curvature_mean": rng.rand(8),
        "gaussian_curvature_std": rng.rand(8),
    })
    df.loc[0, "depth_std"] = np.nan
    with pytest.raises(ValueError, match="Non-finite or non-numeric"):
        run_ablation_experiments(df)
