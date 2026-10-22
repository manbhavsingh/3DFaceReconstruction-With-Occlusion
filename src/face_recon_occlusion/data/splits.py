"""
Subject-disjoint splitting and metadata validation module for PAD experiments.
"""

import numpy as np
import pandas as pd


def normalize_id(sid):
    """Normalize sample ID string by stripping whitespace and common extension suffixes."""
    if pd.isna(sid):
        return ""
    s = str(sid).strip()
    for ext in [".jpg", ".jpeg", ".png", ".obj", ".mat"]:
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
    return s


def _validate_subject_ids(series, split_name):
    if series.isna().any() or series.astype(str).str.strip().eq("").any():
        raise ValueError(f"{split_name} contains missing or blank subject IDs.")


def verify_subject_disjoint(train_df, test_df, subject_col="subject_id"):
    """Verify that non-empty train and test partitions have no shared subject IDs."""
    for name, frame in (("train", train_df), ("test", test_df)):
        if subject_col not in frame.columns:
            raise KeyError(f"Missing required subject column '{subject_col}' in {name} split.")
        if frame.empty:
            raise ValueError(f"{name} split is empty.")
        _validate_subject_ids(frame[subject_col], name)

    train_subjects = set(train_df[subject_col].astype(str).str.strip())
    test_subjects = set(test_df[subject_col].astype(str).str.strip())
    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        raise ValueError(
            f"CRITICAL ERROR: Subject identity leak detected! "
            f"Overlapping subjects ({len(overlap)}): {sorted(overlap)}"
        )
    return True


def create_subject_disjoint_split(
    df, subject_col="subject_id", label_col="label", test_ratio=0.25, random_state=42
):
    """
    Create a reproducible subject-disjoint split with both binary classes in each partition.

    Raises ValueError when IDs or labels are invalid or no feasible subject partition
    can place both bona-fide (0) and spoof (1) samples in train and test.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    missing = [name for name in (subject_col, label_col) if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    try:
        test_ratio = float(test_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("test_ratio must be a number strictly between 0 and 1.") from exc
    if not np.isfinite(test_ratio) or not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be a finite number strictly between 0 and 1.")

    subjects = df[subject_col]
    _validate_subject_ids(subjects, "dataset")
    labels = pd.to_numeric(df[label_col], errors="coerce")
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError("label column must contain both binary labels 0 (bona fide) and 1 (spoof), with no missing values.")

    subject_values = subjects.astype(str).str.strip()
    unique_subjects = np.array(sorted(subject_values.unique()))
    if len(unique_subjects) < 2:
        raise ValueError("At least two distinct subjects are required for a subject-disjoint split.")

    n_test = max(1, min(len(unique_subjects) - 1, int(len(unique_subjects) * test_ratio)))
    chosen_split = None
    for seed_offset in range(100):
        rng = np.random.RandomState(int(random_state) + seed_offset)
        shuffled = unique_subjects.copy()
        rng.shuffle(shuffled)
        test_subjects = set(shuffled[:n_test])
        candidate = pd.Series(
            np.where(subject_values.isin(test_subjects), "test", "train"),
            index=df.index,
        )
        train_labels = set(labels[candidate == "train"].unique())
        test_labels = set(labels[candidate == "test"].unique())
        if train_labels == {0, 1} and test_labels == {0, 1}:
            chosen_split = candidate
            break

    if chosen_split is None:
        raise ValueError(
            "Could not create a subject-disjoint split containing both labels in train "
            "and test after 100 deterministic attempts. Check class coverage across subjects "
            "or choose a different test_ratio."
        )

    result = df.copy()
    result["split"] = chosen_split
    verify_subject_disjoint(
        result[result["split"] == "train"],
        result[result["split"] == "test"],
        subject_col=subject_col,
    )
    return result


def summarize_split(
    df,
    subject_col="subject_id",
    label_col="label",
    attack_col=None,
    device_col=None,
):
    """Report statistics for a validated train/test partition."""
    if "split" not in df.columns:
        raise KeyError("DataFrame must contain a 'split' column ('train'/'test').")
    if df["split"].isna().any() or set(df["split"].unique()) != {"train", "test"}:
        raise ValueError("split column must contain only 'train' and 'test', with both present.")

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    verify_subject_disjoint(train_df, test_df, subject_col=subject_col)

    summary = {
        "n_total": len(df),
        "n_train_samples": len(train_df),
        "n_test_samples": len(test_df),
        "n_train_subjects": train_df[subject_col].nunique(),
        "n_test_subjects": test_df[subject_col].nunique(),
        "train_class_dist": train_df[label_col].value_counts().to_dict(),
        "test_class_dist": test_df[label_col].value_counts().to_dict(),
    }
    if attack_col and attack_col in df.columns:
        summary["train_attack_dist"] = train_df[attack_col].value_counts().to_dict()
        summary["test_attack_dist"] = test_df[attack_col].value_counts().to_dict()
    if device_col and device_col in df.columns:
        summary["train_device_dist"] = train_df[device_col].value_counts().to_dict()
        summary["test_device_dist"] = test_df[device_col].value_counts().to_dict()
    return summary
