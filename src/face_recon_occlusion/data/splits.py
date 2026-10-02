"""
Subject-disjoint splitting and metadata validation module for PAD experiments.
"""

import numpy as np
import pandas as pd


def verify_subject_disjoint(train_df, test_df, subject_col="subject_id"):
    """
    Verify that train and test subject sets are strictly disjoint.

    Args:
        train_df (pd.DataFrame): Training subset.
        test_df (pd.DataFrame): Testing subset.
        subject_col (str): Subject ID column name.

    Raises:
        ValueError: If any subject ID overlaps between train and test splits.
    """
    train_subjects = set(train_df[subject_col].unique())
    test_subjects = set(test_df[subject_col].unique())

    overlap = train_subjects.intersection(test_subjects)
    if len(overlap) > 0:
        raise ValueError(
            f"CRITICAL ERROR: Subject identity leak detected! "
            f"Overlapping subjects ({len(overlap)}): {sorted(list(overlap))}"
        )

    return True


def create_subject_disjoint_split(
    df, subject_col="subject_id", label_col="label", test_ratio=0.3, random_state=42
):
    """
    Create a subject-disjoint train/test split. Ensures both splits contain bona fide and spoof samples.

    Args:
        df (pd.DataFrame): Full dataset metadata DataFrame.
        subject_col (str): Subject ID column.
        label_col (str, optional): Target label column for class-balanced subject splitting.
        test_ratio (float): Ratio of subjects to place in test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with an added 'split' column ('train' or 'test').
    """
    df = df.copy()
    unique_subjects = np.array(sorted(df[subject_col].unique()))

    rng = np.random.RandomState(random_state)
    
    # Try up to 100 seeds starting from random_state to ensure both classes exist in both splits
    for seed_offset in range(100):
        current_seed = random_state + seed_offset
        rng_trial = np.random.RandomState(current_seed)
        shuffled_subjects = unique_subjects.copy()
        rng_trial.shuffle(shuffled_subjects)

        n_test = max(1, int(len(unique_subjects) * test_ratio))
        test_subjects = set(shuffled_subjects[:n_test])
        train_subjects = set(shuffled_subjects[n_test:])

        temp_split = df[subject_col].apply(lambda s: "test" if s in test_subjects else "train")
        train_labels = df[temp_split == "train"][label_col].unique() if label_col in df.columns else [0, 1]
        test_labels = df[temp_split == "test"][label_col].unique() if label_col in df.columns else [0, 1]

        if len(train_labels) > 1 and len(test_labels) > 1:
            break

    df["split"] = temp_split

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    verify_subject_disjoint(train_df, test_df, subject_col=subject_col)

    return df



def summarize_split(
    df,
    subject_col="subject_id",
    label_col="label",
    attack_col=None,
    device_col=None,
):
    """
    Report statistics for a dataset split.

    Args:
        df (pd.DataFrame): DataFrame containing 'split' column.
        subject_col (str): Subject ID column.
        label_col (str): Label column (0=Bona Fide, 1=Spoof).
        attack_col (str, optional): Attack type column.
        device_col (str, optional): Device/camera column.

    Returns:
        dict: Detailed statistics dictionary.
    """
    if "split" not in df.columns:
        raise KeyError("DataFrame must contain a 'split' column ('train'/'test').")

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
