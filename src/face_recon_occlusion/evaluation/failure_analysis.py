"""
Failure analysis and error breakdown module for geometry-based face PAD.
"""

import numpy as np
import pandas as pd


def analyze_failures(
    test_df,
    y_pred,
    y_prob=None,
    subject_col="subject_id",
    attack_col=None,
    device_col=None,
):
    """
    Perform granular failure analysis on test predictions.

    Args:
        test_df (pd.DataFrame): Test subset DataFrame.
        y_pred (np.ndarray): Model predictions (0=Bona Fide, 1=Spoof).
        y_prob (np.ndarray, optional): Spoof probabilities.
        subject_col (str): Subject ID column.
        attack_col (str, optional): Attack type column.
        device_col (str, optional): Device/camera column.

    Returns:
        dict: Breakdown of false positives, false negatives, per-attack rates, and per-device rates.
    """
    df = test_df.copy()
    df["y_pred"] = y_pred
    if y_prob is not None:
        df["y_prob"] = y_prob

    # False Positives: True Bona Fide (0) misclassified as Spoof (1)
    fp_df = df[(df["label"] == 0) & (df["y_pred"] == 1)]

    # False Negatives: True Spoof (1) misclassified as Bona Fide (0)
    fn_df = df[(df["label"] == 1) & (df["y_pred"] == 0)]

    analysis = {
        "n_test_total": len(df),
        "n_false_positives": len(fp_df),
        "n_false_negatives": len(fn_df),
        "false_positive_subjects": sorted(list(fp_df[subject_col].unique())),
        "false_negative_subjects": sorted(list(fn_df[subject_col].unique())),
    }

    # Per-attack breakdown
    if attack_col and attack_col in df.columns:
        attack_analysis = {}
        spoof_subset = df[df["label"] == 1]
        for atk, grp in spoof_subset.groupby(attack_col):
            total_atk = len(grp)
            fn_atk = len(grp[grp["y_pred"] == 0]) # Missed attacks
            apcer_atk = fn_atk / total_atk if total_atk > 0 else 0.0
            attack_analysis[str(atk)] = {
                "total_samples": total_atk,
                "missed_attacks": fn_atk,
                "apcer": apcer_atk,
            }
        analysis["per_attack_breakdown"] = attack_analysis

    # Per-device breakdown
    if device_col and device_col in df.columns:
        device_analysis = {}
        for dev, grp in df.groupby(device_col):
            n_dev = len(grp)
            n_err = len(grp[grp["label"] != grp["y_pred"]])
            err_rate = n_err / n_dev if n_dev > 0 else 0.0
            device_analysis[str(dev)] = {
                "total_samples": n_dev,
                "errors": n_err,
                "error_rate": err_rate,
            }
        analysis["per_device_breakdown"] = device_analysis

    return analysis
