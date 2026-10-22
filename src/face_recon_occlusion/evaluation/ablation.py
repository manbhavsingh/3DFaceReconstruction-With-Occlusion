"""
Ablation experiment runner for 3D face geometry PAD features.
"""

import numpy as np
import pandas as pd
from ..data.splits import verify_subject_disjoint
from ..models.svm import train_svm
from .pad_metrics import compute_pad_metrics


ABLATION_EXPERIMENTS = {
    "Exp_1_Depth_Only": [
        "depth_std",
        "depth_range",
    ],
    "Exp_2_Depth_Gradient": [
        "depth_std",
        "depth_range",
        "gradient_mean",
        "gradient_std",
    ],
    "Exp_3_Curvature_Only": [
        "gaussian_curvature_mean",
        "gaussian_curvature_std",
    ],
    "Exp_4_Depth_Curvature": [
        "depth_std",
        "depth_range",
        "gaussian_curvature_mean",
        "gaussian_curvature_std",
    ],
    "Exp_5_Full_Baseline": [
        "depth_std",
        "depth_range",
        "gradient_mean",
        "gradient_std",
        "gaussian_curvature_mean",
        "gaussian_curvature_std",
    ],
}


def run_ablation_experiments(df, subject_col="subject_id", label_col="label", random_state=42):
    """
    Run all 5 required ablation experiments on a subject-disjoint dataset.

    Args:
        df (pd.DataFrame): DataFrame containing geometry features, label, and 'split' ('train'/'test').
        subject_col (str): Subject ID column.
        label_col (str): Target label column (0=Bona Fide, 1=Spoof).
        random_state (int): Seed for SVM reproducibility.

    Returns:
        pd.DataFrame: Summary table of metrics across all 5 experiments.
    """
    required = [subject_col, label_col, "split"]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise KeyError(f"Missing required columns: {missing_required}")
    if df.empty or df["split"].isna().any() or set(df["split"].unique()) != {"train", "test"}:
        raise ValueError("split column must contain only 'train' and 'test', with both present.")

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    verify_subject_disjoint(train_df, test_df, subject_col=subject_col)

    y_train = pd.to_numeric(train_df[label_col], errors="coerce").to_numpy()
    y_test = pd.to_numeric(test_df[label_col], errors="coerce").to_numpy()
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError("Labels contain missing or non-numeric values.")
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Both train and test partitions must contain binary labels 0 and 1.")

    results = []

    for exp_name, feature_list in ABLATION_EXPERIMENTS.items():
        # Verify feature existence
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns for {exp_name}: {missing}")

        X_train = train_df[feature_list].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        X_test = test_df[feature_list].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
            raise ValueError(f"Non-finite or non-numeric values found for {exp_name}.")

        # Train pipeline on train_df ONLY
        pipeline = train_svm(
            X_train, y_train, random_state=random_state
        )

        y_pred = pipeline.predict(X_test)

        # Extract continuous decision scores for ROC-AUC ranking evaluation.
        # In scikit-learn binary SVC, decision_function corresponds to classes_[1].
        # Verify alignment with positive class 1 (Spoof).
        y_score = None
        if hasattr(pipeline, "decision_function"):
            decision = pipeline.decision_function(X_test)
            classes = list(getattr(pipeline, "classes_", [0, 1]))
            if classes == [0, 1]:
                y_score = decision
            elif classes == [1, 0]:
                y_score = -decision
            else:
                y_score = decision
        elif hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_pad_metrics(y_test, y_pred, y_score=y_score)
        metrics["experiment"] = exp_name
        metrics["num_features"] = len(feature_list)
        metrics["feature_list"] = ", ".join(feature_list)

        results.append(metrics)

    res_df = pd.DataFrame(results)
    ordered_cols = [
        "experiment", "num_features", "accuracy", "apcer", "bpcer", "acer",
        "roc_auc", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "feature_list"
    ]
    return res_df[ordered_cols]
