"""
Ablation experiment runner for 3D face geometry PAD features.
"""

import pandas as pd
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
    if "split" not in df.columns:
        raise KeyError("DataFrame must contain 'split' column ('train'/'test').")

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    y_train = train_df[label_col].values
    y_test = test_df[label_col].values

    results = []

    for exp_name, feature_list in ABLATION_EXPERIMENTS.items():
        # Verify feature existence
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns for {exp_name}: {missing}")

        X_train = train_df[feature_list].values
        X_test = test_df[feature_list].values

        # Train pipeline on train_df ONLY
        pipeline = train_svm(
            X_train, y_train, random_state=random_state
        )

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = compute_pad_metrics(y_test, y_pred, y_prob)
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
