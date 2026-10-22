#!/usr/bin/env python3
"""
Comprehensive Final Analysis, Figure Generator, and Report Builder for BTP Part 2.

Tasks:
  1. Analyze 5 Ablation Experiments (Depth, Depth+Gradient, Curvature, Depth+Curvature, All 6 Features).
  2. Perform Subject-Disjoint Robustness Check (Secondary Split).
  3. Granular Failure Analysis (FP/FN, Subjects, Attack Types, Feature Distributions).
  4. Generate 9 Publication-Ready Figures in outputs/final_figures/.
  5. Save final_experiment_summary.json and final_results.csv.
  6. Write FINAL_ANALYSIS.md report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure src/ is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from face_recon_occlusion.data import (
    create_subject_disjoint_split,
    normalize_id,
    summarize_split,
    verify_subject_disjoint,
)
from face_recon_occlusion.evaluation import (
    analyze_failures,
    compute_pad_metrics,
    run_ablation_experiments,
)
from face_recon_occlusion.models import train_svm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BTP Part 2 Final Analysis & Reports")
    parser.add_argument(
        "--features-csv",
        type=str,
        default="outputs/oulu_geometry_features.csv",
        help="Input CSV containing extracted geometry features.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/final_analysis",
        help="Output directory for final summary, CSVs, and figures.",
    )
    parser.add_argument(
        "--seed-primary",
        type=int,
        default=42,
        help="Primary subject-disjoint split seed.",
    )
    parser.add_argument(
        "--seed-robustness",
        type=int,
        default=100,
        help="Secondary robustness subject-disjoint split seed.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Explicitly permit generating mock synthetic features for code verification if input CSV is missing.",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Force recomputation of primary split even if 'split' column is present in features CSV.",
    )
    return parser.parse_args()


def generate_pipeline_diagram(save_path):
    """Figure 1: Overall System Architecture Pipeline Diagram"""
    fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
    ax.axis("off")

    stages = [
        "Input Image\n(OULU-NPU)",
        "BiSeNet Parsing &\nInpainting",
        "Deep3DFaceRecon\n3D Reconstruction",
        "3D Surface Mesh\n(.OBJ Coordinate Space)",
        "6 Geometric Features\n(Depth, Gradient, Curvature)",
        "Subject-Disjoint\nRBF-SVM Classifier",
        "PAD Decision\n(Bona Fide vs Spoof)",
    ]

    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#2b5c8f", lw=1.5)

    for i, stage in enumerate(stages):
        x = i * 2.2
        ax.text(x, 0.5, stage, ha="center", va="center", size=9, weight="bold", bbox=bbox_props)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=((i + 1) * 2.2 - 0.7, 0.5),
                xytext=(x + 0.7, 0.5),
                arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=2),
            )

    ax.set_xlim(-1, len(stages) * 2.2 - 1)
    ax.set_ylim(0, 1)
    plt.title("Figure 1: Geometric 3D Surface Cue-Based Face Presentation Attack Detection Pipeline", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_ablation_plot(ablation_df, save_path):
    """Figure 6: Ablation Study Performance Comparison"""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.set_style("whitegrid")

    experiments = ablation_df["experiment"].values
    metrics = ["acer", "apcer", "bpcer", "roc_auc", "f1"]
    metric_labels = ["ACER (Lower is better)", "APCER", "BPCER", "ROC-AUC", "F1-Score"]

    x = np.arange(len(experiments))
    width = 0.15

    colors = ["#d95f02", "#7570b3", "#e7298a", "#66a61e", "#1b9e77"]

    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        values = ablation_df[m].values
        rects = ax.bar(x + i * width, values, width, label=label, color=colors[i])
        for rect in rects:
            height = rect.get_height()
            if np.isfinite(height):
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_ylabel("Score / Error Rate")
    ax.set_title("Figure 6: Feature Ablation Study Metric Comparison (Pilot-100)", fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(["Exp 1\n(Depth)", "Exp 2\n(Depth+Grad)", "Exp 3\n(Curvature)", "Exp 4\n(Depth+Curv)", "Exp 5\n(All 6 Features)"], fontsize=9)
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_confusion_matrices(cm_primary, cm_robust, save_path):
    """Figure 7: Confusion Matrices for Primary & Robustness Splits"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    for ax, cm, title in zip(axes, [cm_primary, cm_robust], ["Primary Split (Seed 42)", "Robustness Split (Seed 100)"]):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Bona Fide (0)", "Spoof (1)"],
            yticklabels=["Bona Fide (0)", "Spoof (1)"],
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title, fontweight="bold")

    plt.suptitle("Figure 7: Subject-Disjoint Baseline Confusion Matrices", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_roc_curves(y_true1, y_prob1, y_true2, y_prob2, save_path):
    """Figure 8: ROC Curves for Primary and Robustness Splits"""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    fpr1, tpr1, _ = roc_curve(y_true1, y_prob1)
    roc_auc1 = auc(fpr1, tpr1)

    ax.plot(fpr1, tpr1, color="#1b9e77", lw=2, label=f"Primary Split (AUC = {roc_auc1:.4f})")

    if y_prob2 is not None and len(np.unique(y_true2)) > 1:
        fpr2, tpr2, _ = roc_curve(y_true2, y_prob2)
        roc_auc2 = auc(fpr2, tpr2)
        ax.plot(fpr2, tpr2, color="#d95f02", lw=2, linestyle="--", label=f"Robustness Split (AUC = {roc_auc2:.4f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle=":")
    ax.set_xlabel("False Positive Rate (BPCER)")
    ax.set_ylabel("True Positive Rate (1 - APCER)")
    ax.set_title("Figure 8: Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_failure_plots(df_test, failures, save_path):
    """Figure 9: Failure Analysis & Feature Value Distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=300)

    # Plot A: Per-attack APCER
    if "per_attack_breakdown" in failures and failures["per_attack_breakdown"]:
        atk_data = failures["per_attack_breakdown"]
        atks = list(atk_data.keys())
        apcers = [atk_data[a]["apcer"] for a in atks]

        axes[0].bar(atks, apcers, color="#7570b3")
        axes[0].set_xlabel("OULU Attack Type ID")
        axes[0].set_ylabel("APCER (Missed Attack Rate)")
        axes[0].set_title("APCER by OULU Attack Type", fontweight="bold")
        axes[0].set_ylim(0, 1.0)
        for i, val in enumerate(apcers):
            axes[0].text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "Per-attack Metadata\nNot Available", ha="center", va="center")

    # Plot B: Feature Boxplot (Failed vs Correct)
    df_test_copy = df_test.copy()
    df_test_copy["Status"] = df_test_copy.apply(
        lambda r: "Correct" if r["label"] == r["y_pred"] else ("FP" if r["label"] == 0 else "FN"), axis=1
    )

    sns.boxplot(data=df_test_copy, x="Status", y="depth_std", hue="Status", palette="Set2", legend=False, ax=axes[1])
    axes[1].set_title("depth_std Distribution by Prediction Status", fontweight="bold")
    axes[1].set_ylabel("depth_std")

    plt.suptitle("Figure 9: Failure Analysis Diagnostic Breakdown", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_feature_distributions(df, save_path):
    """Figure 2: Compare each geometric feature by label."""
    feature_columns = [
        "depth_std", "depth_range", "gradient_mean",
        "gradient_std", "gaussian_curvature_mean",
        "gaussian_curvature_std",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=220)
    for ax, feature in zip(axes.flatten(), feature_columns):
        sns.boxplot(data=df, x="label", y=feature, ax=ax)
        ax.set_title(feature.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Label (0 = Bona Fide, 1 = Spoof)")
        ax.set_ylabel(feature)
    fig.suptitle("Figure 2: Geometry Feature Distributions by Class (Pilot-100)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_feature_correlation(df, save_path):
    """Figure 3: Pairwise correlation of the six geometry features."""
    feature_columns = [
        "depth_std", "depth_range", "gradient_mean",
        "gradient_std", "gaussian_curvature_mean",
        "gaussian_curvature_std",
    ]
    corr = df[feature_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Figure 3: Correlation Between Geometry Features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_split_class_distribution(df_primary, df_robust, save_path):
    """Figure 4: Verify class representation across train/test partitions."""
    import pandas as pd

    records = []
    for name, frame in [("Primary (seed 42)", df_primary), ("Robustness (seed 100)", df_robust)]:
        for split_name in ("train", "test"):
            part = frame[frame["split"] == split_name]
            counts = part["label"].value_counts()
            for label in (0, 1):
                records.append({
                    "evaluation": name,
                    "split": split_name,
                    "label": str(label),
                    "count": int(counts.get(label, 0)),
                })
    plot_df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=220, sharey=True)
    for ax, eval_name in zip(axes, ["Primary (seed 42)", "Robustness (seed 100)"]):
        subset = plot_df[plot_df["evaluation"] == eval_name]
        sns.barplot(data=subset, x="split", y="count", hue="label", ax=ax)
        ax.set_title(eval_name, fontweight="bold")
        ax.set_xlabel("Partition")
        ax.set_ylabel("Number of samples")
    fig.suptitle("Figure 4: Class Counts Across Subject-Disjoint Splits", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def generate_subject_split_overview(df_primary, df_robust, save_path):
    """Figure 5: Subject counts across primary and robustness partitions."""
    import pandas as pd

    records = []
    for name, frame in [("Primary (seed 42)", df_primary), ("Robustness (seed 100)", df_robust)]:
        for split_name in ("train", "test"):
            part = frame[frame["split"] == split_name]
            records.append({
                "evaluation": name,
                "split": split_name,
                "subjects": int(part["subject_id"].nunique()),
                "samples": int(len(part)),
            })
    plot_df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=220)
    for ax, metric, ylabel in [
        (axes[0], "subjects", "Unique subjects"),
        (axes[1], "samples", "Samples"),
    ]:
        sns.barplot(data=plot_df, x="evaluation", y=metric, hue="split", ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=10)
    fig.suptitle("Figure 5: Subject and Sample Counts by Split", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def make_json_safe(value):
    """Convert NumPy/Pandas values and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {str(make_json_safe(k)): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [make_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return make_json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_final_report(summary, ablation_df, output_dir):
    """Write a self-contained, cautious Markdown report beside generated results."""
    primary = summary["baseline_metrics_primary"]
    robust = summary["baseline_metrics_robustness"]
    primary_split = summary["primary_split"]
    robust_split = summary["robustness_split"]
    failures = summary["failure_analysis"]
    synthetic = str(summary.get("dataset", "")).lower().startswith("synthetic")

    def fmt(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{number:.4f}" if np.isfinite(number) else "N/A"

    ablation_lines = [
        "| Experiment | Features | Accuracy | APCER | BPCER | ACER | ROC-AUC | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ablation_df.iterrows():
        ablation_lines.append(
            f"| {row['experiment']} | {int(row['num_features'])} | "
            f"{fmt(row['accuracy'])} | {fmt(row['apcer'])} | "
            f"{fmt(row['bpcer'])} | {fmt(row['acer'])} | "
            f"{fmt(row['roc_auc'])} | {fmt(row['f1'])} |"
        )

    report = f"""# BTP Part 2 — Geometry-Based Face Presentation-Attack Detection

**Dataset label:** {summary.get("dataset", "OULU-NPU Pilot-100")}  
**Protocol:** subject-disjoint pilot split; not an official OULU-NPU protocol result  
**Timestamp:** {summary.get("timestamp", "not recorded")}

> {'**Synthetic software-validation run only. These metrics are fabricated test outputs and must not be reported as experimental findings.**' if synthetic else '**Research caution:** this is a small pilot evaluation. It does not establish generalizable face anti-spoofing or liveness performance.'}

## 1. Baseline results

| Metric | Primary split (seed 42) | Robustness split (seed 100) |
|---|---:|---:|
| Accuracy | {fmt(primary.get("accuracy"))} | {fmt(robust.get("accuracy"))} |
| APCER | {fmt(primary.get("apcer"))} | {fmt(robust.get("apcer"))} |
| BPCER | {fmt(primary.get("bpcer"))} | {fmt(robust.get("bpcer"))} |
| ACER | {fmt(primary.get("acer"))} | {fmt(robust.get("acer"))} |
| ROC-AUC | {fmt(primary.get("roc_auc"))} | {fmt(robust.get("roc_auc"))} |
| Precision | {fmt(primary.get("precision"))} | {fmt(robust.get("precision"))} |
| Recall | {fmt(primary.get("recall"))} | {fmt(robust.get("recall"))} |
| F1 | {fmt(primary.get("f1"))} | {fmt(robust.get("f1"))} |

Primary split: {primary_split.get("n_train_samples")} training / {primary_split.get("n_test_samples")} test samples; {primary_split.get("n_train_subjects")} training / {primary_split.get("n_test_subjects")} held-out subjects.

Robustness split: {robust_split.get("n_train_samples")} training / {robust_split.get("n_test_samples")} test samples; {robust_split.get("n_train_subjects")} training / {robust_split.get("n_test_subjects")} held-out subjects.

## 2. Five feature ablations

{chr(10).join(ablation_lines)}

These rankings describe this particular pilot and split. They do not establish which feature group will generalize best to other subjects, devices, environments, or attack media.

## 3. Failure analysis

- False positives: {failures.get("n_false_positives", "N/A")}
- False negatives: {failures.get("n_false_negatives", "N/A")}
- False-positive subject IDs: {", ".join(map(str, failures.get("false_positive_subjects", []))) or "None"}
- False-negative subject IDs: {", ".join(map(str, failures.get("false_negative_subjects", []))) or "None"}

## 4. Generated figures

The `final_figures/` directory should contain nine PNG files, numbered `01` through `09`: pipeline; feature distributions; feature correlation; split class balance; subject/sample split overview; ablation comparison; confusion matrices; ROC curves; and failure analysis.

## 5. Limitations

1. The 100-sample pilot is not an official OULU-NPU protocol evaluation.
2. The two subject-disjoint splits provide only preliminary evidence; independent protocol-based evaluation is required.
3. Reconstructed mesh geometry can contain errors and smoothing artifacts from the upstream monocular reconstruction model.
4. Six global scalar features may miss local geometric cues.
5. A successful software run verifies execution, not the scientific validity or real-world effectiveness of the PAD method.
6. Do not claim production-ready liveness detection from these results.

## 6. Reproducibility and outputs

The input feature CSV, split seeds, experiment summary, five-ablation results, failure analysis, and figure outputs are saved alongside this report. See `final_experiment_summary.json` for the machine-readable metrics and `final_results.csv` for ablation results.
"""
    report_path = Path(output_dir) / "FINAL_ANALYSIS.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main():
    args = parse_args()
    features_csv = Path(args.features_csv)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "final_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    synthetic_generated = False

    if not features_csv.exists():
        if not args.allow_synthetic:
            raise FileNotFoundError(
                f"Features CSV not found: {features_csv}. "
                f"To generate mock synthetic data for code testing only, explicitly pass --allow-synthetic."
            )
        print(f"⚠️ WARNING: Generating synthetic mock Pilot-100 dataset (--allow-synthetic set). This is for code testing only, NOT real experimental results.")
        synthetic_generated = True
        np.random.seed(42)
        n_samples = 100
        subjects = np.array([f"sub_{(i % 20) + 1:02d}" for i in range(n_samples)])
        labels = np.array([1 if (i % 5) != 0 else 0 for i in range(n_samples)])

        depth_std = np.random.normal(0.112, 0.02, n_samples) + labels * 0.015
        depth_range = np.random.normal(0.478, 0.05, n_samples) + labels * 0.03
        gradient_mean = np.random.normal(0.572, 0.08, n_samples) - labels * 0.02
        gradient_std = np.random.normal(0.628, 0.09, n_samples) + labels * 0.01
        curvature_mean = np.random.normal(-0.003, 0.001, n_samples) + labels * 0.0005
        curvature_std = np.random.normal(0.045, 0.008, n_samples) + labels * 0.003

        attack_ids = [str((i % 4) + 2) if labels[i] == 1 else "0" for i in range(n_samples)]
        device_ids = [str((i % 6) + 1) for i in range(n_samples)]

        df = pd.DataFrame({
            "sample_id": [f"{i+1}_{labels[i]+1}_{(i%20)+1}_{(i%6)+1}_1" for i in range(n_samples)],
            "subject_id": subjects,
            "label": labels,
            "attack_id": attack_ids,
            "device_id": device_ids,
            "depth_std": depth_std,
            "depth_range": depth_range,
            "gradient_mean": gradient_mean,
            "gradient_std": gradient_std,
            "gaussian_curvature_mean": curvature_mean,
            "gaussian_curvature_std": curvature_std,
        })
        features_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(features_csv, index=False)
        print(f"✅ Saved synthetic Pilot-100 feature CSV to {features_csv}")

    print("==========================================")
    print("BTP PART 2 FINAL ANALYSIS & REPORT BUILDER")
    print("==========================================")
    print("Note: Subject-disjoint pilot split evaluation (not an official OULU-NPU protocol)")
    print(f"Input CSV    : {features_csv}")
    print(f"Output Dir   : {output_dir}")
    print(f"Figures Dir  : {figures_dir}")
    print("==========================================")

    df = pd.read_csv(features_csv)

    features_list = [
        "depth_std",
        "depth_range",
        "gradient_mean",
        "gradient_std",
        "gaussian_curvature_mean",
        "gaussian_curvature_std",
    ]

    # SECTION 1: PRIMARY SPLIT & ABLATION STUDY
    print("\n📊 Executing Primary Split (Seed 42) & 5 Ablation Experiments...")
    if "split" in df.columns and not args.force_resplit:
        print("ℹ️ Reusing existing primary 'split' column from input CSV to guarantee split consistency across all evaluations.")
        df_primary = df
    else:
        df_primary = create_subject_disjoint_split(df, subject_col="subject_id", label_col="label", test_ratio=0.25, random_state=args.seed_primary)

    train_primary = df_primary[df_primary["split"] == "train"]
    test_primary = df_primary[df_primary["split"] == "test"]
    verify_subject_disjoint(train_primary, test_primary, subject_col="subject_id")

    ablation_df = run_ablation_experiments(df_primary, subject_col="subject_id", label_col="label", random_state=args.seed_primary)

    # Primary baseline model (extract continuous decision scores for ROC-AUC)
    svm_primary = train_svm(train_primary[features_list].values, train_primary["label"].values, random_state=args.seed_primary)
    y_pred_prim = svm_primary.predict(test_primary[features_list].values)
    y_score_prim = svm_primary.decision_function(test_primary[features_list].values)
    classes_prim = list(getattr(svm_primary, "classes_", [0, 1]))
    if classes_prim == [1, 0]:
        y_score_prim = -y_score_prim
    m_primary = compute_pad_metrics(test_primary["label"].values, y_pred_prim, y_score=y_score_prim)

    # SECTION 2: ROBUSTNESS CHECK (SECONDARY SPLIT)
    print("\n🔄 Executing Robustness Check (Secondary Split, Seed 100)...")
    df_robust = create_subject_disjoint_split(df, subject_col="subject_id", label_col="label", test_ratio=0.25, random_state=args.seed_robustness)
    train_robust = df_robust[df_robust["split"] == "train"]
    test_robust = df_robust[df_robust["split"] == "test"]
    verify_subject_disjoint(train_robust, test_robust, subject_col="subject_id")

    svm_robust = train_svm(train_robust[features_list].values, train_robust["label"].values, random_state=args.seed_primary)
    y_pred_rob = svm_robust.predict(test_robust[features_list].values)
    y_score_rob = svm_robust.decision_function(test_robust[features_list].values) if hasattr(svm_robust, "decision_function") else None
    if y_score_rob is not None:
        classes_rob = list(getattr(svm_robust, "classes_", [0, 1]))
        if classes_rob == [1, 0]:
            y_score_rob = -y_score_rob
    m_robust = compute_pad_metrics(test_robust["label"].values, y_pred_rob, y_score=y_score_rob)

    # SECTION 3: FAILURE ANALYSIS
    print("\n🔍 Performing Granular Failure Analysis...")
    failures_primary = analyze_failures(
        test_primary,
        y_pred_prim,
        y_score_prim,
        subject_col="subject_id",
        attack_col="attack_id" if "attack_id" in test_primary.columns else None,
        device_col="device_id" if "device_id" in test_primary.columns else None,
    )

    # SECTION 4: FIGURE GENERATION
    print("\n🎨 Generating 9 Publication-Ready Figures...")
    generate_pipeline_diagram(figures_dir / "01_pipeline_diagram.png")
    generate_feature_distributions(df, figures_dir / "02_feature_distributions.png")
    generate_feature_correlation(df, figures_dir / "03_feature_correlation.png")
    generate_split_class_distribution(df_primary, df_robust, figures_dir / "04_class_distribution.png")
    generate_subject_split_overview(df_primary, df_robust, figures_dir / "05_subject_split_overview.png")
    generate_ablation_plot(ablation_df, figures_dir / "06_ablation_comparison.png")
    generate_confusion_matrices(np.array(m_primary["confusion_matrix"]), np.array(m_robust["confusion_matrix"]), figures_dir / "07_confusion_matrix.png")
    generate_roc_curves(test_primary["label"].values, y_score_prim, test_robust["label"].values, y_score_rob, figures_dir / "08_roc_curve.png")

    test_primary_copy = test_primary.copy()
    test_primary_copy["y_pred"] = y_pred_prim
    generate_failure_plots(test_primary_copy, failures_primary, figures_dir / "09_failure_analysis.png")

    print(f"✅ Saved figures to {figures_dir}")

    # SECTION 5: MACHINE READABLE SUMMARY & CSV
    final_results_df = ablation_df.copy()
    final_results_df.to_csv(output_dir / "final_results.csv", index=False)

    summary_json = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "Synthetic mock data (software validation only)" if synthetic_generated else "OULU-NPU Pilot-100 (not official protocol)",
        "protocol": "subject_disjoint_pilot_split (NOT official OULU-NPU protocol evaluation)",
        "score_type": "svm_decision_function (distance to hyperplane, positive class = 1/Spoof)",
        "total_samples": len(df),
        "total_subjects": df["subject_id"].nunique(),
        "primary_split": summarize_split(df_primary, subject_col="subject_id", label_col="label"),
        "robustness_split": summarize_split(df_robust, subject_col="subject_id", label_col="label"),
        "baseline_metrics_primary": m_primary,
        "baseline_metrics_robustness": m_robust,
        "ablation_experiments": ablation_df.to_dict(orient="records"),
        "failure_analysis": failures_primary,
    }

    figure_files = sorted(path.name for path in figures_dir.glob("*.png"))
    expected_figures = [f"{index:02d}_" for index in range(1, 10)]
    if len(figure_files) != 9 or any(not any(name.startswith(prefix) for name in figure_files) for prefix in expected_figures):
        raise RuntimeError(f"Expected 9 numbered figure files, found {len(figure_files)}: {figure_files}")
    summary_json["figure_files"] = figure_files
    summary_json["synthetic_data_used"] = synthetic_generated
    summary_json["output_dir"] = str(output_dir)

    summary_path = output_dir / "final_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary_json), f, indent=2, allow_nan=False)

    report_path = write_final_report(summary_json, ablation_df, output_dir)

    print(f"✅ Saved summary JSON: {summary_path}")
    print(f"✅ Saved final results CSV: {output_dir / 'final_results.csv'}")
    print(f"✅ Saved Markdown report: {report_path}")
    print(f"✅ Verified {len(figure_files)} publication-ready figure files.")


if __name__ == "__main__":
    main()
