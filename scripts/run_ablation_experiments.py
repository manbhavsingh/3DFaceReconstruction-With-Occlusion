#!/usr/bin/env python3
"""
Reproducible Subject-Disjoint Ablation Experiment Runner for 3D Face Geometry PAD.

Executes all 5 required ablation experiments:
  1. Depth Only
  2. Depth + Gradient
  3. Curvature Only
  4. Depth + Curvature
  5. Depth + Gradient + Curvature (Full Baseline)

Saves:
  - experiment_config.json
  - results.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# Ensure src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_recon_occlusion.data import (
    create_subject_disjoint_split,
    summarize_split,
    verify_subject_disjoint,
)
from face_recon_occlusion.evaluation import analyze_failures, run_ablation_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D Geometry PAD Ablation Experiments")
    parser.add_argument(
        "--features-csv",
        type=str,
        required=True,
        help="Input CSV containing extracted geometry features and metadata.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Optional dataset metadata CSV to merge if subject_id or label is missing in features CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablation_results",
        help="Directory to save experiment_config.json and results.csv.",
    )
    parser.add_argument(
        "--subject-col",
        type=str,
        default="subject_id",
        help="Subject ID column name.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Target label column (0=Bona Fide, 1=Spoof).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def normalize_id(sid):
    if pd.isna(sid):
        return ""
    s = str(sid).strip()
    for ext in [".jpg", ".jpeg", ".png", ".obj", ".mat"]:
        if s.lower().endswith(ext):
            s = s[:-len(ext)]
    return s


def main():
    args = parse_args()
    features_csv = Path(args.features_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        print(f"Error: Input features CSV not found: {features_csv}")
        sys.exit(1)

    print("==========================================")
    print("3D FACE GEOMETRY PAD ABLATION EXPERIMENTS")
    print("==========================================")
    print(f"Input CSV    : {features_csv}")
    print(f"Output Dir   : {output_dir}")
    print(f"Random Seed  : {args.seed}")
    print("==========================================")

    df = pd.read_csv(features_csv)

    # Backfill metadata if subject_col or label_col is missing
    if (args.subject_col not in df.columns or args.label_col not in df.columns) and args.metadata_csv:
        meta_path = Path(args.metadata_csv)
        if meta_path.exists():
            print(f"ℹ️ '{args.subject_col}' or '{args.label_col}' missing in features CSV. Merging metadata from {meta_path}...")
            meta_df = pd.read_csv(meta_path)
            
            id_col = None
            for candidate in ["sample_id", "filename", "image_path", "name", "id"]:
                if candidate in meta_df.columns:
                    id_col = candidate
                    break

            if id_col and "sample_id" in df.columns:
                df["_norm_id"] = df["sample_id"].apply(normalize_id)
                meta_df["_norm_id"] = meta_df[id_col].apply(normalize_id)
                cols_to_merge = [c for c in [args.subject_col, args.label_col, "attack_id", "device_id"] if c in meta_df.columns and c not in df.columns]
                df = df.merge(meta_df[["_norm_id"] + cols_to_merge], on="_norm_id", how="left").drop(columns=["_norm_id"])

    # Validate column presence
    if args.subject_col not in df.columns:
        raise KeyError(
            f"Missing required column '{args.subject_col}' in DataFrame! "
            f"Available columns: {list(df.columns)}. "
            f"Please ensure metadata CSV is provided or feature extraction includes subject_id."
        )

    if args.label_col not in df.columns:
        raise KeyError(
            f"Missing required column '{args.label_col}' in DataFrame! "
            f"Available columns: {list(df.columns)}."
        )

    # Enforce subject-disjoint split if 'split' column is not already present
    if "split" not in df.columns:
        print("Creating subject-disjoint train/test split...")
        df = create_subject_disjoint_split(
            df, subject_col=args.subject_col, test_ratio=0.3, random_state=args.seed
        )

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # Verify Subject Disjointness
    print("\n🔍 Verifying Subject Disjointness...")
    verify_subject_disjoint(train_df, test_df, subject_col=args.subject_col)
    print("✅ Verified: train_subjects ∩ test_subjects = EMPTY SET.")

    # Report Data Summaries
    summary = summarize_split(df, subject_col=args.subject_col, label_col=args.label_col)
    print("\nData Split Summary:")
    print(f"  Train Samples  : {summary['n_train_samples']} (Subjects: {summary['n_train_subjects']})")
    print(f"  Test Samples   : {summary['n_test_samples']} (Subjects: {summary['n_test_subjects']})")
    print(f"  Train Class Dist : {summary['train_class_dist']}")
    print(f"  Test Class Dist  : {summary['test_class_dist']}")

    # Save Experiment Config
    config = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "input_csv": str(features_csv),
        "subject_col": args.subject_col,
        "label_col": args.label_col,
        "classifier": "StandardScaler + RBF SVC(C=1.0, class_weight='balanced')",
        "split_summary": summary,
    }

    config_path = output_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config: {config_path}")

    # Execute 5 Ablation Experiments
    print("\n🚀 Executing 5 Ablation Experiments...")
    ablation_df = run_ablation_experiments(
        df, subject_col=args.subject_col, label_col=args.label_col, random_state=args.seed
    )

    results_path = output_dir / "results.csv"
    ablation_df.to_csv(results_path, index=False)

    print("\n==========================================================================================")
    print("ABLATION EXPERIMENT RESULTS SUMMARY")
    print("==========================================================================================")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(ablation_df[["experiment", "num_features", "accuracy", "apcer", "bpcer", "acer", "roc_auc", "f1"]])
    print("==========================================================================================")
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
