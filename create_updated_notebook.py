import json
from pathlib import Path

def make_cell(cell_type, lines):
    formatted_lines = [l if l.endswith('\n') else l + '\n' for l in lines]
    return {
        'cell_type': cell_type,
        'metadata': {},
        'source': formatted_lines
    }

cells = [
    # HEADER
    make_cell('markdown', [
        '# BTP Part 2: 3D Face Geometry-Based Face Presentation Attack Detection (PAD)',
        '## Complete Research Execution, Robustness, Ablation & Reporting Notebook (OULU Pilot-100)',
        '',
        '**Research Question:** *Can geometric information extracted from reconstructed 3D facial surfaces provide discriminative information for face presentation-attack detection?*',
        '',
        'This notebook orchestrates the complete research workflow: calling repository modules (`src/face_recon_occlusion/`) and scripts (`scripts/`), performing multi-split robustness evaluation, failure analysis, generating publication-ready figures, and exporting scientific summary files.',
        '',
        '---'
    ]),

    # SECTION 1
    make_cell('markdown', [
        '## SECTION 1 — Environment Setup',
        'Inspect Python environment, mount Google Drive (`/content/drive/MyDrive/`), and clone/pull the code repository.'
    ]),
    make_cell('code', [
        'import os',
        'import sys',
        'import subprocess',
        'from pathlib import Path',
        'from datetime import datetime',
        '',
        'try:',
        '    from IPython.display import display, Image',
        'except ImportError:',
        '    display = print',
        '',
        'print(f"Python Version   : {sys.version}")',
        'print(f"Current Directory: {Path.cwd()}")',
        'print(f"Timestamp        : {datetime.now().isoformat()}")'
    ]),
    make_cell('code', [
        '# Mount Google Drive (Colab Execution)',
        'try:',
        '    from google.colab import drive',
        '    drive.mount("/content/drive")',
        '    print("✅ Google Drive mounted successfully at /content/drive")',
        'except ImportError:',
        '    print("ℹ️ Running outside Google Colab environment. Assuming local filesystem storage.")'
    ]),
    make_cell('code', [
        '# Repository Management',
        'repo_dir = Path("/content/3DFaceReconstruction-With-Occlusion")',
        'if not repo_dir.exists():',
        '    repo_dir = Path.cwd()',
        '',
        'if (repo_dir / ".git").exists():',
        '    print(f"Repository detected at {repo_dir}. Pulling latest main branch...")',
        '    subprocess.run(["git", "pull", "origin", "main"], check=False)',
        'else:',
        '    print(f"Cloning repository to {repo_dir}...")',
        '    subprocess.run(["git", "clone", "https://github.com/manbhavsingh/3DFaceReconstruction-With-Occlusion", str(repo_dir)], check=True)',
        '',
        'print(f"✅ Current Working Directory: {Path.cwd()}")'
    ]),
    make_cell('code', [
        '# Verify Repository Directory Structure',
        'assert (repo_dir / "src").exists(), "Missing src/ directory in repository!"',
        'assert (repo_dir / "scripts").exists(), "Missing scripts/ directory in repository!"',
        'print("✅ Repository layout verified (src/ and scripts/ present).")'
    ]),

    # SECTION 2
    make_cell('markdown', [
        '## SECTION 2 — Dependency Verification',
        'Verify core scientific dependencies (numpy, pandas, scipy, trimesh, scikit-learn, matplotlib, seaborn, tqdm).'
    ]),
    make_cell('code', [
        'import numpy as np',
        'import pandas as pd',
        'import scipy',
        'import trimesh',
        'import sklearn',
        'import matplotlib',
        'import seaborn',
        'import tqdm',
        '',
        'print("Dependencies Verified:")',
        'print(f"  numpy        : {np.__version__}")',
        'print(f"  pandas       : {pd.__version__}")',
        'print(f"  scipy        : {scipy.__version__}")',
        'print(f"  trimesh      : {trimesh.__version__}")',
        'print(f"  scikit-learn : {sklearn.__version__}")',
        'print(f"  matplotlib   : {matplotlib.__version__}")',
        'print(f"  seaborn      : {seaborn.__version__}")',
        'print(f"  tqdm         : {tqdm.__version__}")',
        'print("✅ All required dependencies are available.")'
    ]),

    # SECTION 3
    make_cell('markdown', [
        '## SECTION 3 — Repository / Package Setup',
        'Make `src/` importable in `sys.path` and test importing package modules.'
    ]),
    make_cell('code', [
        '# Add src/ directory to sys.path',
        'src_dir = str(repo_dir / "src")',
        'if src_dir not in sys.path:',
        '    sys.path.insert(0, src_dir)',
        '',
        'from face_recon_occlusion.geometry import (',
        '    extract_depth_features,',
        '    extract_gradient_features,',
        '    extract_curvature_features,',
        '    extract_geometry_features',
        ')',
        'from face_recon_occlusion.data import create_subject_disjoint_split, verify_subject_disjoint, summarize_split',
        'from face_recon_occlusion.models import train_svm, build_svm_pipeline',
        'from face_recon_occlusion.evaluation import compute_pad_metrics, run_ablation_experiments, analyze_failures',
        '',
        'print("✅ Successfully imported repository modules:")',
        'print("   - face_recon_occlusion.geometry (depth, gradient, curvature, features)")',
        'print("   - face_recon_occlusion.data (splits, metadata summaries)")',
        'print("   - face_recon_occlusion.models (SVM pipeline)")',
        'print("   - face_recon_occlusion.evaluation (PAD metrics, ablation, failure analysis)")'
    ]),

    # SECTION 4
    make_cell('markdown', [
        '## SECTION 4 — Drive / Data Validation',
        'Validate dataset root directory, metadata CSV, and 3D OBJ reconstructions.'
    ]),
    make_cell('code', [
        'drive_root = Path("/content/drive/MyDrive/BTP_Part2_OULU/subject_disjoint_pilot_100")',
        'if not drive_root.exists():',
        '    drive_root = repo_dir / "outputs"',
        '',
        'meta_path = drive_root / "oulu_subject_disjoint_pilot_100.csv"',
        'if not meta_path.exists():',
        '    meta_path = drive_root / "oulu_geometry_features.csv"',
        'if not meta_path.exists():',
        '    meta_path = repo_dir / "outputs" / "oulu_geometry_features.csv"',
        '',
        'mesh_dir = drive_root / "reconstructions"',
        'if not mesh_dir.exists():',
        '    mesh_dir = repo_dir / "outputs" / "reconstructions"',
        '',
        'print(f"Data Root Path  : {drive_root} (Exists: {drive_root.exists()})")',
        'print(f"Metadata / CSV  : {meta_path} (Exists: {meta_path.exists()})")',
        'print(f"Reconstructions : {mesh_dir} (Exists: {mesh_dir.exists()})")',
        '',
        'assert drive_root.exists(), f"Data root directory not found: {drive_root}"',
        'assert meta_path.exists(), f"Metadata CSV not found: {meta_path}"',
        '',
        'if mesh_dir.exists():',
        '    obj_files = sorted(list(mesh_dir.rglob("*.obj")))',
        '    print(f"Total OBJ Meshes Found: {len(obj_files)}")',
        'else:',
        '    print("ℹ️ Reconstructions folder not present locally; feature extraction CSV will be loaded from cache.")',
        'print("✅ Data validation complete.")'
    ]),

    # SECTION 5
    make_cell('markdown', [
        '## SECTION 5 — Metadata Inspection',
        'Inspect metadata CSV columns, subject count, and class distribution.'
    ]),
    make_cell('code', [
        'meta_df = pd.read_csv(meta_path)',
        'print(f"Metadata Shape: {meta_df.shape}")',
        'print(f"Metadata Columns: {list(meta_df.columns)}\\n")',
        'print("First 5 Rows:")',
        'display(meta_df.head())',
        '',
        'print("\\nSubject Distribution:")',
        'print(f"  Total Unique Subjects: {meta_df[\'subject_id\'].nunique()}")',
        'print(f"  Subject IDs: {sorted(meta_df[\'subject_id\'].unique())}")',
        '',
        'print("\\nClass Distribution (0 = Bona Fide, 1 = Spoof):")',
        'print(meta_df["label"].value_counts().sort_index())',
        '',
        'for col in ["sample_id", "subject_id", "label"]:',
        '    assert col in meta_df.columns, f"CRITICAL: Missing required column \'{col}\' in metadata CSV!"',
        '',
        'print("\\n✅ Metadata fields verified successfully.")'
    ]),

    # SECTION 6
    make_cell('markdown', [
        '## SECTION 6 — SINGLE OBJ GEOMETRY TEST',
        'Sanity Check: Extract 6 geometric features from a sample OBJ mesh if present and verify finite non-NaN values.'
    ]),
    make_cell('code', [
        'features_to_check = [',
        '    "depth_std", "depth_range",',
        '    "gradient_mean", "gradient_std",',
        '    "gaussian_curvature_mean", "gaussian_curvature_std"',
        ']',
        '',
        'if "obj_files" in locals() and len(obj_files) > 0:',
        '    test_obj_path = obj_files[0]',
        '    print(f"Testing Geometry Extraction on Sample OBJ: {test_obj_path.name}")',
        '    test_feats = extract_geometry_features(test_obj_path)',
        '    print("\\n==========================================")',
        '    print("SINGLE OBJ GEOMETRY FEATURE OUTPUT")',
        '    print("==========================================")',
        '    for fname in features_to_check:',
        '        val = test_feats[fname]',
        '        print(f"{fname:25s}: {val:.8f}")',
        '        assert np.isfinite(val) and not np.isnan(val), f"Invalid feature value: {val}"',
        '    print("\\n✅ Single OBJ geometry extraction sanity test PASSED.")',
        'else:',
        '    print("ℹ️ Skipping single OBJ test since raw OBJ files are handled via pre-extracted CSV cache.")'
    ]),

    # SECTION 7
    make_cell('markdown', [
        '## SECTION 7 — CACHE-FIRST FEATURE EXTRACTION',
        'Execute batch geometry feature extraction script across dataset.'
    ]),
    make_cell('code', [
        'output_feature_csv = drive_root / "oulu_geometry_features.csv"',
        'if not output_feature_csv.exists():',
        '    output_feature_csv = repo_dir / "outputs" / "oulu_geometry_features.csv"',
        '',
        'if mesh_dir.exists() and len(list(mesh_dir.rglob("*.obj"))) > 0:',
        '    subprocess.run([',
        '        "python3", "scripts/run_geometry_extraction.py",',
        '        "--mesh-dir", str(mesh_dir),',
        '        "--metadata-csv", str(meta_path),',
        '        "--output-csv", str(output_feature_csv)',
        '    ], check=True)',
        'else:',
        '    print(f"ℹ️ Reusing cached feature CSV at: {output_feature_csv}")',
        '',
        'assert output_feature_csv.exists(), f"Feature CSV not found: {output_feature_csv}"',
        'print(f"✅ Feature extraction table ready: {output_feature_csv}")'
    ]),

    # SECTION 8
    make_cell('markdown', [
        '## SECTION 8 — FEATURE CSV VALIDATION & METADATA MERGE',
        'Load extracted feature table, normalize sample IDs, backfill metadata if needed, and verify data integrity.'
    ]),
    make_cell('code', [
        'feats_df = pd.read_csv(output_feature_csv)',
        'print(f"Loaded Feature Table Shape: {feats_df.shape}")',
        'print(f"Initial Columns: {list(feats_df.columns)}\\n")',
        '',
        'def norm_id(sid):',
        '    if pd.isna(sid): return ""',
        '    s = str(sid).strip()',
        '    for ext in [".jpg", ".jpeg", ".png", ".obj", ".mat"]:',
        '        if s.lower().endswith(ext):',
        '            s = s[:-len(ext)]',
        '    return s',
        '',
        'if "subject_id" not in feats_df.columns or "label" not in feats_df.columns:',
        '    print("ℹ️ Backfilling missing metadata (\'subject_id\', \'label\') into feature table...")',
        '    feats_df["_norm_id"] = feats_df["sample_id"].apply(norm_id)',
        '    meta_df["_norm_id"] = meta_df["sample_id"].apply(norm_id)',
        '    cols_to_add = [c for c in ["subject_id", "label", "attack_id", "device_id"] if c in meta_df.columns and c not in feats_df.columns]',
        '    feats_df = feats_df.merge(meta_df[["_norm_id"] + cols_to_add], on="_norm_id", how="left").drop(columns=["_norm_id"])',
        '    feats_df.to_csv(output_feature_csv, index=False)',
        '    print(f"✅ Updated feature table saved to {output_feature_csv}")',
        '',
        'for fname in features_to_check:',
        '    assert fname in feats_df.columns, f"Missing feature column: {fname}"',
        '    assert feats_df[fname].isna().sum() == 0, f"NaNs found in {fname}"',
        '',
        'print("Summary Statistics of Extracted Baseline 6 Features:")',
        'display(feats_df[features_to_check].describe())',
        'print("\\n✅ Feature CSV validation and metadata backfill PASSED.")'
    ]),

    # SECTION 9
    make_cell('markdown', [
        '## SECTION 9 — PRIMARY SUBJECT-DISJOINT SPLIT VALIDATION (Seed 42)',
        'Verify zero subject identity leakage between primary training and testing splits.'
    ]),
    make_cell('code', [
        'feats_df1 = create_subject_disjoint_split(feats_df.copy(), subject_col="subject_id", label_col="label", test_ratio=0.25, random_state=42)',
        '',
        'train_subset1 = feats_df1[feats_df1["split"] == "train"]',
        'test_subset1 = feats_df1[feats_df1["split"] == "test"]',
        '',
        'verify_subject_disjoint(train_subset1, test_subset1, subject_col="subject_id")',
        '',
        'split_summary1 = summarize_split(feats_df1, subject_col="subject_id", label_col="label")',
        'print("==========================================")',
        'print("PRIMARY SUBJECT-DISJOINT SPLIT (SEED 42)")',
        'print("==========================================")',
        'print(f"Train Samples  : {split_summary1[\'n_train_samples\']} (Subjects: {split_summary1[\'n_train_subjects\']})")',
        'print(f"Test Samples   : {split_summary1[\'n_test_samples\']} (Subjects: {split_summary1[\'n_test_subjects\']})")',
        'print(f"Train Subjects : {sorted(list(train_subset1[\'subject_id\'].unique()))}")',
        'print(f"Test Subjects  : {sorted(list(test_subset1[\'subject_id\'].unique()))}")',
        'print(f"Train Classes  : {split_summary1[\'train_class_dist\']}")',
        'print(f"Test Classes   : {split_summary1[\'test_class_dist\']}")',
        '',
        'overlap1 = set(train_subset1["subject_id"]).intersection(set(test_subset1["subject_id"]))',
        'print(f"Subject Overlap: {overlap1} (Count: {len(overlap1)})")',
        'print("\\n✅ Primary subject-disjoint split verified: ZERO subject identity leakage.")'
    ]),

    # SECTION 10
    make_cell('markdown', [
        '## SECTION 10 — BASELINE SVM EVALUATION (Seed 42)',
        'Train `StandardScaler` + `SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)` fitted strictly on primary training split.'
    ]),
    make_cell('code', [
        'X_train_full1 = train_subset1[features_to_check].values',
        'y_train_full1 = train_subset1["label"].values',
        'X_test_full1 = test_subset1[features_to_check].values',
        'y_test_full1 = test_subset1["label"].values',
        '',
        'svm_model1 = train_svm(X_train_full1, y_train_full1, C=1.0, kernel="rbf", class_weight="balanced", random_state=42)',
        'print("✅ Baseline RBF SVM pipeline fitted strictly on Primary X_train.")',
        '',
        'y_pred_baseline1 = svm_model1.predict(X_test_full1)',
        'y_prob_baseline1 = svm_model1.predict_proba(X_test_full1)[:, 1]'
    ]),

    # SECTION 11
    make_cell('markdown', [
        '## SECTION 11 — PRIMARY PAD METRICS COMPUTATION',
        'Compute ISO/IEC 30107-3 metrics (APCER, BPCER, ACER, ROC-AUC, Accuracy, Precision, Recall, F1, Confusion Matrix).'
    ]),
    make_cell('code', [
        'base_metrics1 = compute_pad_metrics(y_test_full1, y_pred_baseline1, y_prob_baseline1)',
        '',
        'print("==========================================")',
        'print("PRIMARY 6-FEATURE SVM EVALUATION RESULTS")',
        'print("==========================================")',
        'print(f"Accuracy   : {base_metrics1[\'accuracy\']:.4f}")',
        'print(f"APCER      : {base_metrics1[\'apcer\']:.4f} (Spoof False Acceptance Rate)")',
        'print(f"BPCER      : {base_metrics1[\'bpcer\']:.4f} (Bona Fide False Rejection Rate)")',
        'print(f"ACER       : {base_metrics1[\'acer\']:.4f} (Average Classification Error Rate)")',
        'print(f"ROC-AUC    : {base_metrics1[\'roc_auc\']:.4f}")',
        'print(f"Precision  : {base_metrics1[\'precision\']:.4f}")',
        'print(f"Recall     : {base_metrics1[\'recall\']:.4f}")',
        'print(f"F1-Score   : {base_metrics1[\'f1\']:.4f}")',
        'print(f"\\nConfusion Matrix (TN, FP, FN, TP):")',
        'print(f"  TN={base_metrics1[\'tn\']}, FP={base_metrics1[\'fp\']}, FN={base_metrics1[\'fn\']}, TP={base_metrics1[\'tp\']}")'
    ]),

    # SECTION 12
    make_cell('markdown', [
        '## SECTION 12 — FIVE ABLATION EXPERIMENTS & RESULTS TABLE DISPLAY',
        'Execute all 5 required ablation feature combinations using `scripts/run_ablation_experiments.py` and display the ACTUAL `ablation_results/results.csv` output table.'
    ]),
    make_cell('code', [
        'ablation_out_dir = drive_root / "ablation_results"',
        'if not ablation_out_dir.exists():',
        '    ablation_out_dir = repo_dir / "outputs" / "ablation_results"',
        '',
        'subprocess.run([',
        '    "python3", "scripts/run_ablation_experiments.py",',
        '    "--features-csv", str(output_feature_csv),',
        '    "--metadata-csv", str(meta_path),',
        '    "--output-dir", str(ablation_out_dir),',
        '    "--seed", "42"',
        '], check=True)',
        '',
        'ablation_results_file = ablation_out_dir / "results.csv"',
        'assert ablation_results_file.exists(), f"Ablation results file not found at {ablation_results_file}"',
        '',
        'ablation_results_df = pd.read_csv(ablation_results_file)',
        'print("\\n==========================================================================================")',
        'print("ACTUAL CONTENTS OF ablation_results/results.csv")',
        'print("==========================================================================================")',
        'display(ablation_results_df[["experiment", "num_features", "accuracy", "apcer", "bpcer", "acer", "roc_auc", "precision", "recall", "f1", "feature_list"]])',
        'print(f"\\n✅ Loaded and displayed actual 5 ablation results from: {ablation_results_file}")'
    ]),

    # SECTION 13
    make_cell('markdown', [
        '## SECTION 13 — SECONDARY SUBJECT-DISJOINT ROBUSTNESS EXPERIMENT (Seed 100)',
        'Perform an independent secondary subject-disjoint split using `random_state=100`. Print train/test subjects, sample counts, class distributions, verify ZERO subject overlap, and compute all PAD metrics and confusion matrix.'
    ]),
    make_cell('code', [
        'feats_df2 = create_subject_disjoint_split(feats_df.copy(), subject_col="subject_id", label_col="label", test_ratio=0.25, random_state=100)',
        'train_subset2 = feats_df2[feats_df2["split"] == "train"]',
        'test_subset2 = feats_df2[feats_df2["split"] == "test"]',
        '',
        'verify_subject_disjoint(train_subset2, test_subset2, subject_col="subject_id")',
        '',
        'split_summary2 = summarize_split(feats_df2, subject_col="subject_id", label_col="label")',
        'print("==========================================")',
        'print("SECONDARY ROBUSTNESS SPLIT (SEED 100)")',
        'print("==========================================")',
        'print(f"Train Samples  : {split_summary2[\'n_train_samples\']} (Subjects: {split_summary2[\'n_train_subjects\']})")',
        'print(f"Test Samples   : {split_summary2[\'n_test_samples\']} (Subjects: {split_summary2[\'n_test_subjects\']})")',
        'print(f"Train Subjects : {sorted(list(train_subset2[\'subject_id\'].unique()))}")',
        'print(f"Test Subjects  : {sorted(list(test_subset2[\'subject_id\'].unique()))}")',
        'print(f"Train Classes  : {split_summary2[\'train_class_dist\']}")',
        'print(f"Test Classes   : {split_summary2[\'test_class_dist\']}")',
        '',
        'overlap2 = set(train_subset2["subject_id"]).intersection(set(test_subset2["subject_id"]))',
        'print(f"Subject Overlap: {overlap2} (Count: {len(overlap2)} - ZERO LEAKAGE VERIFIED)")',
        '',
        'X_train_full2 = train_subset2[features_to_check].values',
        'y_train_full2 = train_subset2["label"].values',
        'X_test_full2 = test_subset2[features_to_check].values',
        'y_test_full2 = test_subset2["label"].values',
        '',
        'svm_model2 = train_svm(X_train_full2, y_train_full2, C=1.0, kernel="rbf", class_weight="balanced", random_state=42)',
        'y_pred_baseline2 = svm_model2.predict(X_test_full2)',
        'y_prob_baseline2 = svm_model2.predict_proba(X_test_full2)[:, 1]',
        '',
        'base_metrics2 = compute_pad_metrics(y_test_full2, y_pred_baseline2, y_prob_baseline2)',
        'print("\\n==========================================")',
        'print("ROBUSTNESS SPLIT (SEED 100) METRICS")',
        'print("==========================================")',
        'print(f"Accuracy   : {base_metrics2[\'accuracy\']:.4f}")',
        'print(f"APCER      : {base_metrics2[\'apcer\']:.4f}")',
        'print(f"BPCER      : {base_metrics2[\'bpcer\']:.4f}")',
        'print(f"ACER       : {base_metrics2[\'acer\']:.4f}")',
        'print(f"ROC-AUC    : {base_metrics2[\'roc_auc\']:.4f}")',
        'print(f"F1-Score   : {base_metrics2[\'f1\']:.4f}")',
        'print(f"Confusion  : TN={base_metrics2[\'tn\']}, FP={base_metrics2[\'fp\']}, FN={base_metrics2[\'fn\']}, TP={base_metrics2[\'tp\']}")',
        'print("\\n✅ Secondary subject-disjoint robustness experiment completed cleanly.")'
    ]),

    # SECTION 14
    make_cell('markdown', [
        '## SECTION 14 — DEEPER FAILURE ANALYSIS',
        'Inspect false positives, false negatives, subject IDs, attack IDs, and error distributions across splits.'
    ]),
    make_cell('code', [
        'failures1 = analyze_failures(',
        '    test_subset1, y_pred_baseline1, y_prob_baseline1,',
        '    subject_col="subject_id",',
        '    attack_col="attack_id" if "attack_id" in test_subset1.columns else None,',
        '    device_col="device_id" if "device_id" in test_subset1.columns else None',
        ')',
        '',
        'print("==========================================")',
        'print("PRIMARY SPLIT (SEED 42) FAILURE ANALYSIS")',
        'print("==========================================")',
        'print(f"Total Test Samples      : {failures1[\'n_test_total\']}")',
        'print(f"False Positives (Bona Fide -> Spoof) : {failures1[\'n_false_positives\']}")',
        'print(f"False Negatives (Spoof -> Bona Fide) : {failures1[\'n_false_negatives\']}")',
        'print(f"FP Subject IDs          : {failures1[\'false_positive_subjects\']}")',
        'print(f"FN Subject IDs          : {failures1[\'false_negative_subjects\']}")',
        '',
        'if "per_attack_breakdown" in failures1:',
        '    import json',
        '    print("\\nPer-Attack Error Breakdown:")',
        '    print(json.dumps(failures1["per_attack_breakdown"], indent=2))'
    ]),

    # SECTION 15
    make_cell('markdown', [
        '## SECTION 15 — FINAL REPORT FIGURES GENERATION & DISPLAY',
        'Generate publication-ready figures using `scripts/generate_final_analysis.py` and display them inline.'
    ]),
    make_cell('code', [
        'final_out_dir = drive_root / "final_analysis"',
        'if not final_out_dir.exists():',
        '    final_out_dir = repo_dir / "outputs" / "final_analysis"',
        '',
        'subprocess.run([',
        '    "python3", "scripts/generate_final_analysis.py",',
        '    "--features-csv", str(output_feature_csv),',
        '    "--output-dir", str(final_out_dir),',
        '    "--seed-primary", "42",',
        '    "--seed-robustness", "100"',
        '], check=True)',
        '',
        'fig_dir = final_out_dir / "final_figures"',
        'print(f"\\nGenerated Figures in {fig_dir}:")',
        'for fig_f in sorted(list(fig_dir.glob("*.png"))):',
        '    print(f"  - {fig_f.name}")',
        '',
        'print("\\n--- Rendered Final Figures ---")',
        'for fig_name in ["01_pipeline_diagram.png", "06_ablation_comparison.png", "07_confusion_matrix.png", "08_roc_curve.png", "09_failure_analysis.png"]:',
        '    fpath = fig_dir / fig_name',
        '    if fpath.exists():',
        '        print(f"\\nFigure: {fig_name}")',
        '        try:',
        '            display(Image(filename=str(fpath), width=700))',
        '        except Exception:',
        '            print(f"  (Image path: {fpath})")'
    ]),

    # SECTION 16
    make_cell('markdown', [
        '## SECTION 16 — FINAL SUMMARY FILES & REPORT EXPORT',
        'Export and verify `final_results.csv`, `final_experiment_summary.json`, and `FINAL_ANALYSIS.md`.'
    ]),
    make_cell('code', [
        'final_csv_file = final_out_dir / "final_results.csv"',
        'final_json_file = final_out_dir / "final_experiment_summary.json"',
        'final_md_file = repo_dir / "FINAL_ANALYSIS.md"',
        '',
        'print(f"Checking final_results.csv       : {final_csv_file.exists()}")',
        'print(f"Checking final_summary.json      : {final_json_file.exists()}")',
        'print(f"Checking FINAL_ANALYSIS.md       : {final_md_file.exists()}")',
        '',
        'if final_csv_file.exists():',
        '    print("\\n=== FINAL RESULTS TABLE (final_results.csv) ===")',
        '    display(pd.read_csv(final_csv_file)[["experiment", "num_features", "accuracy", "apcer", "bpcer", "acer", "roc_auc", "f1"]])',
        '',
        'print("\\n✅ Summary files export and verification complete.")'
    ]),

    # SECTION 17
    make_cell('markdown', [
        '## SECTION 17 — OUTPUT FILES VERIFICATION',
        'Final verification cell confirming that all required experimental outputs, figures, and summary reports exist.'
    ]),
    make_cell('code', [
        'print("==========================================================================================")',
        'print("FINAL EXPERIMENTAL VERIFICATION CHECKLIST")',
        'print("==========================================================================================")',
        'expected_files = [',
        '    output_feature_csv,',
        '    ablation_out_dir / "experiment_config.json",',
        '    ablation_out_dir / "results.csv",',
        '    final_out_dir / "final_results.csv",',
        '    final_out_dir / "final_experiment_summary.json",',
        '    final_out_dir / "final_figures" / "01_pipeline_diagram.png",',
        '    final_out_dir / "final_figures" / "06_ablation_comparison.png",',
        '    final_out_dir / "final_figures" / "07_confusion_matrix.png",',
        '    final_out_dir / "final_figures" / "08_roc_curve.png",',
        '    final_out_dir / "final_figures" / "09_failure_analysis.png",',
        '    repo_dir / "FINAL_ANALYSIS.md",',
        ']',
        '',
        'all_passed = True',
        'for ef in expected_files:',
        '    status = "✅ EXISTS" if ef.exists() else "❌ MISSING"',
        '    if not ef.exists(): all_passed = False',
        '    print(f"  {status} : {ef}")',
        '',
        'print("==========================================================================================")',
        'if all_passed:',
        '    print("🎉 ALL EXPECTED EXPERIMENTAL OUTPUT FILES VERIFIED SUCCESSFULLY.")',
        'else:',
        '    print("⚠️ WARNING: Some expected output files were not found.")'
    ])
]

notebook_json = {
    'cells': cells,
    'metadata': {
        'language_info': {
            'name': 'python'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

output_path = Path('/Users/shirishasingh/Desktop/BTPPART2/BTP_Part2_Geometry_PAD_OULU.ipynb')
with open(output_path, 'w') as f:
    json.dump(notebook_json, f, indent=1)

print('Updated Jupyter Notebook successfully at:', output_path)
