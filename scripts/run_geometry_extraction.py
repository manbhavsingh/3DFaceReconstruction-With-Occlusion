#!/usr/bin/env python3
"""
Cache-First and Restart-Safe Batch Geometry Feature Extractor for 3D Face Meshes (.OBJ).

Research Workflow:
  Reconstructed OBJ Mesh -> Depth / Gradient / Gaussian Curvature -> 6 Baseline Features -> Feature CSV

Restart-Safe Policy:
  1. Load existing CSV if present.
  2. Detect already processed sample_ids.
  3. Extract features ONLY for missing/unprocessed samples.
  4. Save/append progress immediately after every sample to survive Colab runtime disconnects.
  5. Never delete or overwrite valid existing entries unless --force-recompute is explicitly set.

Usage:
  python scripts/run_geometry_extraction.py \
      --mesh-dir /content/drive/MyDrive/BTP_Part2_OULU/subject_disjoint_pilot_100/reconstructions \
      --metadata-csv /content/drive/MyDrive/BTP_Part2_OULU/subject_disjoint_pilot_100/oulu_subject_disjoint_pilot_100.csv \
      --output-csv /content/drive/MyDrive/BTP_Part2_OULU/subject_disjoint_pilot_100/oulu_geometry_features.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Ensure package in src/ is importable when script is executed from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from face_recon_occlusion.geometry import extract_geometry_features
except ImportError as err:
    print(f"Import Error: {err}")
    print("Ensure script is executed from repository root or src/ is on PYTHONPATH.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache-First and Restart-Safe Batch 3D Geometry Feature Extractor"
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        required=True,
        help="Directory containing reconstructed OBJ mesh files (searched recursively).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Optional path to dataset metadata CSV (containing sample_id, subject_id, label, attack_id, etc.).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Output CSV path for saving extracted geometry features.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Number of samples between incremental CSV saves (default: 1 for maximum restart safety).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of all samples, ignoring existing cache.",
    )
    return parser.parse_args()


def load_cache(output_csv, force_recompute=False):
    """
    Load existing feature table cache from output_csv if present.

    Returns:
        dict: Mapping of sample_id -> row_dict, and list of existing DataFrames.
    """
    cache_dict = {}
    if not output_csv.exists() or force_recompute:
        return cache_dict

    try:
        df = pd.read_csv(output_csv)
        required_cols = [
            "sample_id",
            "depth_std",
            "depth_range",
            "gradient_mean",
            "gradient_std",
            "gaussian_curvature_mean",
            "gaussian_curvature_std",
        ]
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️ Cache at {output_csv} missing required feature columns. Will re-extract.")
            return cache_dict

        for _, row in df.iterrows():
            cache_dict[str(row["sample_id"])] = row.to_dict()

        print(f"✅ Found valid cache with {len(cache_dict)} processed samples at {output_csv}.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to read existing cache at {output_csv} ({e}). Will compute missing.")

    return cache_dict


def save_progress(records, output_csv):
    """Save records DataFrame atomically to disk."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    # Ensure standard column order
    key_cols = [
        "sample_id", "subject_id", "label", "attack_id", "device_id",
        "depth_std", "depth_range", "gradient_mean", "gradient_std",
        "gaussian_curvature_mean", "gaussian_curvature_std",
        "xy_scale", "curvature_radius", "obj_path"
    ]
    present_cols = [c for c in key_cols if c in df.columns] + [c for c in df.columns if c not in key_cols]
    df = df[present_cols]
    df.to_csv(output_csv, index=False)


def main():
    args = parse_args()
    mesh_dir = Path(args.mesh_dir)
    output_csv = Path(args.output_csv)

    if not mesh_dir.exists():
        print(f"❌ Error: Mesh directory does not exist: {mesh_dir}")
        sys.exit(1)

    # 1. Load Metadata if provided
    metadata_map = {}
    if args.metadata_csv and Path(args.metadata_csv).exists():
        print(f"Reading dataset metadata from: {args.metadata_csv}")
        meta_df = pd.read_csv(args.metadata_csv)
        if "sample_id" in meta_df.columns:
            for _, r in meta_df.iterrows():
                metadata_map[str(r["sample_id"])] = r.to_dict()
        else:
            print("⚠️ Metadata CSV missing 'sample_id' column. Matching by filename stem.")

    # 2. Load Existing Cache
    cache_dict = load_cache(output_csv, force_recompute=args.force_recompute)

    # 3. Discover OBJ files
    obj_files = sorted(list(mesh_dir.rglob("*.obj")))
    print(f"Found {len(obj_files)} OBJ mesh files in {mesh_dir}.")

    if len(obj_files) == 0:
        print(f"⚠️ Warning: No .obj files found in {mesh_dir}.")
        sys.exit(0)

    records = list(cache_dict.values())
    reused_count = 0
    computed_count = 0
    failed_count = 0

    # 4. Processing Loop
    pbar = tqdm(obj_files, desc="Batch Feature Extraction")
    for idx, obj_path in enumerate(pbar):
        sample_id = obj_path.stem

        # CACHE-FIRST: Skip already processed sample
        if sample_id in cache_dict and not args.force_recompute:
            reused_count += 1
            continue

        try:
            feats = extract_geometry_features(obj_path)
            rec = {
                "sample_id": sample_id,
                "obj_path": str(obj_path),
                **feats,
            }

            # Merge metadata attributes (subject_id, label, attack_id, device_id, etc.)
            if sample_id in metadata_map:
                for k, v in metadata_map[sample_id].items():
                    if k not in rec:
                        rec[k] = v

            records.append(rec)
            cache_dict[sample_id] = rec
            computed_count += 1

            # RESTART-SAFE: Incremental save
            if computed_count % args.save_interval == 0:
                save_progress(records, output_csv)

        except Exception as err:
            failed_count += 1
            print(f"\n❌ Error processing OBJ {obj_path}: {err}")

    # Final Save
    if computed_count > 0:
        save_progress(records, output_csv)

    print("\n==========================================")
    print("Geometry Feature Extraction Summary")
    print("==========================================")
    print(f"Total Meshes Found : {len(obj_files)}")
    print(f"Reused Cache       : {reused_count}")
    print(f"Newly Extracted    : {computed_count}")
    print(f"Failed Extraction  : {failed_count}")
    print(f"Output CSV         : {output_csv}")
    print("==========================================")


if __name__ == "__main__":
    main()
