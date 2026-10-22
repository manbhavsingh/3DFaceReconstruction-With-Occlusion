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
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure package in src/ is importable when script is executed from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from face_recon_occlusion.geometry import extract_geometry_features
    from face_recon_occlusion.data import normalize_id
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
    """Load a validated feature cache, failing closed instead of risking prior results."""
    if not output_csv.exists() or force_recompute:
        return {}
    try:
        df = pd.read_csv(output_csv)
    except Exception as exc:
        raise ValueError(
            f"Cannot read existing feature cache {output_csv}; inspect/repair it or use "
            "--force-recompute only if replacing it is intentional."
        ) from exc

    required_cols = [
        "sample_id", "depth_std", "depth_range", "gradient_mean", "gradient_std",
        "gaussian_curvature_mean", "gaussian_curvature_std",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Existing feature cache {output_csv} is missing columns {missing}; it was left untouched.")
    if df["sample_id"].isna().any():
        raise ValueError(f"Existing feature cache {output_csv} contains missing sample_id values.")
    normalized_ids = df["sample_id"].map(normalize_id)
    if normalized_ids.eq("").any() or normalized_ids.duplicated().any():
        raise ValueError(f"Existing feature cache {output_csv} contains blank or duplicate normalized sample IDs.")
    feature_values = df[required_cols[1:]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError(f"Existing feature cache {output_csv} contains non-finite/non-numeric features; it was left untouched.")

    cache_dict = {}
    for _, row in df.iterrows():
        cache_dict[normalize_id(row["sample_id"])] = row.to_dict()
    print(f"Found validated cache with {len(cache_dict)} processed samples at {output_csv}.")
    return cache_dict


def save_progress(records, output_csv):
    """Write a checkpoint to a temporary file, then replace the previous CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    key_cols = [
        "sample_id", "subject_id", "label", "attack_id", "device_id",
        "depth_std", "depth_range", "gradient_mean", "gradient_std",
        "gaussian_curvature_mean", "gaussian_curvature_std",
        "xy_scale", "curvature_radius", "obj_path",
    ]
    present_cols = [col for col in key_cols if col in df.columns]
    present_cols += [col for col in df.columns if col not in present_cols]
    df = df[present_cols]

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_csv.name}.", suffix=".tmp", dir=str(output_csv.parent)
    )
    os.close(fd)
    try:
        df.to_csv(tmp_name, index=False)
        os.replace(tmp_name, output_csv)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main():
    args = parse_args()
    mesh_dir = Path(args.mesh_dir)
    output_csv = Path(args.output_csv)

    if args.save_interval < 1:
        raise SystemExit("--save-interval must be an integer >= 1.")
    if args.metadata_csv and not Path(args.metadata_csv).is_file():
        raise SystemExit(f"Metadata CSV does not exist: {args.metadata_csv}")
    if args.metadata_csv and output_csv.resolve() == Path(args.metadata_csv).resolve():
        raise SystemExit("Output CSV must not overwrite the metadata input CSV.")

    if not mesh_dir.is_dir():
        print(f"❌ Error: Mesh directory does not exist: {mesh_dir}")
        sys.exit(1)

    # 1. Load Metadata if provided with extension-agnostic key mapping
    metadata_map = {}
    if args.metadata_csv and Path(args.metadata_csv).exists():
        print(f"Reading dataset metadata from: {args.metadata_csv}")
        meta_df = pd.read_csv(args.metadata_csv)

        id_col = None
        for candidate in ["sample_id", "filename", "image_path", "name", "id"]:
            if candidate in meta_df.columns:
                id_col = candidate
                break

        if id_col:
            for _, r in meta_df.iterrows():
                norm_key = normalize_id(r[id_col])
                metadata_map[norm_key] = r.to_dict()
                # Also store exact raw string if different
                metadata_map[str(r[id_col]).strip()] = r.to_dict()
            print(f"✅ Mapped {len(metadata_map)} metadata records using ID column '{id_col}'.")
        else:
            print("⚠️ Metadata CSV missing recognized ID column (sample_id, filename, name, etc.).")

    # 2. Load Existing Cache
    cache_dict = load_cache(output_csv, force_recompute=args.force_recompute)

    # 3. Discover OBJ files
    obj_files = sorted(list(mesh_dir.rglob("*.obj")))
    print(f"Found {len(obj_files)} OBJ mesh files in {mesh_dir}.")

    if not obj_files:
        print(f"Error: No .obj files found in {mesh_dir}.")
        sys.exit(2)

    stems = [normalize_id(path.stem) for path in obj_files]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(
            "Multiple OBJ files resolve to the same sample_id; extraction would create "
            f"ambiguous cache rows: {duplicates[:20]}"
        )

    records = list(cache_dict.values())
    reused_count = 0
    computed_count = 0
    failed_count = 0

    # 4. Processing Loop
    pbar = tqdm(obj_files, desc="Batch Feature Extraction")
    for idx, obj_path in enumerate(pbar):
        sample_id = obj_path.stem
        norm_stem = normalize_id(sample_id)

        # Check if already in cache
        cache_key = normalize_id(sample_id)
        if cache_key in cache_dict and not args.force_recompute:
            # Backfill metadata if missing in cached row
            rec = cache_dict[cache_key]
            matched_meta = metadata_map.get(norm_stem) or metadata_map.get(sample_id)
            if matched_meta:
                for k, v in matched_meta.items():
                    if k not in rec or pd.isna(rec[k]):
                        rec[k] = v
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
            matched_meta = metadata_map.get(norm_stem) or metadata_map.get(sample_id)
            if matched_meta:
                for k, v in matched_meta.items():
                    if k not in rec:
                        rec[k] = v

            records.append(rec)
            cache_dict[cache_key] = rec
            computed_count += 1

            # RESTART-SAFE: Incremental save
            if computed_count % args.save_interval == 0:
                save_progress(records, output_csv)

        except Exception as err:
            failed_count += 1
            print(f"\n❌ Error processing OBJ {obj_path}: {err}")

    # Backfill any remaining cached records if metadata was missing
    if metadata_map:
        for rec in records:
            sid = str(rec.get("sample_id", ""))
            norm_sid = normalize_id(sid)
            matched_meta = metadata_map.get(norm_sid) or metadata_map.get(sid)
            if matched_meta:
                for k, v in matched_meta.items():
                    if k not in rec or pd.isna(rec[k]):
                        rec[k] = v

    # Final Save
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
    if failed_count:
        print(f"ERROR: {failed_count} mesh(es) failed extraction. Partial progress was saved; inspect the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
