#!/usr/bin/env python3
"""Evaluate Deep3DFaceRecon combined render PNG files with 2D metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from face_recon_occlusion.metrics_2d import compute_2d_metrics, split_deep3d_render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path, help="Folder containing Deep3DFaceRecon PNG results.")
    parser.add_argument("--output-csv", required=True, type=Path, help="CSV path for computed metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str | float | int]] = []
    for png_path in sorted(args.results_dir.glob("*.png")):
        image = cv2.imread(str(png_path))
        if image is None:
            raise ValueError(f"Could not read result image: {png_path}")
        original, rendered = split_deep3d_render(image)
        metrics = compute_2d_metrics(original, rendered)
        rows.append(
            {
                "filename": png_path.name,
                "mse": metrics.mse,
                "psnr": metrics.psnr,
                "ssim": metrics.ssim,
                "pixels_used": metrics.pixels_used,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "mse", "psnr", "ssim", "pixels_used"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote metrics for {len(rows)} files to {args.output_csv}.")


if __name__ == "__main__":
    main()
