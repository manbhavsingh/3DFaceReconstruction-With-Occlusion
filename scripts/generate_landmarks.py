#!/usr/bin/env python3
"""Generate Deep3DFaceRecon-compatible five-point landmarks for an image folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from face_recon_occlusion.landmarks import generate_mtcnn_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path, help="Folder containing input images.")
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=None,
        help="Output folder for .txt landmarks. Defaults to IMAGE_DIR/detections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = generate_mtcnn_landmarks(args.image_dir, args.detections_dir)
    detected = sum(result.detected for result in results)
    print(f"Generated landmarks for {detected}/{len(results)} images.")


if __name__ == "__main__":
    main()
