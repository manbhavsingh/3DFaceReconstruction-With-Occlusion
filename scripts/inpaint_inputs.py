#!/usr/bin/env python3
"""Create inpainted input images using BiSeNet face parsing and OpenCV inpainting."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from face_recon_occlusion.inpainting import process_image_folder
from face_recon_occlusion.segmentation import SegmentationConfig, load_bisenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing original images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for inpainted images.")
    parser.add_argument("--weights-path", required=True, type=Path, help="BiSeNet 79999_iter.pth path.")
    parser.add_argument(
        "--face-parsing-repo",
        required=True,
        type=Path,
        help="Path to the face-parsing.PyTorch repository containing model.py.",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip images that already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = SegmentationConfig(
        weights_path=args.weights_path,
        face_parsing_repo=args.face_parsing_repo,
    )
    net = load_bisenet(config, device)
    results = process_image_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        net=net,
        device=device,
        overwrite=not args.no_overwrite,
    )
    print(f"Inpainted {len(results)} images into {args.output_dir}.")


if __name__ == "__main__":
    main()
