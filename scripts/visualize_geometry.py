#!/usr/bin/env python3
"""
Multi-panel visualization generator script for reconstructed 3D facial geometry.
"""

import argparse
import sys
from pathlib import Path

# Ensure src directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_recon_occlusion.geometry.visualization import visualize_geometry_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 3D Face Geometry Sample")
    parser.add_argument("--obj-path", type=str, required=True, help="Path to reconstructed OBJ mesh.")
    parser.add_argument("--original-img", type=str, default=None, help="Path to original input image.")
    parser.add_argument("--render-img", type=str, default=None, help="Path to reconstructed render PNG.")
    parser.add_argument("--output-png", type=str, required=True, help="Save path for generated figure PNG.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Generating 5-panel visualization for mesh: {args.obj_path}")
    visualize_geometry_sample(
        obj_path=args.obj_path,
        original_img_path=args.original_img,
        render_img_path=args.render_img,
        output_path=args.output_png,
    )
    print(f"✅ Saved figure: {args.output_png}")


if __name__ == "__main__":
    main()
