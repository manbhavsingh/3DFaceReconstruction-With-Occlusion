"""Reusable helpers for the 3D face reconstruction with occlusion workflow."""

try:
    from .inpainting import inpaint_image, make_occlusion_mask, process_image_folder
except ImportError:
    inpaint_image = None
    make_occlusion_mask = None
    process_image_folder = None

from .metrics_2d import Reconstruction2DMetrics, compute_2d_metrics, split_deep3d_render
from .geometry import (
    extract_curvature_features,
    extract_depth_features,
    extract_geometry_features,
    extract_gradient_features,
)

__all__ = [
    "Reconstruction2DMetrics",
    "compute_2d_metrics",
    "extract_curvature_features",
    "extract_depth_features",
    "extract_geometry_features",
    "extract_gradient_features",
    "inpaint_image",
    "make_occlusion_mask",
    "process_image_folder",
    "split_deep3d_render",
]

