"""Reusable helpers for the 3D face reconstruction with occlusion workflow."""

try:
    from .inpainting import inpaint_image, make_occlusion_mask, process_image_folder
except ImportError:
    inpaint_image = None
    make_occlusion_mask = None
    process_image_folder = None

from .metrics_2d import Reconstruction2DMetrics, compute_2d_metrics, split_deep3d_render

__all__ = [
    "Reconstruction2DMetrics",
    "compute_2d_metrics",
    "inpaint_image",
    "make_occlusion_mask",
    "process_image_folder",
    "split_deep3d_render",
]
