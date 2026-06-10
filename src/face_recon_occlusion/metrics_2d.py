"""2D render metrics for Deep3DFaceRecon combined output images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass(frozen=True)
class Reconstruction2DMetrics:
    """Masked 2D render quality metrics."""

    mse: float
    psnr: float
    ssim: float
    pixels_used: int


def split_deep3d_render(combined_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split Deep3DFaceRecon's side-by-side PNG into input and render columns."""

    if combined_bgr is None or combined_bgr.ndim != 3:
        raise ValueError("combined_bgr must be a BGR image with shape (H, W, 3)")

    height, width = combined_bgr.shape[:2]
    if width < height * 2:
        raise ValueError("combined render must contain at least two square columns")
    original = combined_bgr[:, 0:height]
    rendered = combined_bgr[:, height : height * 2]
    return original, rendered


def default_render_mask(rendered_bgr: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Build a simple non-black mask from the rendered column."""

    gray = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2GRAY)
    return (gray > threshold).astype(bool)


def compute_2d_metrics(
    original_bgr: np.ndarray,
    rendered_bgr: np.ndarray,
    mask: np.ndarray | None = None,
) -> Reconstruction2DMetrics:
    """Compute MSE, PSNR, and SSIM for a masked original/render pair."""

    if original_bgr.shape != rendered_bgr.shape:
        raise ValueError("original_bgr and rendered_bgr must have the same shape")

    mask = default_render_mask(rendered_bgr) if mask is None else mask.astype(bool)
    pixels_used = int(mask.sum())
    if pixels_used == 0:
        raise ValueError("mask contains no valid pixels")

    original_f = original_bgr.astype(np.float32)
    rendered_f = rendered_bgr.astype(np.float32)
    diff = original_f - rendered_f
    mse_value = float(np.mean((diff[mask]) ** 2))
    psnr_value = float(peak_signal_noise_ratio(original_bgr[mask], rendered_bgr[mask], data_range=255))

    ys, xs = np.where(mask)
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    crop_original = original_bgr[ymin:ymax, xmin:xmax]
    crop_rendered = rendered_bgr[ymin:ymax, xmin:xmax]
    ssim_value = float(
        structural_similarity(
            crop_original,
            crop_rendered,
            channel_axis=-1,
            data_range=255,
        )
    )

    return Reconstruction2DMetrics(
        mse=mse_value,
        psnr=psnr_value,
        ssim=ssim_value,
        pixels_used=pixels_used,
    )
