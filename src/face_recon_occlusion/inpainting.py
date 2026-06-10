"""Occlusion-mask construction and inpainting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from .segmentation import DEFAULT_FACE_LABELS, get_segmentation, make_face_mask

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class InpaintResult:
    """Paths and pixel counts for one processed image."""

    input_path: Path
    output_path: Path
    face_pixels: int
    inpaint_pixels: int


def iter_images(folder: Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    """List image files in deterministic order."""

    normalized = tuple(ext.lower() for ext in extensions)
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in normalized)


def make_occlusion_mask(face_mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Create an OpenCV inpainting mask from a binary face mask.

    Non-face pixels become ``255`` and face pixels become ``0``.  This preserves
    the behavior of the original notebook while making the policy explicit.
    """

    if face_mask.ndim != 2:
        raise ValueError("face_mask must be a 2D binary mask")

    occlusion_mask = (face_mask == 0).astype(np.uint8) * 255
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
    return occlusion_mask


def inpaint_image(image_bgr: np.ndarray, occlusion_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Inpaint an image using OpenCV Telea inpainting."""

    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a BGR image with shape (H, W, 3)")
    if occlusion_mask.shape != image_bgr.shape[:2]:
        raise ValueError("occlusion_mask must match the image height and width")
    return cv2.inpaint(image_bgr, occlusion_mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)


def process_image_folder(
    input_dir: Path,
    output_dir: Path,
    net,
    device,
    labels: Iterable[int] = DEFAULT_FACE_LABELS,
    overwrite: bool = True,
) -> list[InpaintResult]:
    """Generate inpainted copies of every image in ``input_dir``."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[InpaintResult] = []
    for image_path in tqdm(iter_images(input_dir), desc="Inpainting images"):
        output_path = output_dir / image_path.name
        if output_path.exists() and not overwrite:
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        segmentation = get_segmentation(net, image_bgr, device)
        face_mask = make_face_mask(segmentation, labels=labels)
        occlusion_mask = make_occlusion_mask(face_mask)
        inpainted = inpaint_image(image_bgr, occlusion_mask)

        if not cv2.imwrite(str(output_path), inpainted):
            raise OSError(f"Failed to write inpainted image: {output_path}")

        results.append(
            InpaintResult(
                input_path=image_path,
                output_path=output_path,
                face_pixels=int(face_mask.sum()),
                inpaint_pixels=int((occlusion_mask > 0).sum()),
            )
        )
    return results
