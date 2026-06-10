"""Face parsing helpers used to build face and occlusion masks.

The project notebook repeatedly defines the same BiSeNet preprocessing and mask
logic.  This module keeps that logic in one place so scripts and notebooks can
share identical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

DEFAULT_FACE_LABELS: tuple[int, ...] = tuple(range(1, 16))
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for BiSeNet face parsing."""

    weights_path: Path
    face_parsing_repo: Path
    labels: Sequence[int] = DEFAULT_FACE_LABELS
    image_size: tuple[int, int] = (512, 512)
    morphology_kernel: int = 7


def build_transform(image_size: tuple[int, int] = (512, 512)) -> transforms.Compose:
    """Create the image transform expected by the common face-parsing BiSeNet."""

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_bisenet(config: SegmentationConfig, device: torch.device):
    """Load a BiSeNet model from a checked-out face-parsing repository.

    The external repository must expose ``model.BiSeNet`` under
    ``config.face_parsing_repo``.  This function deliberately avoids modifying
    global notebook state beyond adding that repository to ``sys.path`` locally.
    """

    import sys

    repo_path = str(config.face_parsing_repo)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from model import BiSeNet

    net = BiSeNet(n_classes=19)
    state = torch.load(config.weights_path, map_location=device)
    if all(key.startswith("module.") for key in state.keys()):
        state = {key[7:]: value for key, value in state.items()}
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


@torch.no_grad()
def get_segmentation(
    net,
    image_bgr: np.ndarray,
    device: torch.device,
    transform: transforms.Compose | None = None,
) -> np.ndarray:
    """Return a BiSeNet label map resized to the original image dimensions."""

    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a BGR image with shape (H, W, 3)")

    transform = transform or build_transform()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    output = net(tensor)[0]
    segmentation = output.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(
        segmentation,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def make_face_mask(
    segmentation: np.ndarray,
    labels: Iterable[int] = DEFAULT_FACE_LABELS,
    kernel_size: int = 7,
) -> np.ndarray:
    """Convert a segmentation map into a binary uint8 face mask.

    Returns a mask with values ``0`` and ``1``.  Keep conversion to ``0``/``255``
    close to OpenCV calls so metric code can use boolean masks without surprises.
    """

    if segmentation.ndim != 2:
        raise ValueError("segmentation must be a 2D label map")

    mask = np.zeros_like(segmentation, dtype=np.uint8)
    for label in labels:
        mask[segmentation == label] = 1

    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask
