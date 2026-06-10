"""Landmark generation helpers for Deep3DFaceRecon input folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

from .inpainting import iter_images


@dataclass(frozen=True)
class LandmarkResult:
    """Result of landmark generation for one image."""

    image_path: Path
    landmark_path: Path | None
    detected: bool


def mtcnn_keypoints_to_deep3d(keypoints: dict[str, tuple[int, int]]) -> np.ndarray:
    """Convert MTCNN keypoints to Deep3DFaceRecon's five-landmark order."""

    return np.asarray(
        [
            keypoints["left_eye"],
            keypoints["right_eye"],
            keypoints["nose"],
            keypoints["mouth_left"],
            keypoints["mouth_right"],
        ],
        dtype=np.float32,
    )


def generate_mtcnn_landmarks(image_dir: Path, detections_dir: Path | None = None) -> list[LandmarkResult]:
    """Create one ``.txt`` landmark file per image using MTCNN."""

    image_dir = Path(image_dir)
    detections_dir = Path(detections_dir) if detections_dir else image_dir / "detections"
    detections_dir.mkdir(parents=True, exist_ok=True)
    detector = MTCNN()

    results: list[LandmarkResult] = []
    for image_path in tqdm(iter_images(image_dir), desc="Generating landmarks"):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            results.append(LandmarkResult(image_path, None, False))
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(image_rgb)
        if not detections:
            results.append(LandmarkResult(image_path, None, False))
            continue

        largest = max(
            detections,
            key=lambda item: item["box"][2] * item["box"][3],
        )
        landmarks = mtcnn_keypoints_to_deep3d(largest["keypoints"])
        landmark_path = detections_dir / f"{image_path.stem}.txt"
        np.savetxt(landmark_path, landmarks, fmt="%.3f")
        results.append(LandmarkResult(image_path, landmark_path, True))
    return results
