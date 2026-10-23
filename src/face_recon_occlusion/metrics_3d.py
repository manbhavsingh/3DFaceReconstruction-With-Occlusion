"""3D mesh comparison metrics for reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MeshMetrics:
    """Mesh distance metrics after optional topology checks."""

    rmse: float
    mae: float
    nme: float
    chamfer: float


def _validate_vertices(vertices: np.ndarray, name: str) -> np.ndarray:
    """Return a finite, non-empty (N, 3) float64 vertex array."""
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one vertex")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


def load_vertices(mesh_path: Path) -> np.ndarray:
    """Load mesh vertices without altering topology."""
    mesh = trimesh.load(mesh_path, process=False)
    return _validate_vertices(np.asarray(mesh.vertices, dtype=np.float64), "mesh vertices")


def normalized_mean_error(source_vertices: np.ndarray, target_vertices: np.ndarray) -> float:
    """Compute NME normalized by the target bounding-box diagonal."""
    source = _validate_vertices(source_vertices, "source_vertices")
    target = _validate_vertices(target_vertices, "target_vertices")
    if source.shape != target.shape:
        raise ValueError(
            f"Mesh vertex shape mismatch: {source.shape} vs {target.shape}"
        )

    distances = np.linalg.norm(source - target, axis=1)
    bbox_min = np.min(target, axis=0)
    bbox_max = np.max(target, axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("target mesh bounding-box diagonal must be finite and positive")
    return float(np.mean(distances) / diagonal)


def chamfer_distance(source_vertices: np.ndarray, target_vertices: np.ndarray) -> float:
    """Symmetric nearest-neighbor squared Chamfer distance between vertex clouds."""
    source = _validate_vertices(source_vertices, "source_vertices")
    target = _validate_vertices(target_vertices, "target_vertices")

    source_tree = cKDTree(source)
    target_tree = cKDTree(target)
    source_to_target, _ = target_tree.query(source)
    target_to_source, _ = source_tree.query(target)
    return float(np.mean(source_to_target**2) + np.mean(target_to_source**2))


def compare_meshes(source_path: Path, target_path: Path) -> MeshMetrics:
    """Compare two meshes expected to share Deep3DFaceRecon topology."""
    source_vertices = load_vertices(source_path)
    target_vertices = load_vertices(target_path)
    if source_vertices.shape != target_vertices.shape:
        raise ValueError(
            f"Mesh vertex shape mismatch: {source_vertices.shape} vs {target_vertices.shape}"
        )

    distances = np.linalg.norm(source_vertices - target_vertices, axis=1)
    return MeshMetrics(
        rmse=float(np.sqrt(np.mean(distances**2))),
        mae=float(np.mean(distances)),
        nme=normalized_mean_error(source_vertices, target_vertices),
        chamfer=chamfer_distance(source_vertices, target_vertices),
    )
