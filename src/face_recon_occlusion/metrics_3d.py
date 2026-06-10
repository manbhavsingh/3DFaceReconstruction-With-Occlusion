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


def load_vertices(mesh_path: Path) -> np.ndarray:
    """Load mesh vertices without altering topology."""

    mesh = trimesh.load(mesh_path, process=False)
    return np.asarray(mesh.vertices, dtype=np.float64)


def normalized_mean_error(source_vertices: np.ndarray, target_vertices: np.ndarray) -> float:
    """Compute NME normalized by the target bounding-box diagonal."""

    distances = np.linalg.norm(source_vertices - target_vertices, axis=1)
    bbox_min = np.min(target_vertices, axis=0)
    bbox_max = np.max(target_vertices, axis=0)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    if diagonal == 0:
        return float("nan")
    return float(np.mean(distances) / diagonal)


def chamfer_distance(source_vertices: np.ndarray, target_vertices: np.ndarray) -> float:
    """Symmetric nearest-neighbor Chamfer distance between two vertex clouds."""

    source_tree = cKDTree(source_vertices)
    target_tree = cKDTree(target_vertices)
    source_to_target, _ = target_tree.query(source_vertices)
    target_to_source, _ = source_tree.query(target_vertices)
    return float(np.mean(source_to_target**2) + np.mean(target_to_source**2))


def compare_meshes(source_path: Path, target_path: Path) -> MeshMetrics:
    """Compare two meshes that are expected to share Deep3DFaceRecon topology."""

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
