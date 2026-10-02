"""
Boundary-aware discrete Gaussian curvature measure and feature extraction module.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def extract_curvature_features(
    mesh,
    radius_multiplier=2.0,
    clip_percentiles=(1, 99)
):
    """
    Calculate discrete Gaussian curvature measure with KD-Tree mesh boundary exclusion.

    Args:
        mesh (trimesh.Trimesh): Trimesh object.
        radius_multiplier (float): Multiplier for median edge length to set neighborhood radius.
        clip_percentiles (tuple): Lower and upper percentiles for robust feature clipping.

    Returns:
        dict: {
            "gaussian_curvature_mean": float,
            "gaussian_curvature_std": float,
            "curvature_radius": float,
            "curvature_clipped": np.ndarray,
            "interior_mask": np.ndarray,
            "raw_measure": np.ndarray
        }
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh instance, got {type(mesh)}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh contains no vertices or faces.")

    median_edge = float(np.median(mesh.edges_unique_length))
    curvature_radius = radius_multiplier * median_edge

    # Trimesh neighborhood-based discrete Gaussian curvature measure
    raw_measure = trimesh.curvature.discrete_gaussian_curvature_measure(
        mesh, vertices, curvature_radius
    )
    raw_measure = np.asarray(raw_measure, dtype=np.float64)

    # Detect boundary vertices (edges belonging to only 1 face)
    edges = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    boundary_vertices = np.unique(boundary_edges)

    if len(boundary_vertices) > 0:
        boundary_tree = cKDTree(vertices[boundary_vertices])
        distance_to_boundary, _ = boundary_tree.query(vertices)
    else:
        distance_to_boundary = np.full(len(vertices), np.inf)

    interior_mask = distance_to_boundary > curvature_radius
    curvature_valid = raw_measure[interior_mask & np.isfinite(raw_measure)]

    if len(curvature_valid) == 0:
        raise ValueError("No valid interior Gaussian curvature values found.")

    c_low = np.percentile(curvature_valid, clip_percentiles[0])
    c_high = np.percentile(curvature_valid, clip_percentiles[1])

    curvature_clipped = np.clip(curvature_valid, c_low, c_high)

    curvature_mean = float(np.mean(curvature_clipped))
    curvature_std = float(np.std(curvature_clipped))

    return {
        "gaussian_curvature_mean": curvature_mean,
        "gaussian_curvature_std": curvature_std,
        "curvature_radius": curvature_radius,
        "curvature_clipped": curvature_clipped,
        "interior_mask": interior_mask,
        "raw_measure": raw_measure,
    }
