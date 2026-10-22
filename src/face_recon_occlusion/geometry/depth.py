"""
Depth extraction and facial bounding-scale normalization module.
"""

import numpy as np


def extract_depth_features(vertices):
    """
    Extract Z-depth statistics normalized by the XY facial bounding diagonal.

    Args:
        vertices (np.ndarray): Nx3 mesh vertex coordinates [X, Y, Z].

    Returns:
        dict: {
            "depth_std": float,
            "depth_range": float,
            "xy_scale": float,
            "z_normalized": np.ndarray
        }
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Vertices must have shape (n, 3); got {vertices.shape}.")
    if len(vertices) < 3:
        raise ValueError("At least three vertices are required for depth features.")
    if not np.isfinite(vertices).all():
        raise ValueError("Vertices contain NaN or infinite coordinates.")

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    xy_width = float(x.max() - x.min())
    xy_height = float(y.max() - y.min())
    xy_scale = float(np.sqrt(xy_width**2 + xy_height**2))

    if not np.isfinite(xy_scale) or xy_scale <= 1e-8:
        raise ValueError(f"Invalid or near-zero XY scale in mesh: {xy_scale}")

    # Center depth by median to eliminate translation ambiguity
    z_centered = z - np.median(z)
    z_normalized = z_centered / xy_scale

    depth_std = float(np.std(z_normalized))
    depth_range = float(np.ptp(z_normalized))

    return {
        "depth_std": depth_std,
        "depth_range": depth_range,
        "xy_scale": xy_scale,
        "z_normalized": z_normalized,
    }
