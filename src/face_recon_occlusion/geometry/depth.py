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
    if len(vertices) == 0:
        raise ValueError("Vertices array is empty.")

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    xy_width = float(x.max() - x.min())
    xy_height = float(y.max() - y.min())
    xy_scale = float(np.sqrt(xy_width**2 + xy_height**2))

    if xy_scale <= 1e-8:
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
