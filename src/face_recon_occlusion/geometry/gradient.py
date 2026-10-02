"""
2D Depth-gradient magnitude calculation and feature extraction module.
"""

import numpy as np
from scipy.interpolate import griddata


def extract_gradient_features(
    vertices,
    grid_size=300,
    xy_scale=None,
    clip_percentiles=(1, 99)
):
    """
    Interpolate depth onto a 2D regular grid and calculate depth-gradient features.

    Args:
        vertices (np.ndarray): Nx3 mesh vertex coordinates [X, Y, Z].
        grid_size (int): Resolution of the 2D grid for depth interpolation.
        xy_scale (float, optional): XY facial diagonal scale for normalization.
                                   If None, computed from vertices.
        clip_percentiles (tuple): Lower and upper percentiles for robust feature clipping.

    Returns:
        dict: {
            "gradient_mean": float,
            "gradient_std": float,
            "gradient_clipped": np.ndarray,
            "grid_X": np.ndarray,
            "grid_Y": np.ndarray,
            "grid_Z": np.ndarray
        }
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError("Vertices array is empty.")

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    if xy_scale is None:
        xy_width = float(x.max() - x.min())
        xy_height = float(y.max() - y.min())
        xy_scale = float(np.sqrt(xy_width**2 + xy_height**2))

    if xy_scale <= 1e-8:
        raise ValueError(f"Invalid or near-zero XY scale: {xy_scale}")

    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method="linear")

    dx = float(xi[1] - xi[0])
    dy = float(yi[1] - yi[0])

    Gy, Gx = np.gradient(Z, dy, dx)
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)

    valid_gradient = gradient_magnitude[np.isfinite(gradient_magnitude)]
    if len(valid_gradient) == 0:
        raise ValueError("No valid depth-gradient values found during grid interpolation.")

    gradient_normalized = valid_gradient / xy_scale

    g_low = np.percentile(gradient_normalized, clip_percentiles[0])
    g_high = np.percentile(gradient_normalized, clip_percentiles[1])

    gradient_clipped = np.clip(gradient_normalized, g_low, g_high)

    gradient_mean = float(np.mean(gradient_clipped))
    gradient_std = float(np.std(gradient_clipped))

    return {
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "gradient_clipped": gradient_clipped,
        "grid_X": X,
        "grid_Y": Y,
        "grid_Z": Z,
        "gradient_magnitude": gradient_magnitude,
    }
