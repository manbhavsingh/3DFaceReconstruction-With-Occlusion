"""
2D depth-gradient magnitude calculation and feature extraction module.
"""

import numpy as np
from scipy.interpolate import griddata


def _validate_clip_percentiles(clip_percentiles):
    """Return validated percentile bounds for robust clipping."""
    if not isinstance(clip_percentiles, (tuple, list, np.ndarray)) or len(clip_percentiles) != 2:
        raise ValueError("clip_percentiles must contain exactly two values (low, high).")
    low, high = (float(clip_percentiles[0]), float(clip_percentiles[1]))
    if not np.isfinite([low, high]).all() or not (0 <= low < high <= 100):
        raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100.")
    return low, high


def extract_gradient_features(
    vertices,
    grid_size=300,
    xy_scale=None,
    clip_percentiles=(1, 99),
):
    """
    Interpolate mesh depth onto a regular XY grid and summarize the slope magnitude.

    The gradient magnitude |∇Z| = sqrt((dZ/dx)^2 + (dZ/dy)^2) is dimensionless.
    It is invariant under uniform 3D scaling and must not be divided by xy_scale.
    """
    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)) or grid_size < 2:
        raise ValueError("grid_size must be an integer >= 2.")

    low, high = _validate_clip_percentiles(clip_percentiles)
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Vertices must have shape (n, 3); got {vertices.shape}.")
    if len(vertices) < 3:
        raise ValueError("At least three vertices are required for gradient interpolation.")
    if not np.isfinite(vertices).all():
        raise ValueError("Vertices contain NaN or infinite coordinates.")

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    x_span, y_span = float(np.ptp(x)), float(np.ptp(y))
    if x_span <= 1e-8 or y_span <= 1e-8:
        raise ValueError("Mesh XY coordinates must span non-zero width and height.")

    if xy_scale is None:
        xy_scale = float(np.hypot(x_span, y_span))
    if not np.isfinite(xy_scale) or xy_scale <= 1e-8:
        raise ValueError(f"Invalid or near-zero XY scale: {xy_scale}")

    xi = np.linspace(float(x.min()), float(x.max()), int(grid_size))
    yi = np.linspace(float(y.min()), float(y.max()), int(grid_size))
    X, Y = np.meshgrid(xi, yi)
    try:
        Z = griddata((x, y), z, (X, Y), method="linear")
    except Exception as exc:
        raise ValueError(
            "Could not interpolate mesh depth on the XY grid; check for duplicate "
            "or degenerate XY coordinates."
        ) from exc

    dx, dy = float(xi[1] - xi[0]), float(yi[1] - yi[0])
    if dx <= 0 or dy <= 0:
        raise ValueError("Grid spacing must be positive.")

    Gy, Gx = np.gradient(Z, dy, dx)
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    valid_gradient = gradient_magnitude[np.isfinite(gradient_magnitude)]
    if valid_gradient.size == 0:
        raise ValueError("No valid depth-gradient values found during grid interpolation.")

    g_low, g_high = np.percentile(valid_gradient, [low, high])
    gradient_clipped = np.clip(valid_gradient, g_low, g_high)
    return {
        "gradient_mean": float(np.mean(gradient_clipped)),
        "gradient_std": float(np.std(gradient_clipped)),
        "gradient_clipped": gradient_clipped,
        "grid_X": X,
        "grid_Y": Y,
        "grid_Z": Z,
        "gradient_magnitude": gradient_magnitude,
    }
