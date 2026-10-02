"""
Unified 6-feature baseline geometry vector extractor for 3D reconstructed faces.
"""

from pathlib import Path
import numpy as np
import trimesh

from .depth import extract_depth_features
from .gradient import extract_gradient_features
from .curvature import extract_curvature_features


def extract_geometry_features(
    mesh_or_path,
    grid_size=300,
    curvature_radius_multiplier=2.0,
    gradient_clip_percentiles=(1, 99),
    curvature_clip_percentiles=(1, 99),
):
    """
    Extract the initial baseline 6-feature geometry vector from a reconstructed 3D mesh (.OBJ).

    Baseline Features:
        1. depth_std
        2. depth_range
        3. gradient_mean
        4. gradient_std
        5. gaussian_curvature_mean
        6. gaussian_curvature_std

    Args:
        mesh_or_path (str | Path | trimesh.Trimesh): Path to OBJ mesh or Trimesh instance.
        grid_size (int): Resolution of grid for depth-gradient calculation.
        curvature_radius_multiplier (float): Neighborhood radius factor for Gaussian curvature.
        gradient_clip_percentiles (tuple): Clipping range for depth gradient.
        curvature_clip_percentiles (tuple): Clipping range for Gaussian curvature.

    Returns:
        dict: Baseline 6 geometry features + metadata.
    """
    if isinstance(mesh_or_path, (str, Path)):
        obj_path = Path(mesh_or_path)
        if not obj_path.exists():
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
        mesh = trimesh.load_mesh(str(obj_path), process=False)
    elif isinstance(mesh_or_path, trimesh.Trimesh):
        mesh = mesh_or_path
    else:
        raise TypeError(f"Unsupported input type: {type(mesh_or_path)}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Invalid or empty mesh: {mesh_or_path}")

    # 1. Depth Features
    depth_res = extract_depth_features(vertices)

    # 2. Depth Gradient Features
    gradient_res = extract_gradient_features(
        vertices,
        grid_size=grid_size,
        xy_scale=depth_res["xy_scale"],
        clip_percentiles=gradient_clip_percentiles,
    )

    # 3. Discrete Gaussian Curvature Features
    curvature_res = extract_curvature_features(
        mesh,
        radius_multiplier=curvature_radius_multiplier,
        clip_percentiles=curvature_clip_percentiles,
    )

    return {
        "depth_std": depth_res["depth_std"],
        "depth_range": depth_res["depth_range"],
        "gradient_mean": gradient_res["gradient_mean"],
        "gradient_std": gradient_res["gradient_std"],
        "gaussian_curvature_mean": curvature_res["gaussian_curvature_mean"],
        "gaussian_curvature_std": curvature_res["gaussian_curvature_std"],
        "xy_scale": depth_res["xy_scale"],
        "curvature_radius": curvature_res["curvature_radius"],
    }
