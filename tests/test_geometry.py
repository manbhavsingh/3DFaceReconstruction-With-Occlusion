"""
Tests for 3D geometric feature extraction: depth, depth gradient, and Gaussian curvature.
"""

import numpy as np
import pytest
import trimesh

from face_recon_occlusion.geometry.depth import extract_depth_features
from face_recon_occlusion.geometry.gradient import extract_gradient_features
from face_recon_occlusion.geometry.curvature import extract_curvature_features
from face_recon_occlusion.geometry.features import extract_geometry_features


def test_depth_scale_invariance():
    """Verify that normalized depth features are invariant under uniform 3D scaling."""
    rng = np.random.RandomState(42)
    # Generate random 3D points
    vertices = rng.uniform(-50, 50, size=(200, 3))

    feats_1 = extract_depth_features(vertices)
    feats_2 = extract_depth_features(vertices * 3.5)

    assert np.isclose(feats_1["depth_std"], feats_2["depth_std"], rtol=1e-5)
    assert np.isclose(feats_1["depth_range"], feats_2["depth_range"], rtol=1e-5)
    assert np.isclose(feats_2["xy_scale"], feats_1["xy_scale"] * 3.5, rtol=1e-5)


def test_depth_gradient_scale_invariance():
    """
    Regression test: verify that depth-gradient magnitude is dimensionless (slope)
    and remains invariant under uniform 3D scaling (sX, sY, sZ).
    """
    rng = np.random.RandomState(42)
    # Generate points on an inclined plane: z = 0.6 * x - 0.4 * y + 2.0
    x = rng.uniform(-20, 20, 800)
    y = rng.uniform(-20, 20, 800)
    z = 0.6 * x - 0.4 * y + 2.0
    verts_unscaled = np.column_stack([x, y, z])

    # Scale 3D coordinates by factor of 2.5
    scale_factor = 2.5
    verts_scaled = verts_unscaled * scale_factor

    res_unscaled = extract_gradient_features(verts_unscaled, grid_size=80)
    res_scaled = extract_gradient_features(verts_scaled, grid_size=80)

    # Expected analytical slope: sqrt(0.6^2 + (-0.4)^2) = sqrt(0.52) ≈ 0.72111
    expected_slope = float(np.sqrt(0.6**2 + 0.4**2))

    assert np.isclose(res_unscaled["gradient_mean"], expected_slope, rtol=0.05)
    # Scale invariance check: scaled mesh must produce the identical slope
    assert np.isclose(
        res_unscaled["gradient_mean"], res_scaled["gradient_mean"], rtol=1e-4
    ), (
        f"Gradient magnitude is not scale-invariant! "
        f"Unscaled={res_unscaled['gradient_mean']:.6f}, Scaled={res_scaled['gradient_mean']:.6f}"
    )
    assert np.isclose(res_unscaled["gradient_std"], res_scaled["gradient_std"], rtol=1e-4)


def test_depth_gradient_non_negative_and_finite():
    """Verify that gradient magnitude values are non-negative and finite."""
    rng = np.random.RandomState(123)
    x = rng.uniform(-10, 10, 300)
    y = rng.uniform(-10, 10, 300)
    z = np.sin(x) + np.cos(y)
    verts = np.column_stack([x, y, z])

    res = extract_gradient_features(verts, grid_size=50)

    assert res["gradient_mean"] >= 0.0
    assert res["gradient_std"] >= 0.0
    assert np.all(res["gradient_clipped"] >= 0.0)
    assert np.isfinite(res["gradient_mean"])
    assert np.isfinite(res["gradient_std"])


def test_curvature_boundary_exclusion():
    """Verify that curvature computation handles boundary vertices and excludes them."""
    # Create an open hemisphere mesh with a clear open boundary circle at z=0
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    verts = sphere.vertices
    # Retain only triangles in the upper hemisphere
    upper_mask = (verts[sphere.faces, 2] >= 0).all(axis=1)
    open_mesh = trimesh.Trimesh(vertices=verts, faces=sphere.faces[upper_mask])

    curv_res = extract_curvature_features(open_mesh, radius_multiplier=2.0)

    assert np.isfinite(curv_res["gaussian_curvature_mean"])
    assert np.isfinite(curv_res["gaussian_curvature_std"])
    assert curv_res["interior_mask"].sum() < len(open_mesh.vertices)  # Boundary vertices were excluded
    assert curv_res["interior_mask"].sum() > 0  # Interior vertices exist
    assert len(curv_res["curvature_clipped"]) > 0


def test_unified_geometry_feature_extraction():
    """Verify that extract_geometry_features returns all 6 baseline scalar features."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    feats = extract_geometry_features(mesh, grid_size=50)

    expected_keys = [
        "depth_std",
        "depth_range",
        "gradient_mean",
        "gradient_std",
        "gaussian_curvature_mean",
        "gaussian_curvature_std",
        "xy_scale",
        "curvature_radius",
    ]
    for k in expected_keys:
        assert k in feats
        assert np.isfinite(feats[k])


def test_gradient_rejects_invalid_grid_size():
    """Reject resolutions that previously caused index errors or invalid grids."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    for grid_size in (0, 1, 1.5, True):
        with pytest.raises(ValueError, match="grid_size"):
            extract_gradient_features(vertices, grid_size=grid_size)


def test_gradient_rejects_non_finite_or_malformed_vertices():
    with pytest.raises(ValueError, match="shape"):
        extract_gradient_features(np.array([1.0, 2.0, 3.0]))
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, np.nan]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        extract_gradient_features(vertices, grid_size=10)


def test_curvature_rejects_invalid_radius_multiplier():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    with pytest.raises(ValueError, match="radius_multiplier"):
        extract_curvature_features(mesh, radius_multiplier=0)
