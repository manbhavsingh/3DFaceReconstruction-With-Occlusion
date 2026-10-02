"""
Diagnostic multi-panel visualization generator for reconstructed 3D face geometry.
"""

from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from .depth import extract_depth_features
from .gradient import extract_gradient_features
from .curvature import extract_curvature_features


def visualize_geometry_sample(
    obj_path,
    original_img_path=None,
    render_img_path=None,
    output_path=None,
    figsize=(20, 4),
    dpi=150
):
    """
    Generate a 5-panel diagnostic visualization:
      [Original Input] | [Reconstruction Render] | [Depth Map] | [Depth Gradient Map] | [Gaussian Curvature Map]

    Args:
        obj_path (str | Path): Path to reconstructed OBJ file.
        original_img_path (str | Path, optional): Path to original input image.
        render_img_path (str | Path, optional): Path to Deep3DFaceRecon rendered PNG.
        output_path (str | Path, optional): Save target path for figure PNG.
        figsize (tuple): Figure size.
        dpi (int): Dots per inch for saved figure.

    Returns:
        matplotlib.figure.Figure: Generated figure.
    """
    obj_path = Path(obj_path)
    mesh = trimesh.load_mesh(str(obj_path), process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    # Compute features
    depth_res = extract_depth_features(vertices)
    grad_res = extract_gradient_features(vertices, grid_size=300, xy_scale=depth_res["xy_scale"])
    curv_res = extract_curvature_features(mesh)

    fig, axes = plt.subplots(1, 5, figsize=figsize)

    # Panel 1: Original Image
    if original_img_path and Path(original_img_path).exists():
        img = plt.imread(str(original_img_path))
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
    else:
        axes[0].text(0.5, 0.5, "Image\nNot Provided", ha="center", va="center")
        axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Panel 2: Rendered PNG
    if render_img_path and Path(render_img_path).exists():
        render_img = plt.imread(str(render_img_path))
        axes[1].imshow(render_img)
        axes[1].set_title("Reconstruction Render")
    else:
        axes[1].text(0.5, 0.5, "Render\nNot Provided", ha="center", va="center")
        axes[1].set_title("3D Render")
    axes[1].axis("off")

    # Panel 3: Depth Map
    X_grid = grad_res["grid_X"]
    Y_grid = grad_res["grid_Y"]
    Z_grid = grad_res["grid_Z"]
    im_depth = axes[2].imshow(
        Z_grid, extent=[X_grid.min(), X_grid.max(), Y_grid.min(), Y_grid.max()],
        origin="lower", cmap="viridis"
    )
    axes[2].set_title(f"Depth Map\nstd: {depth_res['depth_std']:.4f}")
    fig.colorbar(im_depth, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis("off")

    # Panel 4: Depth Gradient Map
    grad_mag = grad_res["gradient_magnitude"]
    im_grad = axes[3].imshow(
        grad_mag, extent=[X_grid.min(), X_grid.max(), Y_grid.min(), Y_grid.max()],
        origin="lower", cmap="plasma"
    )
    axes[3].set_title(f"Depth Gradient |∇Z|\nmean: {grad_res['gradient_mean']:.4f}")
    fig.colorbar(im_grad, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis("off")

    # Panel 5: Boundary-Aware Gaussian Curvature Map
    interior_mask = curv_res["interior_mask"]
    raw_k = curv_res["raw_measure"].copy()
    raw_k[~interior_mask] = np.nan

    valid_k = raw_k[np.isfinite(raw_k)]
    low_k, high_k = np.percentile(valid_k, [1, 99])
    clipped_k = np.clip(raw_k, low_k, high_k)

    x_v = vertices[:, 0]
    y_v = vertices[:, 1]
    faces_v = np.asarray(mesh.faces)

    trip = axes[4].tripcolor(
        x_v, y_v, faces_v, clipped_k, shading="gouraud", cmap="coolwarm"
    )
    axes[4].set_title(f"Gaussian Curvature K\nmean: {curv_res['gaussian_curvature_mean']:.4f}")
    axes[4].set_aspect("equal")
    fig.colorbar(trip, ax=axes[4], fraction=0.046, pad=0.04)
    axes[4].axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")

    return fig
