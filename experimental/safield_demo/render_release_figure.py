"""Render a release-quality SAField experimental figure without Blender."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch

from poseshield.common.utils import sixd_to_mesh


def set_equal_axes(ax: Any, vertices: np.ndarray, zoom: float = 0.88) -> None:
    """Set centered equal 3D axes for clean figure panels."""
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    mid = (min_vals + max_vals) / 2.0
    radius = float(np.max(max_vals - min_vals) / (2.0 * zoom))
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")


def draw_mesh(
    ax: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
    elev: float,
    azim: float,
    title: str,
) -> None:
    """Draw one shaded SMPL-H mesh panel."""
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color=color,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    ax.view_init(elev=elev, azim=azim)
    set_equal_axes(ax, vertices)
    ax.set_title(title, fontsize=13, pad=4)


def main() -> None:
    """Render a high-resolution figure for the code-release README."""
    examples_path = DEMO_DIR / "demo_examples.json"
    output_path = REPO_ROOT / "assets/safield_experimental_shape_demo.png"
    smpl_model_path = REPO_ROOT / "deps/body_models"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    example = json.loads(examples_path.read_text(encoding="utf-8"))[0]
    device = torch.device("cpu")
    smpl_model = smplx.create(
        smpl_model_path,
        model_type="smplh",
        gender="male",
        ext="npz",
        use_pca=False,
    ).to(device)

    pose_init = np.asarray(example["pose_init"], dtype=np.float32).reshape(21, 6)
    pose_a = np.asarray(example["opt_theta_thin"], dtype=np.float32).reshape(21, 6)
    pose_b = np.asarray(example["opt_theta_fat"], dtype=np.float32).reshape(21, 6)
    beta_a = torch.tensor(example["beta_thin"], dtype=torch.float32, device=device).view(1, 10)
    beta_b = torch.tensor(example["beta_fat"], dtype=torch.float32, device=device).view(1, 10)

    a_init, faces = sixd_to_mesh(smpl_model, pose_init, device=device, betas=beta_a)
    b_init, _ = sixd_to_mesh(smpl_model, pose_init, device=device, betas=beta_b)
    a_resolved, _ = sixd_to_mesh(smpl_model, pose_a, device=device, betas=beta_a)
    b_resolved, _ = sixd_to_mesh(smpl_model, pose_b, device=device, betas=beta_b)

    panels = [
        ("Shape A input", a_init, (0.12, 0.28, 0.72), 88, -85),
        ("Shape B input", b_init, (0.82, 0.33, 0.12), 88, -85),
        ("Shape A resolved", a_resolved, (0.12, 0.28, 0.72), 88, -85),
        ("Shape B resolved", b_resolved, (0.82, 0.33, 0.12), 88, -85),
    ]

    fig = plt.figure(figsize=(15.5, 6.2), facecolor="white")
    for i, (title, vertices, color, elev, azim) in enumerate(panels):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        draw_mesh(ax, vertices, faces, color, elev, azim, title)

    meta = example.get("metadata", {})
    fig.text(
        0.5,
        0.045,
        (
            "Experimental shape-aware collision resolution: same colliding pose, "
            "different body shapes, exact-FCL collision-free outputs"
        ),
        ha="center",
        fontsize=13,
        color="#333333",
    )
    fig.text(
        0.5,
        0.012,
        (
            f"shape difference: {meta.get('shape_mvd_resolved', 0.0) * 100:.2f} cm   "
            f"MVD: A {meta.get('mvd_a', 0.0) * 100:.2f} cm / B {meta.get('mvd_b', 0.0) * 100:.2f} cm"
        ),
        ha="center",
        fontsize=10,
        color="#666666",
    )
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.13, wspace=0.0)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
