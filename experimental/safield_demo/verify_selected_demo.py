"""Verify the selected SAField demo example with exact FCL and shape overlays."""
from __future__ import annotations

import argparse
import json
import pickle
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

from poseshield.common.collision import self_collision_status
from poseshield.common.utils import sixd_to_mesh


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples_path", default=str(DEMO_DIR / "demo_examples.json"))
    parser.add_argument("--example_idx", type=int, default=0)
    parser.add_argument(
        "--smpl_model_path",
        default=str(REPO_ROOT / "deps/body_models"),
    )
    parser.add_argument("--distances_path", default=str(REPO_ROOT / "deps/distances.pkl"))
    parser.add_argument("--topology_threshold", type=int, default=40)
    parser.add_argument(
        "--output_dir",
        default=str(DEMO_DIR / "artifacts/release_example"),
    )
    return parser.parse_args()


def load_example(path: Path, example_idx: int) -> dict[str, Any]:
    """Load one demo example from JSON."""
    examples = json.loads(path.read_text(encoding="utf-8"))
    if example_idx < 0 or example_idx >= len(examples):
        raise IndexError(f"example_idx {example_idx} out of range for {len(examples)} examples")
    return examples[example_idx]


def make_meshes(example: dict[str, Any], smpl_model: Any, device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build initial and resolved meshes for both demo shapes."""
    pose_init = np.asarray(example["pose_init"], dtype=np.float32).reshape(21, 6)
    pose_a = np.asarray(example["opt_theta_thin"], dtype=np.float32).reshape(21, 6)
    pose_b = np.asarray(example["opt_theta_fat"], dtype=np.float32).reshape(21, 6)
    beta_a = torch.tensor(example["beta_thin"], dtype=torch.float32, device=device).view(1, 10)
    beta_b = torch.tensor(example["beta_fat"], dtype=torch.float32, device=device).view(1, 10)

    verts_a_init, faces = sixd_to_mesh(smpl_model, pose_init, device=device, betas=beta_a)
    verts_a_resolved, _ = sixd_to_mesh(smpl_model, pose_a, device=device, betas=beta_a)
    verts_b_init, _ = sixd_to_mesh(smpl_model, pose_init, device=device, betas=beta_b)
    verts_b_resolved, _ = sixd_to_mesh(smpl_model, pose_b, device=device, betas=beta_b)
    meshes = {
        "shape_a_initial": verts_a_init,
        "shape_a_resolved": verts_a_resolved,
        "shape_b_initial": verts_b_init,
        "shape_b_resolved": verts_b_resolved,
    }
    return meshes, faces


def fcl_report(
    meshes: dict[str, np.ndarray],
    faces: np.ndarray,
    distances: dict[tuple[int, int], int],
    topology_threshold: int,
) -> dict[str, dict[str, float | bool]]:
    """Run exact FCL self-collision checks for each mesh."""
    report: dict[str, dict[str, float | bool]] = {}
    for name, vertices in meshes.items():
        has_collision, total_penetration_depth = self_collision_status(
            vertices,
            faces,
            distances,
            topology_threshold=topology_threshold,
        )
        report[name] = {
            "has_collision": bool(has_collision),
            "collision_rate": None,
            "total_penetration_depth": float(total_penetration_depth),
        }
    return report


def set_equal_axes(ax: Any, vertices: np.ndarray) -> None:
    """Set equal 3D axes for a set of vertices."""
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    mid = (min_vals + max_vals) / 2.0
    radius = float(np.max(max_vals - min_vals) / 2.0)
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)
    ax.axis("off")


def plot_overlay(
    ax: Any,
    meshes: list[tuple[np.ndarray, str, tuple[float, float, float], float]],
    faces: np.ndarray,
    elev: float,
    azim: float,
    title: str,
) -> None:
    """Draw translucent mesh overlays in one 3D subplot."""
    all_vertices = np.concatenate([verts for verts, _, _, _ in meshes], axis=0)
    for verts, label, color, alpha in meshes:
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            color=color,
            alpha=alpha,
            linewidth=0.0,
            shade=True,
            label=label,
        )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    set_equal_axes(ax, all_vertices)


def render_shape_comparison(meshes: dict[str, np.ndarray], faces: np.ndarray, out_path: Path) -> None:
    """Render explicit A/B shape overlays and initial/resolved motion overlays."""
    views = [(90, -90, "front"), (90, 0, "side"), (0, -90, "top")]
    row_defs = [
        (
            "Initial A/B shape overlay",
            [
                (meshes["shape_a_initial"], "A", (0.20, 0.42, 0.85), 0.42),
                (meshes["shape_b_initial"], "B", (0.90, 0.42, 0.20), 0.42),
            ],
        ),
        (
            "Resolved A/B shape overlay",
            [
                (meshes["shape_a_resolved"], "A", (0.20, 0.42, 0.85), 0.42),
                (meshes["shape_b_resolved"], "B", (0.90, 0.42, 0.20), 0.42),
            ],
        ),
        (
            "Shape A initial/resolved motion",
            [
                (meshes["shape_a_initial"], "initial", (0.45, 0.45, 0.45), 0.30),
                (meshes["shape_a_resolved"], "resolved", (0.20, 0.70, 0.35), 0.48),
            ],
        ),
        (
            "Shape B initial/resolved motion",
            [
                (meshes["shape_b_initial"], "initial", (0.45, 0.45, 0.45), 0.30),
                (meshes["shape_b_resolved"], "resolved", (0.20, 0.70, 0.35), 0.48),
            ],
        ),
    ]

    fig = plt.figure(figsize=(13.5, 16.0))
    for row_idx, (row_title, row_meshes) in enumerate(row_defs):
        for col_idx, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(4, 3, row_idx * 3 + col_idx + 1, projection="3d")
            plot_overlay(ax, row_meshes, faces, elev, azim, f"{row_title} - {view_name}")
    fig.suptitle("Selected Demo: Shape Difference and Resolution Motion Overlays", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run FCL checks and render an explicit shape-comparison image."""
    args = parse_args()
    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    example = load_example(Path(args.examples_path), args.example_idx)
    smpl_model = smplx.create(
        args.smpl_model_path,
        model_type="smplh",
        gender="male",
        ext="npz",
        use_pca=False,
    ).to(device)
    with Path(args.distances_path).open("rb") as f:
        distances = pickle.load(f)

    meshes, faces = make_meshes(example, smpl_model, device)
    report = {
        "example_idx": args.example_idx,
        "dataset_idx": example["idx"],
        "topology_threshold": args.topology_threshold,
        "fcl": fcl_report(meshes, faces, distances, args.topology_threshold),
    }
    report_path = output_dir / "selected_fcl_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    comparison_path = output_dir / "selected_shape_comparison.png"
    render_shape_comparison(meshes, faces, comparison_path)

    print(json.dumps(report, indent=2))
    print(f"shape_comparison={comparison_path}")


if __name__ == "__main__":
    main()
