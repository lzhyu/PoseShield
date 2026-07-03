"""Try stronger A/B body-shape pairs for the selected SAField demo pose."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
sys.path.insert(0, str(DEMO_DIR))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch

from inference import optimize_pose
from network import SAFieldNetwork
from poseshield.common.collision import self_collision_status
from poseshield.common.utils import sixd_to_mesh


@dataclass
class StrongShapeCandidate:
    """Record for a stronger-shape candidate and its validation metrics."""

    name: str
    beta_a: list[float]
    beta_b: list[float]
    opt_theta_a: list[float]
    opt_theta_b: list[float]
    g_init_a: float
    g_init_b: float
    g_opt_a: float
    g_opt_b: float
    fcl_a_initial: bool
    fcl_a_resolved: bool
    fcl_b_initial: bool
    fcl_b_resolved: bool
    penetration_a_initial: float
    penetration_a_resolved: float
    penetration_b_initial: float
    penetration_b_resolved: float
    mvd_a: float
    mvd_b: float
    pure_pose_mvd: float
    shape_mvd_initial: float
    shape_mvd_resolved: float
    score: float
    render_path: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples_path", default=str(DEMO_DIR / "demo_examples.json"))
    parser.add_argument("--example_idx", type=int, default=0)
    parser.add_argument("--model_path", default=str(DEMO_DIR / "best_scc_model.pth"))
    parser.add_argument(
        "--smpl_model_path",
        default=str(REPO_ROOT / "deps/body_models"),
    )
    parser.add_argument("--distances_path", default=str(REPO_ROOT / "deps/distances.pkl"))
    parser.add_argument("--output_dir", default=str(DEMO_DIR / "artifacts/release_example"))
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max_itr", type=int, default=100)
    parser.add_argument("--topology_threshold", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--update_demo_examples", action="store_true")
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device) -> SAFieldNetwork:
    """Load the SAField model in eval mode."""
    model = SAFieldNetwork(
        theta_dim=126,
        beta_dim=10,
        hidden_dim=512,
        K=8,
        num_layers_g0=12,
        num_layers_phi=6,
        num_layers_shape=4,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def field_value(model: SAFieldNetwork, theta: np.ndarray, beta: np.ndarray, device: torch.device) -> float:
    """Evaluate SAField g(theta, beta)."""
    with torch.no_grad():
        theta_t = torch.tensor(theta.reshape(1, -1), dtype=torch.float32, device=device)
        beta_t = torch.tensor(beta.reshape(1, -1), dtype=torch.float32, device=device)
        return float(model(theta_t, beta_t).item())


def stronger_beta_pairs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return fixed high-contrast beta pairs inside [-2, 2]."""
    raw_pairs = [
        ("amplified_current", [0.0, 2.0, -2.0, 0.8, 2.0, -2.0, 0.8, 0.0, 2.0, 0.0]),
        ("height_width_contrast", [2.0, 1.8, -1.8, 0.0, 2.0, -1.5, 0.0, 0.0, 1.5, 0.0]),
        ("mass_distribution_contrast", [1.6, 2.0, -2.0, 1.5, -1.5, 2.0, -1.2, 0.0, 1.8, -1.0]),
        ("simple_extreme", [2.0, 2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 0.0, 2.0, 0.0]),
    ]
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, beta in raw_pairs:
        beta_a = np.clip(np.asarray(beta, dtype=np.float32), -2.0, 2.0)
        pairs.append((name, beta_a, -beta_a))
    return pairs


def make_mesh(
    smpl_model: Any,
    pose: np.ndarray,
    beta: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one SMPL-H mesh from 6D pose and beta."""
    beta_t = torch.tensor(beta, dtype=torch.float32, device=device).view(1, 10)
    return sixd_to_mesh(smpl_model, pose.reshape(21, 6), device=device, betas=beta_t)


def fcl_status(
    vertices: np.ndarray,
    faces: np.ndarray,
    distances: dict[tuple[int, int], int],
    topology_threshold: int,
) -> tuple[bool, float]:
    """Return exact self-collision flag and total penetration depth."""
    has_collision, penetration = self_collision_status(
        vertices,
        faces,
        distances,
        topology_threshold=topology_threshold,
    )
    return bool(has_collision), float(penetration)


def set_axes(ax: Any, vertices: np.ndarray) -> None:
    """Use common equal axes for one 3D subplot."""
    min_vals = vertices.min(axis=0)
    max_vals = vertices.max(axis=0)
    mid = (min_vals + max_vals) / 2.0
    radius = float(np.max(max_vals - min_vals) / 2.0)
    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)
    ax.axis("off")


def render_separate_grid(
    meshes: dict[str, np.ndarray],
    faces: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Render A/B separately in adjacent columns, without overlays."""
    views = [(90, -90, "front"), (90, 0, "side"), (0, -90, "top")]
    panels = [
        ("Shape A initial", "a_initial", (0.55, 0.66, 0.90)),
        ("Shape B initial", "b_initial", (0.90, 0.62, 0.48)),
        ("Shape A resolved", "a_resolved", (0.35, 0.58, 0.95)),
        ("Shape B resolved", "b_resolved", (0.95, 0.48, 0.30)),
    ]
    all_vertices = np.concatenate(list(meshes.values()), axis=0)

    fig = plt.figure(figsize=(18.0, 10.0))
    for row, (row_title, key, color) in enumerate(panels):
        for col, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(4, 3, row * 3 + col + 1, projection="3d")
            verts = meshes[key]
            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                triangles=faces,
                color=color,
                linewidth=0.0,
                shade=True,
            )
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{row_title} - {view_name}", fontsize=10)
            set_axes(ax, all_vertices)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_pair(
    name: str,
    pose_init: np.ndarray,
    beta_a: np.ndarray,
    beta_b: np.ndarray,
    model: SAFieldNetwork,
    smpl_model: Any,
    distances: dict[tuple[int, int], int],
    device: torch.device,
    output_dir: Path,
    threshold: float,
    max_itr: int,
    topology_threshold: int,
) -> StrongShapeCandidate:
    """Optimize and exact-FCL validate one stronger shape pair."""
    g_init_a = field_value(model, pose_init, beta_a, device)
    g_init_b = field_value(model, pose_init, beta_b, device)
    opt_a, _ = optimize_pose(pose_init, beta_a, model, device, max_itr=max_itr, threshold=threshold)
    opt_b, _ = optimize_pose(pose_init, beta_b, model, device, max_itr=max_itr, threshold=threshold)
    g_opt_a = field_value(model, opt_a, beta_a, device)
    g_opt_b = field_value(model, opt_b, beta_b, device)

    verts_a_init, faces = make_mesh(smpl_model, pose_init, beta_a, device)
    verts_a_resolved, _ = make_mesh(smpl_model, opt_a, beta_a, device)
    verts_b_init, _ = make_mesh(smpl_model, pose_init, beta_b, device)
    verts_b_resolved, _ = make_mesh(smpl_model, opt_b, beta_b, device)

    fcl_a_initial, pen_a_initial = fcl_status(verts_a_init, faces, distances, topology_threshold)
    fcl_a_resolved, pen_a_resolved = fcl_status(verts_a_resolved, faces, distances, topology_threshold)
    fcl_b_initial, pen_b_initial = fcl_status(verts_b_init, faces, distances, topology_threshold)
    fcl_b_resolved, pen_b_resolved = fcl_status(verts_b_resolved, faces, distances, topology_threshold)

    mvd_a = float(np.mean(np.linalg.norm(verts_a_init - verts_a_resolved, axis=-1)))
    mvd_b = float(np.mean(np.linalg.norm(verts_b_init - verts_b_resolved, axis=-1)))
    verts_b_resolved_on_a, _ = make_mesh(smpl_model, opt_b, beta_a, device)
    pure_pose_mvd = float(np.mean(np.linalg.norm(verts_a_resolved - verts_b_resolved_on_a, axis=-1)))
    shape_mvd_initial = float(np.mean(np.linalg.norm(verts_a_init - verts_b_init, axis=-1)))
    shape_mvd_resolved = float(np.mean(np.linalg.norm(verts_a_resolved - verts_b_resolved, axis=-1)))

    valid_fcl = fcl_a_initial and fcl_b_initial and not fcl_a_resolved and not fcl_b_resolved
    resolved_by_field = g_opt_a >= threshold - 0.005 and g_opt_b >= threshold - 0.005
    move_penalty = 10.0 * max(max(mvd_a, mvd_b) - 0.06, 0.0)
    score = (
        4.0 * shape_mvd_resolved
        + 2.0 * pure_pose_mvd
        - move_penalty
        + (10.0 if valid_fcl else -10.0)
        + (2.0 if resolved_by_field else -2.0)
    )

    meshes = {
        "a_initial": verts_a_init,
        "a_resolved": verts_a_resolved,
        "b_initial": verts_b_init,
        "b_resolved": verts_b_resolved,
    }
    render_path = output_dir / f"strong_shape_{name}.png"
    render_title = (
        f"{name} | shape diff={shape_mvd_resolved * 100:.2f}cm | "
        f"MVD A={mvd_a * 100:.2f}cm B={mvd_b * 100:.2f}cm | "
        f"pose diff={pure_pose_mvd * 100:.2f}cm"
    )
    render_separate_grid(meshes, faces, render_path, render_title)

    return StrongShapeCandidate(
        name=name,
        beta_a=beta_a.astype(float).tolist(),
        beta_b=beta_b.astype(float).tolist(),
        opt_theta_a=opt_a.astype(float).tolist(),
        opt_theta_b=opt_b.astype(float).tolist(),
        g_init_a=g_init_a,
        g_init_b=g_init_b,
        g_opt_a=g_opt_a,
        g_opt_b=g_opt_b,
        fcl_a_initial=fcl_a_initial,
        fcl_a_resolved=fcl_a_resolved,
        fcl_b_initial=fcl_b_initial,
        fcl_b_resolved=fcl_b_resolved,
        penetration_a_initial=pen_a_initial,
        penetration_a_resolved=pen_a_resolved,
        penetration_b_initial=pen_b_initial,
        penetration_b_resolved=pen_b_resolved,
        mvd_a=mvd_a,
        mvd_b=mvd_b,
        pure_pose_mvd=pure_pose_mvd,
        shape_mvd_initial=shape_mvd_initial,
        shape_mvd_resolved=shape_mvd_resolved,
        score=score,
        render_path=str(render_path.resolve()),
    )


def update_demo_examples(path: Path, example_idx: int, base_example: dict[str, Any], selected: StrongShapeCandidate) -> None:
    """Update the selected example with stronger shapes and metadata."""
    examples = json.loads(path.read_text(encoding="utf-8"))
    updated = dict(base_example)
    updated["beta_thin"] = selected.beta_a
    updated["beta_fat"] = selected.beta_b
    updated["opt_theta_thin"] = selected.opt_theta_a
    updated["opt_theta_fat"] = selected.opt_theta_b
    metadata = dict(updated.get("metadata", {}))
    metadata.update(
        {
            "selection": "stronger_shape_exact_fcl_verified",
            "selection_reason": "A/B shapes were strengthened within [-2, 2] and re-optimized; exact FCL verifies both initial meshes collide and both resolved meshes are collision-free.",
            "strong_shape_name": selected.name,
            "strong_shape_render_path": selected.render_path,
            "strong_shape_score": selected.score,
            "g_init_a": selected.g_init_a,
            "g_init_b": selected.g_init_b,
            "g_opt_a": selected.g_opt_a,
            "g_opt_b": selected.g_opt_b,
            "mvd_a": selected.mvd_a,
            "mvd_b": selected.mvd_b,
            "pure_pose_mvd": selected.pure_pose_mvd,
            "shape_mvd_initial": selected.shape_mvd_initial,
            "shape_mvd_resolved": selected.shape_mvd_resolved,
            "fcl_shape_a_initial_collision": selected.fcl_a_initial,
            "fcl_shape_a_resolved_collision": selected.fcl_a_resolved,
            "fcl_shape_b_initial_collision": selected.fcl_b_initial,
            "fcl_shape_b_resolved_collision": selected.fcl_b_resolved,
        }
    )
    updated["metadata"] = metadata
    examples[example_idx] = updated
    path.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run stronger-shape search and optionally update demo_examples.json."""
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / "strong_shape"
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_path = Path(args.examples_path)
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    base_example = examples[args.example_idx]
    pose_init = np.asarray(base_example["pose_init"], dtype=np.float32).reshape(-1)

    model = load_model(Path(args.model_path), device)
    smpl_model = smplx.create(
        args.smpl_model_path,
        model_type="smplh",
        gender="male",
        ext="npz",
        use_pca=False,
    ).to(device)
    with Path(args.distances_path).open("rb") as f:
        distances = pickle.load(f)

    candidates = []
    for name, beta_a, beta_b in stronger_beta_pairs():
        candidate = evaluate_pair(
            name,
            pose_init,
            beta_a,
            beta_b,
            model,
            smpl_model,
            distances,
            device,
            output_dir,
            args.threshold,
            args.max_itr,
            args.topology_threshold,
        )
        candidates.append(candidate)
        print(
            f"[Candidate] {name} score={candidate.score:.3f} "
            f"fcl=({candidate.fcl_a_initial}->{candidate.fcl_a_resolved},"
            f"{candidate.fcl_b_initial}->{candidate.fcl_b_resolved}) "
            f"MVD=({candidate.mvd_a * 100:.2f},{candidate.mvd_b * 100:.2f})cm "
            f"shape_diff={candidate.shape_mvd_resolved * 100:.2f}cm "
            f"render={candidate.render_path}"
        )

    valid = [
        c
        for c in candidates
        if c.fcl_a_initial
        and c.fcl_b_initial
        and not c.fcl_a_resolved
        and not c.fcl_b_resolved
        and c.g_opt_a >= args.threshold - 0.005
        and c.g_opt_b >= args.threshold - 0.005
    ]
    if not valid:
        raise RuntimeError("No stronger-shape candidate passed exact FCL and field checks.")

    selected = max(valid, key=lambda c: c.score)
    report = {
        "selected": asdict(selected),
        "candidates": [asdict(c) for c in sorted(candidates, key=lambda x: x.score, reverse=True)],
    }
    report_path = output_dir / "strong_shape_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[Selected] {selected.name} score={selected.score:.3f}")
    print(f"[Selected] render={selected.render_path}")
    print(f"[Report] {report_path}")

    if args.update_demo_examples:
        update_demo_examples(examples_path, args.example_idx, base_example, selected)
        print(f"[Update] Updated {examples_path}")


if __name__ == "__main__":
    main()
