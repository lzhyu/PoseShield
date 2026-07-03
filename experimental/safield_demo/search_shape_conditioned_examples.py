"""Search visually compelling shape-conditioned SAField demo examples.

The script starts from colliding poses, tries shape pairs in beta in [-2, 2],
optimizes the same initial pose under each shape, renders the best candidates,
and optionally asks Qwen3-VL-8B-Instruct to score visual quality.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
PROJECT_TMP = REPO_ROOT / "tmp"
PROJECT_TMP.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_TMP / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_TMP / "cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DEMO_DIR))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from inference import optimize_pose
from network import SAFieldNetwork
from poseshield.common.utils import sixd_to_mesh


@dataclass
class Candidate:
    """Serializable record for a searched candidate and its metrics."""

    rank_score: float
    dataset_idx: int
    pair_idx: int
    pose_init: list[float]
    beta_a: list[float]
    beta_b: list[float]
    opt_theta_a: list[float]
    opt_theta_b: list[float]
    success_a: bool
    success_b: bool
    g_init_a: float
    g_init_b: float
    g_opt_a: float
    g_opt_b: float
    pose_l2: float
    move_l2_a: float
    move_l2_b: float
    mvd_a: float | None = None
    mvd_b: float | None = None
    pure_pose_mvd: float | None = None
    render_path: str | None = None
    vlm_score: float | None = None
    vlm_raw: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments and return an argparse namespace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to a colliding-pose dataset NPZ file.")
    parser.add_argument("--model_path", default=str(DEMO_DIR / "best_scc_model.pth"))
    parser.add_argument("--smpl_model_path", default=str(REPO_ROOT / "deps/body_models"))
    parser.add_argument("--examples_path", default=str(DEMO_DIR / "demo_examples.json"))
    parser.add_argument("--output_dir", default=str(DEMO_DIR / "artifacts"))
    parser.add_argument("--max_poses", type=int, default=40)
    parser.add_argument("--shape_pairs", type=int, default=8)
    parser.add_argument("--top_k_render", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max_itr", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vlm_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--use_vlm", action="store_true")
    parser.add_argument("--update_demo_examples", action="store_true")
    return parser.parse_args()


def load_safield(model_path: str, device: torch.device) -> SAFieldNetwork:
    """Load the SAField network checkpoint and return an eval-mode model."""
    model = SAFieldNetwork(
        theta_dim=126,
        beta_dim=10,
        hidden_dim=512,
        K=8,
        num_layers_g0=12,
        num_layers_phi=6,
        num_layers_shape=4,
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def field_value(model: SAFieldNetwork, theta: np.ndarray, beta: np.ndarray, device: torch.device) -> float:
    """Return the scalar SAField value g(theta, beta)."""
    with torch.no_grad():
        theta_t = torch.tensor(theta.reshape(1, -1), dtype=torch.float32, device=device)
        beta_t = torch.tensor(beta.reshape(1, -1), dtype=torch.float32, device=device)
        return float(model(theta_t, beta_t).item())


def shape_pairs(rng: np.random.RandomState, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate beta pairs inside [-2, 2] with intentionally visible contrast."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    def add(a: list[float]) -> None:
        beta_a = np.asarray(a, dtype=np.float32)
        beta_b = -beta_a
        pairs.append((np.clip(beta_a, -2.0, 2.0), np.clip(beta_b, -2.0, 2.0)))

    add([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    add([2.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    add([1.6, -1.6, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    while len(pairs) < count:
        beta_a = rng.uniform(-2.0, 2.0, size=10).astype(np.float32)
        beta_a[rng.rand(10) < 0.35] = 0.0
        if np.linalg.norm(beta_a) < 2.5:
            beta_a[0] = 2.0 if rng.rand() < 0.5 else -2.0
        pairs.append((beta_a, -beta_a))
    return pairs[:count]


def numeric_score(
    g_init_a: float,
    g_init_b: float,
    g_opt_a: float,
    g_opt_b: float,
    move_l2_a: float,
    move_l2_b: float,
    pose_l2: float,
    threshold: float,
) -> float:
    """Score a candidate before mesh rendering or VLM review."""
    if g_init_a >= 0.0 or g_init_b >= 0.0:
        return -1e6
    if g_opt_a < threshold - 0.005 or g_opt_b < threshold - 0.005:
        return -1e6

    move_penalty = 0.25 * (move_l2_a + move_l2_b)
    depth_bonus = min(abs(g_init_a), 0.2) + min(abs(g_init_b), 0.2)
    diff_bonus = min(pose_l2, 1.5)
    margin_bonus = min(g_opt_a - threshold, 0.05) + min(g_opt_b - threshold, 0.05)
    return 3.0 * diff_bonus + depth_bonus + margin_bonus - move_penalty


def select_render_candidates(candidates: list[Candidate], top_k: int) -> list[Candidate]:
    """Select a balanced render set with both high-score and modest-motion candidates."""
    if top_k <= 0:
        return []

    sorted_all = sorted(candidates, key=lambda c: c.rank_score, reverse=True)
    selected: list[Candidate] = []
    seen: set[tuple[int, int]] = set()

    def add_many(pool: list[Candidate], limit: int) -> None:
        for candidate in pool:
            key = (candidate.dataset_idx, candidate.pair_idx)
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= limit:
                break

    modest_motion = [
        c for c in sorted_all
        if max(c.move_l2_a, c.move_l2_b) <= 0.65 and c.pose_l2 >= 0.08
    ]
    clear_diff = [
        c for c in sorted_all
        if c.pose_l2 >= 0.18 and max(c.move_l2_a, c.move_l2_b) <= 0.90
    ]

    add_many(modest_motion, max(1, top_k // 2))
    add_many(clear_diff, max(len(selected), (3 * top_k) // 4))
    add_many(sorted_all, top_k)
    return selected[:top_k]


def render_candidate_grid(candidate: Candidate, smpl_model: Any, device: torch.device, out_path: Path) -> dict[str, float]:
    """Render initial/resolved meshes for both shapes and return vertex metrics."""
    beta_a_t = torch.tensor(candidate.beta_a, dtype=torch.float32, device=device).view(1, 10)
    beta_b_t = torch.tensor(candidate.beta_b, dtype=torch.float32, device=device).view(1, 10)
    pose_init = np.asarray(candidate.pose_init, dtype=np.float32).reshape(21, 6)
    pose_a = np.asarray(candidate.opt_theta_a, dtype=np.float32).reshape(21, 6)
    pose_b = np.asarray(candidate.opt_theta_b, dtype=np.float32).reshape(21, 6)

    verts_init_a, faces = sixd_to_mesh(smpl_model, pose_init, device=device, betas=beta_a_t)
    verts_opt_a, _ = sixd_to_mesh(smpl_model, pose_a, device=device, betas=beta_a_t)
    verts_init_b, _ = sixd_to_mesh(smpl_model, pose_init, device=device, betas=beta_b_t)
    verts_opt_b, _ = sixd_to_mesh(smpl_model, pose_b, device=device, betas=beta_b_t)
    verts_opt_b_on_a, _ = sixd_to_mesh(smpl_model, pose_b, device=device, betas=beta_a_t)

    mvd_a = float(np.mean(np.linalg.norm(verts_init_a - verts_opt_a, axis=-1)))
    mvd_b = float(np.mean(np.linalg.norm(verts_init_b - verts_opt_b, axis=-1)))
    pure_pose_mvd = float(np.mean(np.linalg.norm(verts_opt_a - verts_opt_b_on_a, axis=-1)))

    meshes = [
        ("A initial", verts_init_a),
        ("A resolved", verts_opt_a),
        ("B initial", verts_init_b),
        ("B resolved", verts_opt_b),
    ]
    all_vertices = np.concatenate([verts for _, verts in meshes], axis=0)
    min_vals = np.min(all_vertices, axis=0)
    max_vals = np.max(all_vertices, axis=0)
    mid = (min_vals + max_vals) / 2.0
    max_range = float(np.max(max_vals - min_vals))
    views = [(90, -90, "front"), (90, 0, "side"), (0, -90, "top")]

    fig = plt.figure(figsize=(13.5, 16.0))
    for row, (state_name, verts) in enumerate(meshes):
        for col, (elev, azim, view_name) in enumerate(views):
            ax = fig.add_subplot(4, 3, row * 3 + col + 1, projection="3d")
            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                triangles=faces,
                shade=True,
                edgecolor="none",
                color=[0.72, 0.72, 0.72],
            )
            ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
            ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
            ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
            ax.view_init(elev=elev, azim=azim)
            ax.axis("off")
            ax.set_title(f"{state_name} - {view_name}", fontsize=10)

    fig.suptitle(
        (
            f"dataset_idx={candidate.dataset_idx} | "
            f"MVD A={mvd_a * 100:.2f}cm B={mvd_b * 100:.2f}cm | "
            f"pose diff={pure_pose_mvd * 100:.2f}cm"
        ),
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"mvd_a": mvd_a, "mvd_b": mvd_b, "pure_pose_mvd": pure_pose_mvd}


def load_qwen3_vl(model_name: str, device: torch.device) -> tuple[Any, Any]:
    """Load Qwen3-VL and its processor for image scoring."""
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    except Exception:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model_cls.from_pretrained(model_name, dtype=dtype, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from generated text."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in VLM output: {text}")
    return json.loads(match.group(0))


def score_with_vlm(model: Any, processor: Any, image_path: str) -> tuple[float, str]:
    """Ask Qwen3-VL to score a rendered candidate and return score plus raw text."""
    prompt = (
        "You are selecting one demo example for shape-conditioned human self-collision resolution. "
        "The image is a 4x3 grid: rows are A initial, A resolved, B initial, B resolved; "
        "columns are front, side, top. Score whether the initial poses look natural but colliding, "
        "the resolved poses look collision-free, the A/B resolved poses differ visibly due to body shape, "
        "and the motion is not excessive. Reply only as JSON with keys "
        "naturalness, resolved, shape_difference, motion_reasonable, overall, reason. "
        "Scores must be integers from 1 to 5."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    parsed = extract_json_object(text)
    overall = float(parsed.get("overall", 0.0))
    components = [
        float(parsed.get("naturalness", 0.0)),
        float(parsed.get("resolved", 0.0)),
        float(parsed.get("shape_difference", 0.0)),
        float(parsed.get("motion_reasonable", 0.0)),
    ]
    score = 0.5 * overall + 0.5 * (sum(components) / len(components))
    return score, text


def write_json(path: Path, data: Any) -> None:
    """Write JSON data with stable formatting."""
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def update_demo_examples(path: Path, winner: Candidate) -> None:
    """Put the selected winner first in demo_examples.json, preserving old entries."""
    old_examples = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    selected = {
        "idx": winner.dataset_idx,
        "pose_init": winner.pose_init,
        "beta_thin": winner.beta_a,
        "beta_fat": winner.beta_b,
        "opt_theta_thin": winner.opt_theta_a,
        "opt_theta_fat": winner.opt_theta_b,
        "metadata": {
            "note": "Selected by Qwen3-VL shape-conditioned search; beta_thin/fat are demo shape A/B labels.",
            "rank_score": winner.rank_score,
            "vlm_score": winner.vlm_score,
            "g_init_a": winner.g_init_a,
            "g_init_b": winner.g_init_b,
            "g_opt_a": winner.g_opt_a,
            "g_opt_b": winner.g_opt_b,
            "mvd_a": winner.mvd_a,
            "mvd_b": winner.mvd_b,
            "pure_pose_mvd": winner.pure_pose_mvd,
        },
    }
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    merged = [selected] + old_examples[:4]
    write_json(path, merged)


def main() -> None:
    """Run the full search, render, optional VLM scoring, and optional update."""
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"qwen3_shape_search_{timestamp}"
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Output: {out_dir}")
    print(f"[Setup] Device: {device}")
    print(f"[Setup] Loading SAField from {args.model_path}")
    model = load_safield(args.model_path, device)

    data = np.load(args.dataset)
    poses = data["local_pose"][: args.max_poses]
    pairs = shape_pairs(rng, args.shape_pairs)
    print(f"[Search] poses={len(poses)} shape_pairs={len(pairs)}")

    candidates: list[Candidate] = []
    ledger_path = out_dir / "numeric_candidates.jsonl"
    with ledger_path.open("w", encoding="utf-8") as ledger:
        for pose_idx, pose_6d in enumerate(poses):
            theta_init = pose_6d.reshape(-1).astype(np.float32)
            for pair_idx, (beta_a, beta_b) in enumerate(pairs):
                g_init_a = field_value(model, theta_init, beta_a, device)
                g_init_b = field_value(model, theta_init, beta_b, device)
                if g_init_a >= 0.0 or g_init_b >= 0.0:
                    continue

                opt_a, success_a = optimize_pose(theta_init, beta_a, model, device, max_itr=args.max_itr, threshold=args.threshold)
                opt_b, success_b = optimize_pose(theta_init, beta_b, model, device, max_itr=args.max_itr, threshold=args.threshold)
                g_opt_a = field_value(model, opt_a, beta_a, device)
                g_opt_b = field_value(model, opt_b, beta_b, device)
                move_l2_a = float(np.linalg.norm(opt_a - theta_init))
                move_l2_b = float(np.linalg.norm(opt_b - theta_init))
                pose_l2 = float(np.linalg.norm(opt_a - opt_b))
                score = numeric_score(
                    g_init_a,
                    g_init_b,
                    g_opt_a,
                    g_opt_b,
                    move_l2_a,
                    move_l2_b,
                    pose_l2,
                    args.threshold,
                )
                if score <= -1e5:
                    continue
                candidate = Candidate(
                    rank_score=score,
                    dataset_idx=pose_idx,
                    pair_idx=pair_idx,
                    pose_init=theta_init.astype(float).tolist(),
                    beta_a=beta_a.astype(float).tolist(),
                    beta_b=beta_b.astype(float).tolist(),
                    opt_theta_a=opt_a.astype(float).tolist(),
                    opt_theta_b=opt_b.astype(float).tolist(),
                    success_a=bool(success_a),
                    success_b=bool(success_b),
                    g_init_a=g_init_a,
                    g_init_b=g_init_b,
                    g_opt_a=g_opt_a,
                    g_opt_b=g_opt_b,
                    pose_l2=pose_l2,
                    move_l2_a=move_l2_a,
                    move_l2_b=move_l2_b,
                )
                candidates.append(candidate)
                ledger.write(json.dumps(asdict(candidate)) + "\n")
                ledger.flush()
                print(
                    f"[Candidate] pose={pose_idx} pair={pair_idx} score={score:.3f} "
                    f"g=({g_init_a:.3f},{g_init_b:.3f})->({g_opt_a:.3f},{g_opt_b:.3f})"
                )

    candidates.sort(key=lambda c: c.rank_score, reverse=True)
    if not candidates:
        raise RuntimeError("No candidate passed numeric filtering.")

    import smplx

    render_candidates = select_render_candidates(candidates, args.top_k_render)
    print(f"[Render] Rendering {len(render_candidates)} balanced candidates")
    smpl_model = smplx.create(
        args.smpl_model_path,
        model_type="smplh",
        gender="male",
        ext="npz",
        use_pca=False,
    ).to(device)

    rendered: list[Candidate] = []
    for rank, candidate in enumerate(render_candidates):
        out_path = render_dir / f"rank{rank:02d}_idx{candidate.dataset_idx}_pair{candidate.pair_idx}.png"
        metrics = render_candidate_grid(candidate, smpl_model, device, out_path)
        candidate.mvd_a = metrics["mvd_a"]
        candidate.mvd_b = metrics["mvd_b"]
        candidate.pure_pose_mvd = metrics["pure_pose_mvd"]
        candidate.render_path = str(out_path.resolve())
        # Prefer visible but not excessive pose differences.
        diff = candidate.pure_pose_mvd
        move = max(candidate.mvd_a, candidate.mvd_b)
        candidate.rank_score += 20.0 * min(diff, 0.03) - 16.0 * max(move - 0.06, 0.0)
        rendered.append(candidate)
        print(
            f"[Render] rank={rank} image={out_path} "
            f"MVD=({candidate.mvd_a * 100:.2f},{candidate.mvd_b * 100:.2f})cm "
            f"pose_diff={candidate.pure_pose_mvd * 100:.2f}cm"
        )

    if args.use_vlm:
        print(f"[VLM] Loading {args.vlm_model}")
        vlm_model, processor = load_qwen3_vl(args.vlm_model, device)
        for candidate in rendered:
            assert candidate.render_path is not None
            try:
                vlm_score, raw = score_with_vlm(vlm_model, processor, candidate.render_path)
                candidate.vlm_score = vlm_score
                candidate.vlm_raw = raw
                candidate.rank_score += 3.0 * vlm_score
                print(f"[VLM] {candidate.render_path} score={vlm_score:.2f} raw={raw}")
            except Exception as exc:
                candidate.vlm_score = 0.0
                candidate.vlm_raw = f"VLM scoring failed: {exc!r}"
                print(f"[VLM][Error] {candidate.render_path}: {exc!r}")

    rendered.sort(key=lambda c: c.rank_score, reverse=True)
    winner = rendered[0]
    write_json(out_dir / "ranked_candidates.json", [asdict(c) for c in rendered])
    write_json(out_dir / "winner.json", asdict(winner))
    print(f"[Winner] dataset_idx={winner.dataset_idx} pair={winner.pair_idx} score={winner.rank_score:.3f}")
    print(f"[Winner] render={winner.render_path}")

    if args.update_demo_examples:
        update_demo_examples(Path(args.examples_path), winner)
        print(f"[Update] Updated {args.examples_path}")


if __name__ == "__main__":
    main()
