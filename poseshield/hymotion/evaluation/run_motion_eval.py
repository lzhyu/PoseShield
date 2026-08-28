"""Score collision-resolved motions on Jitter, MFD, FSR, and RPD.

Evaluates motions that Stage 2 has already produced; it does not run the
optimization itself. Generate the outputs first (see the Evaluation section of
the README), then point this at the directory holding them:

    python -m poseshield.hymotion.evaluation.run_motion_eval \\
        --motion_dir data/motion_canonical \\
        --results_dir outputs/motion_eval \\
        --output_dir outputs/motion_eval

For every input motion `<stem>.npy` in `--motion_dir`, the matching Stage 2 run
is expected at `<results_dir>/<stem>/<stage2_subdir>/`.

Which frame is scored
---------------------
Stage 2 saves a snapshot every few optimization steps and, by default, also
re-ranks those snapshots with the exact-FCL collision check to pick the one it
writes as `optimized_motion.npy`. This evaluator deliberately scores the
**final optimization step** instead. That selection step chooses the checkpoint
minimizing exact penetration depth -- which is precisely what RPD reports -- so
scoring the selected checkpoint would let the reported metric partly reflect the
selector rather than the optimizer. Scoring the last step keeps the metric
independent of it.

Metric details, including the references each metric follows, are in the
corresponding modules under `metrics/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.common.motion_fk import DEFAULT_DISTANCES_PATH, motion_to_joints  # noqa: E402
from poseshield.hymotion.evaluation.metrics.foot_skating import (  # noqa: E402
    compute_foot_skating_ratio,
)
from poseshield.hymotion.evaluation.metrics.jitter import compute_jitter  # noqa: E402
from poseshield.hymotion.evaluation.metrics.motion_feature_distance import (  # noqa: E402
    MotionEncoderUnavailable,
    compute_mfd,
)
from poseshield.hymotion.evaluation.metrics.penetration import compute_rpd  # noqa: E402
from poseshield.hymotion.utils.motion_format import load_motion  # noqa: E402

DEFAULT_FPS = 30.0
NUM_BODY_JOINTS = 22
_STEP_PATTERN = re.compile(r"^step_(\d+)\.npy$")


def find_last_step_motion(stage2_dir: Path) -> Path | None:
    """Locate the final optimization step's motion snapshot in a Stage 2 output.

    The filename encodes the step index and therefore depends on how many steps
    Stage 2 ran, so the highest index present is used rather than a fixed name.

    Args:
        stage2_dir: A Stage 2 output directory, i.e. the one containing
            `checkpoint_motions/`.

    Returns:
        Path to the highest-numbered `step_*.npy`, or None if no snapshots were
        saved.
    """
    checkpoint_dir = stage2_dir / "checkpoint_motions"
    if not checkpoint_dir.is_dir():
        return None
    steps = [
        (int(m.group(1)), path)
        for path in checkpoint_dir.glob("step_*.npy")
        if (m := _STEP_PATTERN.match(path.name))
    ]
    if not steps:
        return None
    return max(steps)[1]


def substitute_root(motion_135: np.ndarray, reference_135: np.ndarray) -> np.ndarray:
    """Replace a motion's root rotation and translation with a reference's.

    Keeps the body pose (`[6:132]`) from `motion_135` while taking root rotation
    (`[0:6]`) and translation (`[132:135]`) from `reference_135`.

    WARNING -- this is load-bearing for Jitter, and it looks removable
    ---------------------------------------------------------------------
    This exists **only** so Jitter measures body articulation and nothing else.
    World-space joint position is a function of (root translation, root
    rotation, body pose), so any jerk already present in the root trajectory
    propagates identically into every joint. On these motions that root-borne
    jerk is a large fraction of the raw number, and it is *the same* in the
    input and the output -- so without this substitution Jitter mostly reports
    how jerky the input's root trajectory was, not what collision resolution
    did to the body.

    Do not "simplify" this away, and do not extend it to the other metrics:

    - **Never apply it to FSR.** Foot skating is only meaningful in world
      space. A planted foot that stays still while the root walks forward is
      correctly *not* skating; swapping in a different root trajectory would
      manufacture apparent sliding out of nothing.
    - **Do not zero the root instead of substituting it.** Zeroing root motion
      to a constant frame was tried and made Jitter *worse*, because it also
      removes the root's natural counter-rotation (e.g. hip micro-rotation that
      stabilizes leg swing in world space). Substituting a real trajectory
      keeps that; substituting the *reference's* trajectory keeps it identical
      on both sides of the comparison.
    - A constant per-motion offset (e.g. floor alignment) cancels under
      Jitter's finite-difference stencil, so it never interacts with this.

    Args:
        motion_135: Motion whose body pose is kept, shape `[T, 135]`.
        reference_135: Motion whose root trajectory is used, shape `[T, 135]`.

    Returns:
        Combined motion, shape `[T, 135]`.

    Raises:
        ValueError: If the two motions have different frame counts. Truncating
            silently would leave Jitter covering a different frame range than
            the other metrics in the same row.
    """
    motion_135 = np.asarray(motion_135, dtype=np.float32)
    reference_135 = np.asarray(reference_135, dtype=np.float32)
    if motion_135.shape[0] != reference_135.shape[0]:
        raise ValueError(
            f"substitute_root: frame count mismatch, motion has {motion_135.shape[0]} "
            f"frames but reference has {reference_135.shape[0]}."
        )
    out = motion_135.copy()
    out[:, 0:6] = reference_135[:, 0:6]
    out[:, 132:135] = reference_135[:, 132:135]
    return out


def score_motion(
    motion: np.ndarray,
    original: np.ndarray,
    device,
    encode_fn,
    distances_path=DEFAULT_DISTANCES_PATH,
) -> dict:
    """Compute all four metrics for one resolved motion.

    Jitter and FSR are deliberately measured on **different** joint sets:

    - Jitter uses joints posed with the *original* motion's root rotation and
      translation, so the only thing that can differ between input and output is
      local body articulation. Without this, any jerk already present in the
      root trajectory would leak identically into every joint and swamp the
      articulation signal the metric is meant to capture.
    - FSR uses raw world-space joints. Ground contact only means anything in
      world space: a planted foot that stays still while the root walks forward
      is correctly not skating, and substituting a different root trajectory
      would manufacture apparent sliding.

    Args:
        motion: Collision-resolved motion, shape `[T, 135]`.
        original: The motion it was resolved from, shape `[T, 135]`.
        device: torch device used for forward kinematics.
        encode_fn: Motion encoder for MFD, or None to report MFD as None.
        distances_path: Face-pair topology distance table used by RPD.

    Returns:
        Dict with keys `jitter`, `mfd`, `fsr`, `rpd`. `mfd` is None when no
        encoder is available.
    """
    joints_pose = motion_to_joints(substitute_root(motion, original), device)[:, :NUM_BODY_JOINTS]
    joints_world = motion_to_joints(motion, device)[:, :NUM_BODY_JOINTS]

    jitter = compute_jitter(joints_pose, fps=DEFAULT_FPS)
    fsr = compute_foot_skating_ratio(joints_world, fps=DEFAULT_FPS)
    rpd = compute_rpd(motion, device, distances_path=distances_path)
    try:
        mfd = compute_mfd(motion, original, encode_fn=encode_fn)
    except MotionEncoderUnavailable:
        mfd = None
    return {"jitter": jitter, "mfd": mfd, "fsr": fsr, "rpd": rpd}


def summarize(rows: list[dict]) -> dict:
    """Average the per-motion metrics, excluding motions that failed.

    Args:
        rows: Per-motion result dicts, including any with `failed=True`.

    Returns:
        Dict with `n` (motions actually scored), `n_failed`, and the mean of
        each metric. Metric values are None when nothing could be averaged.
    """
    ok = [r for r in rows if not r.get("failed", False)]
    summary: dict = {"n": len(ok), "n_failed": len(rows) - len(ok)}
    for key in ("jitter", "mfd", "fsr", "rpd"):
        values = [r[key] for r in ok if r.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary


def parse_args() -> argparse.Namespace:
    """Parse motion evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Score collision-resolved motions on Jitter, MFD, FSR, and RPD."
    )
    parser.add_argument(
        "--motion_dir", type=Path, required=True,
        help="Directory of input motions (*.npy) that were resolved.",
    )
    parser.add_argument(
        "--results_dir", type=Path, required=True,
        help="Directory of Stage 2 outputs, one subdirectory per input motion stem.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Where to write motion_eval_results.json. Defaults to --results_dir.",
    )
    parser.add_argument(
        "--stage2_subdir", type=str, default="stage2",
        help="Stage 2 subdirectory inside each per-motion result directory.",
    )
    parser.add_argument(
        "--distances", type=Path, default=DEFAULT_DISTANCES_PATH,
        help="Face-pair topology distance table used by the exact collision check.",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Score only the first N motions. 0 (default) scores all of them.",
    )
    parser.add_argument(
        "--no_mfd", action="store_true",
        help="Skip MFD, which needs an optional third-party motion encoder download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = args.output_dir or args.results_dir

    encode_fn = None
    if not args.no_mfd:
        # Build the encoder up front so a missing download fails once, here, rather
        # than once per motion after the user has already waited through the run.
        from poseshield.hymotion.evaluation.motion_features import motionclip_encoder

        motionclip_encoder._ENCODER_SINGLETON = motionclip_encoder.MotionFeatureEncoder(device=device)
        encode_fn = motionclip_encoder.encode

    motion_paths = sorted(args.motion_dir.glob("*.npy"))
    if not motion_paths:
        raise SystemExit(f"No motions found in {args.motion_dir}")
    if args.limit:
        motion_paths = motion_paths[: args.limit]

    rows: list[dict] = []
    for motion_path in motion_paths:
        stage2_dir = args.results_dir / motion_path.stem / args.stage2_subdir
        start = time.time()
        try:
            last_step = find_last_step_motion(stage2_dir)
            if last_step is None:
                raise FileNotFoundError(
                    f"No checkpoint_motions/step_*.npy under {stage2_dir}. Stage 2 saves "
                    "these when --save_checkpoint_motions_every > 0 (its default is 10); "
                    "re-run Stage 2 for this motion if the directory is missing."
                )
            original = load_motion(motion_path)
            resolved = load_motion(last_step)
            metrics = score_motion(
                resolved, original, device, encode_fn, distances_path=args.distances
            )
        except Exception as exc:  # noqa: BLE001 - report and continue to the next motion
            print(f"[FAIL] {motion_path.name}: {exc}")
            rows.append({"motion": motion_path.name, "failed": True, "error": str(exc)})
            continue

        metrics.update({"motion": motion_path.name, "failed": False,
                        "elapsed_s": time.time() - start})
        rows.append(metrics)
        mfd_str = f"{metrics['mfd']:.4f}" if metrics["mfd"] is not None else "   n/a"
        print(
            f"{motion_path.name:35s} jitter={metrics['jitter']:.4f} mfd={mfd_str} "
            f"fsr={metrics['fsr']:6.2f}% rpd={metrics['rpd']:.4f}"
        )

    summary = summarize(rows)
    print("\n=== Motion collision resolution ===")
    print(f"motions scored: {summary['n']}" + (f" ({summary['n_failed']} failed)" if summary["n_failed"] else ""))
    for key, label in (("jitter", "Jitter"), ("mfd", "MFD"), ("fsr", "FSR(%)"), ("rpd", "RPD")):
        value = summary[key]
        print(f"  {label:8s} {'n/a' if value is None else f'{value:.4f}'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "motion_eval_results.json"
    with out_path.open("w") as handle:
        json.dump({"per_motion": rows, "summary": summary}, handle, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
