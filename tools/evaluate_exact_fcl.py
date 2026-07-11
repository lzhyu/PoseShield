"""Evaluate exact SMPL-H mesh self-collisions with python-fcl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.common.collision import DEFAULT_MOTION_TOPOLOGY_THRESHOLD, load_topology_distances
from poseshield.hymotion.utils.motion_format import load_motion


def parse_args() -> argparse.Namespace:
    """Parse exact collision evaluation arguments."""
    parser = argparse.ArgumentParser(description="Exact SMPL-H mesh/FCL collision check")
    parser.add_argument("--motion", type=Path, required=True, help="Canonical [frames, 135] motion")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and plot outputs")
    parser.add_argument("--distances", type=Path, default=PROJECT_ROOT / "deps/topology_distances_30_60.npz")
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA when available")
    parser.add_argument(
        "--topology-threshold",
        type=int,
        default=DEFAULT_MOTION_TOPOLOGY_THRESHOLD,
        help="Topological face-distance threshold for motion exact-FCL checks.",
    )
    parser.add_argument(
        "--allow-collisions-exit-zero",
        action="store_true",
        help=(
            "Write exact-FCL collision results but do not fail the process when "
            "collisions remain. Useful for batch aggregation where collided "
            "outputs are still valid negative examples."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Generate meshes and save per-frame exact FCL penetration statistics."""
    args = parse_args()
    motion_path = args.motion.resolve()
    output_dir = args.output_dir.resolve()
    distance_path = args.distances.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    motion = load_motion(
        motion_path,
        translation_layout="xyz",
        rotation_joint_layout="root_first",
    )
    if not distance_path.is_file():
        raise FileNotFoundError(distance_path)

    import smplx
    import torch
    from tqdm import tqdm

    from poseshield.common.collision import self_collision_status
    from poseshield.hymotion.dno.dno_loss import (
        BODY_MODEL_PATH_,
        matrix_to_axis_angle_torch,
        rotation_6d_to_matrix_torch,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rotations = torch.from_numpy(motion[:, :132]).float().to(device)
    root_rotation = rotations[:, :6]
    body_rotation = rotations[:, 6:].reshape(-1, 21, 6)
    global_axis_angles = matrix_to_axis_angle_torch(
        rotation_6d_to_matrix_torch(root_rotation)
    )
    body_axis_angles = matrix_to_axis_angle_torch(
        rotation_6d_to_matrix_torch(body_rotation)
    ).flatten(1)
    translation = torch.from_numpy(motion[:, 132:135]).float().to(device)

    smpl_model = smplx.create(
        BODY_MODEL_PATH_,
        model_type="smplh",
        gender="neutral",
        ext="npz",
        use_pca=False,
        batch_size=motion.shape[0],
    ).to(device)
    with torch.no_grad():
        smpl_output = smpl_model(
            global_orient=global_axis_angles,
            body_pose=body_axis_angles,
            transl=translation,
            return_verts=True,
        )
    vertices = smpl_output.vertices.cpu().numpy()
    faces = smpl_model.faces
    distances = load_topology_distances(distance_path)

    collision_flags: list[bool] = []
    penetration_depths: list[float] = []
    for frame_vertices in tqdm(vertices, desc="Exact FCL collision check"):
        has_collision, penetration_depth = self_collision_status(
            frame_vertices,
            faces,
            distances,
            topology_threshold=args.topology_threshold,
        )
        collision_flags.append(bool(has_collision))
        penetration_depths.append(float(penetration_depth))

    penetration_array = np.asarray(penetration_depths, dtype=np.float64)
    collision_frames = np.flatnonzero(collision_flags).tolist()
    results = {
        "motion": str(motion_path),
        "num_frames": int(motion.shape[0]),
        "exact_collision_free": len(collision_frames) == 0,
        "num_collision_frames": len(collision_frames),
        "collision_frame_ratio": len(collision_frames) / motion.shape[0],
        "collision_frame_indices": collision_frames,
        "topology_threshold": int(args.topology_threshold),
        "mean_penetration_depth": float(penetration_array.mean()),
        "max_penetration_depth": float(penetration_array.max()),
        "total_penetration_depth": float(penetration_array.sum()),
        "penetration_depth_seq": penetration_depths,
    }
    result_path = output_dir / "exact_fcl_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(penetration_array, color="red", linewidth=1.2)
    axis.set_xlabel("Frame")
    axis.set_ylabel("Total penetration depth")
    axis.set_title("Exact SMPL-H mesh/FCL penetration depth")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(output_dir / "exact_fcl_penetration_depth.png", dpi=150)
    plt.close(figure)

    print(f"EXACT_FCL collision_frames={len(collision_frames)}/{motion.shape[0]}")
    print(f"EXACT_FCL exact_collision_free={len(collision_frames) == 0}")
    print(f"EXACT_FCL mean_penetration_depth={penetration_array.mean():.9f}")
    print(f"EXACT_FCL max_penetration_depth={penetration_array.max():.9f}")
    print(f"EXACT_FCL total_penetration_depth={penetration_array.sum():.9f}")
    print(f"EXACT_FCL results={result_path}")
    if collision_frames and not args.allow_collisions_exit_zero:
        raise RuntimeError(
            f"Exact FCL found collisions in {len(collision_frames)} frames"
        )


if __name__ == "__main__":
    main()
