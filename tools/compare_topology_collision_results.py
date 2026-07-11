"""Compare legacy and compact topology filters on actual meshes/motions."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import smplx
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.common.collision import load_topology_distances, self_collision_status  # noqa: E402
from poseshield.hymotion.dno.dno_loss import (  # noqa: E402
    BODY_MODEL_PATH_,
    matrix_to_axis_angle_torch,
    rotation_6d_to_matrix_torch,
)
from poseshield.hymotion.utils.motion_format import load_motion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=PROJECT_ROOT / "deps/distances.pkl")
    parser.add_argument("--compact", type=Path, default=PROJECT_ROOT / "deps/topology_distances_30_60.npz")
    parser.add_argument("--motions", type=Path, nargs="*", default=[])
    parser.add_argument("--motion-threshold", type=int, default=40)
    parser.add_argument("--motion-frame-stride", type=int, default=30)
    parser.add_argument("--pose-objs", type=Path, nargs="*", default=[])
    parser.add_argument("--pose-threshold", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def motion_vertices(motion: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    rotations = torch.from_numpy(motion[:, :132]).float().to(device)
    root_rotation = rotations[:, :6]
    body_rotation = rotations[:, 6:].reshape(-1, 21, 6)
    global_axis_angles = matrix_to_axis_angle_torch(rotation_6d_to_matrix_torch(root_rotation))
    body_axis_angles = matrix_to_axis_angle_torch(rotation_6d_to_matrix_torch(body_rotation)).flatten(1)
    translation = torch.from_numpy(motion[:, 132:135]).float().to(device)

    smpl_model = smplx.create(
        BODY_MODEL_PATH_,
        model_type="smplh",
        gender="neutral",
        ext="npz",
        use_pca=False,
        batch_size=len(motion),
    ).to(device)
    with torch.no_grad():
        output = smpl_model(
            global_orient=global_axis_angles,
            body_pose=body_axis_angles,
            transl=translation,
            return_verts=True,
        )
    vertices = output.vertices.cpu().numpy()
    faces = smpl_model.faces
    return vertices, faces


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:4]:
                    face.append(int(token.split("/")[0]) - 1)
                faces.append(face)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def compare_mesh(vertices, faces, legacy, compact, threshold):
    legacy_flag, legacy_depth = self_collision_status(
        vertices,
        faces,
        legacy,
        topology_threshold=threshold,
    )
    compact_flag, compact_depth = self_collision_status(
        vertices,
        faces,
        compact,
        topology_threshold=threshold,
    )
    return {
        "legacy_collision": bool(legacy_flag),
        "compact_collision": bool(compact_flag),
        "legacy_penetration_depth": float(legacy_depth),
        "compact_penetration_depth": float(compact_depth),
        "collision_match": bool(legacy_flag) == bool(compact_flag),
        "penetration_depth_match": abs(float(legacy_depth) - float(compact_depth)) <= 1e-12,
    }


def main() -> int:
    args = parse_args()
    with args.legacy.open("rb") as handle:
        legacy = pickle.load(handle)
    compact = load_topology_distances(args.compact)
    device = torch.device(args.device)

    results = []
    for motion_path in args.motions:
        motion = load_motion(motion_path, translation_layout="xyz", rotation_joint_layout="root_first")
        frame_indices = list(range(0, len(motion), args.motion_frame_stride))
        if frame_indices[-1] != len(motion) - 1:
            frame_indices.append(len(motion) - 1)
        vertices, faces = motion_vertices(motion[frame_indices], device)
        for local_idx, frame_idx in enumerate(frame_indices):
            result = compare_mesh(vertices[local_idx], faces, legacy, compact, args.motion_threshold)
            result.update(
                {
                    "kind": "motion",
                    "sample": str(motion_path),
                    "frame": int(frame_idx),
                    "threshold": int(args.motion_threshold),
                }
            )
            results.append(result)

    for obj_path in args.pose_objs:
        vertices, faces = load_obj_mesh(obj_path)
        result = compare_mesh(vertices, faces, legacy, compact, args.pose_threshold)
        result.update(
            {
                "kind": "pose_obj",
                "sample": str(obj_path),
                "frame": None,
                "threshold": int(args.pose_threshold),
            }
        )
        results.append(result)

    mismatches = [
        row
        for row in results
        if not row["collision_match"] or not row["penetration_depth_match"]
    ]
    report = {
        "legacy": str(args.legacy),
        "compact": str(args.compact),
        "num_results": len(results),
        "num_mismatches": len(mismatches),
        "mismatches": mismatches,
        "results": results,
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
