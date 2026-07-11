"""Export exact-FCL contact masks for optional motion render highlighting.

Prerequisite for Blender contact overlays: generate these masks from the same
original motion file, SMPL-H mesh topology, and TOPO threshold that will be used
for rendering. The render script consumes the saved face masks and does not
infer contacts visually.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import smplx
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.common.collision import (  # noqa: E402
    DEFAULT_MOTION_TOPOLOGY_THRESHOLD,
    load_topology_distances,
    self_collision_contact_faces,
)
from poseshield.hymotion.dno.dno_loss import (  # noqa: E402
    BODY_MODEL_PATH_,
    matrix_to_axis_angle_torch,
    rotation_6d_to_matrix_torch,
)
from poseshield.hymotion.utils.motion_format import load_motion  # noqa: E402


def dilate_faces(faces: np.ndarray, seed_faces: list[int], rings: int) -> list[int]:
    """Expand a face set by mesh-edge adjacency for more visible overlays."""
    selected = {int(face) for face in seed_faces}
    if rings <= 0 or not selected:
        return sorted(selected)

    vertex_to_faces: dict[int, set[int]] = {}
    for face_idx, face in enumerate(faces):
        for vertex_idx in face:
            vertex_to_faces.setdefault(int(vertex_idx), set()).add(int(face_idx))

    frontier = set(selected)
    for _ in range(rings):
        next_frontier: set[int] = set()
        for face_idx in frontier:
            for vertex_idx in faces[face_idx]:
                next_frontier.update(vertex_to_faces[int(vertex_idx)])
        next_frontier.difference_update(selected)
        selected.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export original-motion contact masks for website renders")
    parser.add_argument("--motions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tmp/website_runs/website_assets/motion_contacts")
    parser.add_argument("--distances", type=Path, default=PROJECT_ROOT / "deps/topology_distances_30_60.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--topology-threshold", type=int, default=DEFAULT_MOTION_TOPOLOGY_THRESHOLD)
    parser.add_argument(
        "--rings",
        type=int,
        default=1,
        help="Dilate exact contact faces by N mesh-adjacency rings for visibility. Use 0 for raw exact faces.",
    )
    parser.add_argument("--max-contacts", type=int, default=1_000_000)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
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
    del smpl_model, output, rotations, global_axis_angles, body_axis_angles, translation
    gc.collect()
    return vertices, faces


def export_one(
    motion_path: Path,
    output_dir: Path,
    distances: dict,
    device: torch.device,
    chunk_size: int,
    topology_threshold: int,
    rings: int,
    max_contacts: int,
    frame_offset: int,
    frame_stride: int,
) -> dict[str, object]:
    motion = load_motion(motion_path, translation_layout="xyz", rotation_joint_layout="root_first")
    frame_indices = list(range(frame_offset, len(motion), frame_stride))
    masks = None
    faces_ref = None
    raw_counts: list[int] = []
    marked_counts: list[int] = []
    penetration_depths: list[float] = []
    collision_frames: list[int] = []

    for batch_start in range(0, len(frame_indices), chunk_size):
        batch_indices = frame_indices[batch_start : batch_start + chunk_size]
        vertices, faces = motion_vertices(motion[batch_indices], device)
        if masks is None:
            faces_ref = faces
            masks = np.zeros((len(motion), len(faces)), dtype=np.bool_)
        for local_idx, frame_vertices in enumerate(vertices):
            frame_idx = batch_indices[local_idx]
            contact_faces, penetration_depth = self_collision_contact_faces(
                frame_vertices,
                faces,
                distances,
                topology_threshold=topology_threshold,
                max_contacts=max_contacts,
            )
            expanded = dilate_faces(faces, contact_faces, rings=rings)
            if expanded:
                masks[frame_idx, np.asarray(expanded, dtype=np.int64)] = True
                collision_frames.append(frame_idx)
            raw_counts.append(len(contact_faces))
            marked_counts.append(len(expanded))
            penetration_depths.append(float(penetration_depth))
        del vertices
        gc.collect()
        print(
            f"{motion_path.stem}: frames {batch_indices[0]}..{batch_indices[-1]}, "
            f"collision_frames={len(collision_frames)}",
            flush=True,
        )

    assert masks is not None and faces_ref is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{motion_path.stem}_contact_masks.npz"
    np.savez_compressed(mask_path, contact_masks=masks, faces=faces_ref)
    summary = {
        "stem": motion_path.stem,
        "motion": str(motion_path),
        "mask_path": str(mask_path),
        "num_frames": int(len(motion)),
        "num_collision_frames": int(len(collision_frames)),
        "collision_frame_indices": collision_frames,
        "topology_threshold": int(topology_threshold),
        "rings": int(rings),
        "max_contacts": int(max_contacts),
        "frame_offset": int(frame_offset),
        "frame_stride": int(frame_stride),
        "raw_contact_face_counts": raw_counts,
        "marked_face_counts": marked_counts,
        "max_penetration_depth": float(max(penetration_depths) if penetration_depths else 0.0),
        "mean_penetration_depth": float(np.mean(penetration_depths) if penetration_depths else 0.0),
    }
    summary_path = output_dir / f"{motion_path.stem}_contact_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    distances = load_topology_distances(args.distances)
    device = torch.device(args.device)
    summaries = [
        export_one(
            motion.resolve(),
            args.output_dir.resolve(),
            distances,
            device,
            args.chunk_size,
            args.topology_threshold,
            args.rings,
            args.max_contacts,
            args.frame_offset,
            args.frame_stride,
        )
        for motion in args.motions
    ]
    (args.output_dir / "manifest.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
