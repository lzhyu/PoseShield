"""Render a side-by-side original/optimized motion MP4 with Blender."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import smplx
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.hymotion.dno.dno_loss import (
    BODY_MODEL_PATH_,
    matrix_to_axis_angle_torch,
    rotation_6d_to_matrix_torch,
)
from poseshield.hymotion.utils.motion_format import load_motion


def parse_args() -> argparse.Namespace:
    """Parse Blender rendering arguments."""
    parser = argparse.ArgumentParser(description="Render a high-quality SMPL-H motion MP4")
    parser.add_argument("--original", type=Path, required=True, help="Original canonical motion")
    parser.add_argument("--optimized", type=Path, required=True, help="Optimized canonical motion")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--blender-path", type=Path, required=True, help="Path to the Blender binary")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="BLENDER_EEVEE")
    return parser.parse_args()


def run_forward_kinematics(motion: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Compute SMPL-H vertices and faces for a public canonical motion."""
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
        output = smpl_model(
            global_orient=global_axis_angles,
            body_pose=body_axis_angles,
            transl=translation,
            return_verts=True,
        )
    return output.vertices.cpu().numpy(), smpl_model.faces


def main() -> None:
    """Create a side-by-side MP4 with Blender and ffmpeg."""
    args = parse_args()
    original = load_motion(args.original)
    optimized = load_motion(args.optimized)
    if original.shape != optimized.shape:
        raise ValueError(f"Motion shapes differ: {original.shape} != {optimized.shape}")
    if not np.array_equal(original[:, 132:135], optimized[:, 132:135]):
        raise AssertionError("Optimized translation does not exactly match the original")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / f"{output_path.stem}_blender_tmp"
    frames_dir = work_dir / "frames"
    mesh_path = work_dir / "meshes.npz"
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    verts_original, faces = run_forward_kinematics(original, device)
    verts_optimized, _ = run_forward_kinematics(optimized, device)
    np.savez_compressed(
        mesh_path,
        verts_a=verts_original,
        verts_b=verts_optimized,
        faces=faces,
    )

    blender_script = Path(__file__).resolve().parent / "blender_render.py"
    subprocess.run(
        [
            str(args.blender_path),
            "-b",
            "-P",
            str(blender_script),
            "--",
            "--mesh-path",
            str(mesh_path),
            "--output-dir",
            str(frames_dir),
            "--engine",
            args.engine,
            "--samples",
            str(args.samples),
            "--fps",
            str(args.fps),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ],
        check=True,
    )
    shutil.rmtree(work_dir)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
