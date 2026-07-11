"""Render a side-by-side original/optimized motion MP4 with Blender.

Optional yellow contact patches require precomputed exact-FCL masks generated
for the same original motion, SMPL-H topology, and topology threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse Blender rendering arguments."""
    parser = argparse.ArgumentParser(description="Render a high-quality SMPL-H motion MP4")
    parser.add_argument("--original", type=Path, default=None, help="Original canonical motion")
    parser.add_argument("--optimized", type=Path, default=None, help="Optimized canonical motion")
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=None,
        help=(
            "Precomputed render mesh package with verts_a, verts_b, faces, and optional contact_masks. "
            "When provided, SMPL-H/body-model dependencies are not needed."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--blender-path", type=Path, required=True, help="Path to the Blender binary")
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to an FFmpeg binary. The preferred encoder is libx264; the script falls back to mpeg4 if unavailable.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--engine",
        choices=("CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"),
        default="BLENDER_EEVEE",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Render every Nth frame for compact website previews")
    parser.add_argument("--start-frame", type=int, default=0, help="Start rendering from this original frame index before striding")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap the number of rendered frames after striding")
    parser.add_argument("--res-x", type=int, default=1920)
    parser.add_argument("--res-y", type=int, default=1080)
    parser.add_argument("--highlight-contact", action="store_true", help="Render yellow contact patches on the red input motion")
    parser.add_argument(
        "--contact-mask-path",
        type=Path,
        default=None,
        help=(
            "Precomputed full-sequence contact mask .npz from tools/export_motion_contact_masks.py. "
            "This is the recommended prerequisite for --highlight-contact."
        ),
    )
    parser.add_argument("--disable-contact-highlight", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_forward_kinematics(motion: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Compute SMPL-H vertices and faces for a public canonical motion."""
    import smplx
    import torch

    from poseshield.hymotion.dno.dno_loss import (
        BODY_MODEL_PATH_,
        matrix_to_axis_angle_torch,
        rotation_6d_to_matrix_torch,
    )

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


def resolve_torch_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def encode_frames_with_ffmpeg(
    ffmpeg_path: str,
    frames_dir: Path,
    output_path: Path,
    fps: int,
) -> None:
    """Encode rendered PNG frames to MP4, falling back when libx264 is unavailable."""
    frame_pattern = str(frames_dir / "frame_%04d.png")
    x264_command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    fallback_command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-c:v",
        "mpeg4",
        "-q:v",
        "3",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    ffmpeg_executable = shutil.which(ffmpeg_path) or (ffmpeg_path if Path(ffmpeg_path).is_file() else None)
    if ffmpeg_executable is None:
        raise FileNotFoundError(
            f"FFmpeg executable not found: {ffmpeg_path}. Install FFmpeg or pass --ffmpeg-path /path/to/ffmpeg."
        )
    x264_command[0] = ffmpeg_executable
    fallback_command[0] = ffmpeg_executable

    try:
        subprocess.run(x264_command, check=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(
            "FFmpeg libx264 encode failed; retrying with MPEG-4 fallback. "
            f"Original error: {exc}",
            file=sys.stderr,
        )
    try:
        subprocess.run(fallback_command, check=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"FFmpeg MPEG-4 fallback failed: {exc}") from exc


def encode_frames_as_gif(frames_dir: Path, output_path: Path, fps: int) -> None:
    """Encode rendered PNG frames as an animated GIF without FFmpeg."""
    from PIL import Image

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    frames = [
        Image.open(path)
        .convert("RGB")
        .quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for path in frame_paths
    ]
    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main() -> None:
    """Create a side-by-side MP4 with Blender and ffmpeg."""
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_blender_tmp_", dir=output_path.parent))
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.mesh_path is not None:
        if args.contact_mask_path is not None:
            raise ValueError("--contact-mask-path is only used with --original/--optimized; precomputed mesh packages include masks")
        source_mesh_path = args.mesh_path.resolve()
        mesh_data = np.load(source_mesh_path)
        required_keys = {"verts_a", "verts_b", "faces"}
        missing = required_keys.difference(mesh_data.files)
        if missing:
            raise ValueError(f"Mesh package {source_mesh_path} is missing required arrays: {sorted(missing)}")
        verts_original = mesh_data["verts_a"]
        verts_optimized = mesh_data["verts_b"]
        faces = mesh_data["faces"]
        if verts_original.shape != verts_optimized.shape:
            raise ValueError(f"Mesh package body shapes differ: {verts_original.shape} != {verts_optimized.shape}")
        if args.start_frame < 0:
            raise ValueError("--start-frame must be >= 0")
        if args.start_frame >= verts_original.shape[0]:
            raise ValueError(f"--start-frame {args.start_frame} is outside the mesh sequence length {verts_original.shape[0]}")
        if "contact_masks" in mesh_data.files:
            contact_masks = mesh_data["contact_masks"].astype(np.bool_)
            if contact_masks.shape[0] != verts_original.shape[0] or contact_masks.shape[1] != len(faces):
                raise ValueError(
                    f"Mesh package contact mask shape {contact_masks.shape} does not match "
                    f"{verts_original.shape[0]} frames and {len(faces)} faces"
                )
        else:
            contact_masks = np.zeros((verts_original.shape[0], len(faces)), dtype=np.bool_)
        verts_original = verts_original[args.start_frame :: args.frame_stride]
        verts_optimized = verts_optimized[args.start_frame :: args.frame_stride]
        contact_masks = contact_masks[args.start_frame :: args.frame_stride]
        if args.max_frames is not None:
            verts_original = verts_original[: args.max_frames]
            verts_optimized = verts_optimized[: args.max_frames]
            contact_masks = contact_masks[: args.max_frames]
        if verts_original.shape[0] == 0:
            raise ValueError("No frames remain after start/stride/max-frame filtering")
        mesh_path = work_dir / "meshes.npz"
        np.savez_compressed(
            mesh_path,
            verts_a=verts_original,
            verts_b=verts_optimized,
            faces=faces,
            contact_masks=contact_masks,
        )
    else:
        if args.original is None or args.optimized is None:
            raise ValueError("Provide either --mesh-path or both --original and --optimized")
        from poseshield.hymotion.utils.motion_format import load_motion

        original = load_motion(args.original)
        optimized = load_motion(args.optimized)
        if original.shape != optimized.shape:
            raise ValueError(f"Motion shapes differ: {original.shape} != {optimized.shape}")
        if not np.array_equal(original[:, 132:135], optimized[:, 132:135]):
            raise AssertionError("Optimized translation does not exactly match the original")
        if args.start_frame < 0:
            raise ValueError("--start-frame must be >= 0")
        if args.start_frame >= original.shape[0]:
            raise ValueError(f"--start-frame {args.start_frame} is outside the motion length {original.shape[0]}")
        highlight_contact = (args.highlight_contact or args.contact_mask_path is not None) and not args.disable_contact_highlight
        if args.highlight_contact and args.contact_mask_path is None:
            raise ValueError(
                "--highlight-contact requires --contact-mask-path. "
                "Generate masks first with tools/export_motion_contact_masks.py."
            )
        contact_masks_full = None
        contact_faces_ref = None
        if highlight_contact and args.contact_mask_path is not None:
            mask_data = np.load(args.contact_mask_path)
            contact_masks_full = mask_data["contact_masks"].astype(np.bool_)
            if "faces" in mask_data.files:
                contact_faces_ref = mask_data["faces"]
            if contact_masks_full.shape[0] != original.shape[0]:
                raise ValueError(
                    f"Contact mask frame count {contact_masks_full.shape[0]} does not match motion frame count {original.shape[0]}"
                )
        if args.start_frame:
            original = original[args.start_frame :]
            optimized = optimized[args.start_frame :]
            if contact_masks_full is not None:
                contact_masks_full = contact_masks_full[args.start_frame :]
        original = original[:: args.frame_stride]
        optimized = optimized[:: args.frame_stride]
        if contact_masks_full is not None:
            contact_masks_full = contact_masks_full[:: args.frame_stride]
        if args.max_frames is not None:
            original = original[: args.max_frames]
            optimized = optimized[: args.max_frames]
            if contact_masks_full is not None:
                contact_masks_full = contact_masks_full[: args.max_frames]

        device = resolve_torch_device(args.device)
        verts_original, faces = run_forward_kinematics(original, device)
        verts_optimized, _ = run_forward_kinematics(optimized, device)
        if contact_masks_full is not None:
            contact_masks = contact_masks_full
            if contact_masks.shape[1] != len(faces):
                raise ValueError(f"Contact mask face count {contact_masks.shape[1]} does not match mesh faces {len(faces)}")
            if contact_faces_ref is not None and not np.array_equal(contact_faces_ref, faces):
                raise ValueError("Contact mask faces do not match the SMPL-H mesh topology used for rendering")
        else:
            contact_masks = np.zeros((len(verts_original), len(faces)), dtype=np.bool_)
        mesh_path = work_dir / "meshes.npz"
        np.savez_compressed(
            mesh_path,
            verts_a=verts_original,
            verts_b=verts_optimized,
            faces=faces,
            contact_masks=contact_masks,
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
            "--res-x",
            str(args.res_x),
            "--res-y",
            str(args.res_y),
            "--color-a",
            "0.85",
            "0.18",
            "0.18",
            "1.0",
            "--color-b",
            "0.25",
            "0.75",
            "0.32",
            "1.0",
        ],
        check=True,
    )
    rendered_frames = sorted(frames_dir.glob("frame_*.png"))
    if not rendered_frames:
        raise RuntimeError(f"Blender did not render any PNG frames under {frames_dir}")
    if output_path.suffix.lower() == ".gif":
        encode_frames_as_gif(frames_dir, output_path, args.fps)
    else:
        encode_frames_with_ffmpeg(args.ffmpeg_path, frames_dir, output_path, args.fps)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
