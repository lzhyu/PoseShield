"""Validate public canonical PoseShield motion files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.hymotion.dno.dno_loss import rotation_6d_to_matrix_torch
from poseshield.hymotion.utils.motion_format import (
    infer_rotation_joint_layout,
    infer_translation_layout,
    load_motion,
    validate_motion_array,
)


def parse_args() -> argparse.Namespace:
    """Parse validation arguments."""
    parser = argparse.ArgumentParser(description="Validate canonical PoseShield motion files")
    parser.add_argument("motions", type=Path, nargs="+", help="Motion .npy files")
    parser.add_argument("--facing-tol-deg", type=float, default=1.0)
    parser.add_argument(
        "--expected-translation-layout",
        choices=["xyz", "xzy"],
        default="xyz",
        help="Expected raw translation layout before any loader conversion.",
    )
    parser.add_argument(
        "--expected-rotation-joint-layout",
        choices=["root_first", "root_last"],
        default="root_first",
        help="Expected raw 22-joint rotation-block order before any loader conversion.",
    )
    return parser.parse_args()


def facing_angle_deg(motion: np.ndarray) -> float:
    """Return frame-0 facing angle relative to +Z in degrees."""
    import torch

    root_rotation = torch.from_numpy(motion[0, :6]).float()
    root_matrix = rotation_6d_to_matrix_torch(root_rotation).cpu().numpy()
    right_xz = np.array([root_matrix[0, 0], root_matrix[2, 0]], dtype=np.float64)
    right_xz /= np.linalg.norm(right_xz) + 1e-12
    forward_xz = np.array([-right_xz[1], right_xz[0]], dtype=np.float64)
    return float(np.degrees(np.arctan2(forward_xz[0], forward_xz[1])))


def main() -> None:
    """Validate every provided motion file."""
    args = parse_args()
    failures: list[str] = []
    for path in args.motions:
        try:
            raw_motion = np.load(path)
            validate_motion_array(raw_motion, name=str(path))
            translation_layout = infer_translation_layout(raw_motion)
            rotation_joint_layout = infer_rotation_joint_layout(
                raw_motion,
                translation_layout=translation_layout,
            )
            if translation_layout != args.expected_translation_layout:
                raise ValueError(
                    "raw translation layout is "
                    f"{translation_layout}, expected {args.expected_translation_layout}"
                )
            if rotation_joint_layout != args.expected_rotation_joint_layout:
                raise ValueError(
                    "raw rotation joint layout is "
                    f"{rotation_joint_layout}, expected {args.expected_rotation_joint_layout}"
                )
            motion = load_motion(path)
            angle = facing_angle_deg(motion)
            if abs(angle) > args.facing_tol_deg:
                raise ValueError(f"frame-0 facing angle is {angle:.3f} degrees")
            trans = motion[:, 132:135]
            if not np.isfinite(trans).all():
                raise ValueError("translation contains non-finite values")
            print(
                f"OK {path}: shape={motion.shape}, "
                f"layout={translation_layout}/{rotation_joint_layout}, "
                f"facing={angle:.3f} deg"
            )
        except Exception as error:
            failures.append(f"{path}: {error}")
            print(f"FAIL {path}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
