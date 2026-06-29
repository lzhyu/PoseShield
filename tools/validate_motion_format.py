"""Validate public canonical PoseShield motion files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.hymotion.dno.dno_loss import rotation_6d_to_matrix_torch
from poseshield.hymotion.utils.motion_format import load_motion


def parse_args() -> argparse.Namespace:
    """Parse validation arguments."""
    parser = argparse.ArgumentParser(description="Validate canonical PoseShield motion files")
    parser.add_argument("motions", type=Path, nargs="+", help="Motion .npy files")
    parser.add_argument("--facing-tol-deg", type=float, default=1.0)
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
            motion = load_motion(path)
            angle = facing_angle_deg(motion)
            if abs(angle) > args.facing_tol_deg:
                raise ValueError(f"frame-0 facing angle is {angle:.3f} degrees")
            trans = motion[:, 132:135]
            if not np.isfinite(trans).all():
                raise ValueError("translation contains non-finite values")
            print(f"OK {path}: shape={motion.shape}, facing={angle:.3f} deg")
        except Exception as error:
            failures.append(f"{path}: {error}")
            print(f"FAIL {path}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
