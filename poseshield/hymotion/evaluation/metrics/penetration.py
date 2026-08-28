"""Residual Penetration Depth (RPD): how much self-collision is left.

RPD is the frame-averaged exact self-penetration depth of the output motion, in
meters. It is the metric the collision-resolution task is actually trying to
minimize; the other three metrics exist to check that it is not minimized at the
cost of motion quality.

Unlike Jitter, FSR, and MFD, RPD has no external reference -- it is defined
directly in the PoseShield paper. It is measured with the exact mesh-level
python-fcl check in `poseshield/common/collision.py`, the same code path as
`tools/evaluate_exact_fcl.py`, rather than with the learned collision field, so
the number is independent of the model being evaluated.
"""

from __future__ import annotations

import numpy as np

from poseshield.common.motion_fk import DEFAULT_DISTANCES_PATH, per_frame_penetration_depths


def compute_rpd(motion_135: np.ndarray, device, distances_path=DEFAULT_DISTANCES_PATH) -> float:
    """Compute the frame-averaged self-penetration depth of a motion.

    Args:
        motion_135: Canonical motion, shape `[T, 135]`.
        device: torch device used for forward kinematics.
        distances_path: Face-pair topology distance table, used to ignore
            contacts between mesh regions that are adjacent on the body surface.

    Returns:
        Mean per-frame penetration depth, in meters. 0.0 for a fully
        collision-free motion.
    """
    depths = per_frame_penetration_depths(motion_135, device, distances_path=distances_path)
    return float(np.mean(depths))
