"""Jitter: temporal smoothness of a motion.

Jitter is the mean magnitude of joint jerk (the third time-derivative of joint
position), reported in units of 10^2 m/s^3. Collision resolution that fixes
interpenetration by adding high-frequency, per-frame corrections shows up here
as a large jitter increase, so this metric is what keeps a low residual
penetration from being bought with visibly unnatural motion.

Reference: Yi et al., "Physical Inertial Poser (PIP): Physics-aware Real-time
Human Motion Tracking from Sparse Inertial Sensors", CVPR 2022. The 10^2 m/s^3
reporting convention comes from the same authors' TransPose (SIGGRAPH 2021).
"""

from __future__ import annotations

import numpy as np

DEFAULT_FPS = 30.0


def compute_jitter(joint_positions: np.ndarray, fps: float = DEFAULT_FPS) -> float:
    """Compute the mean jerk magnitude of a joint trajectory.

    Jerk is estimated with the standard 4-tap finite-difference stencil
    `p[t+3] - 3 p[t+2] + 3 p[t+1] - p[t]`, divided by `dt^3`.

    Args:
        joint_positions: World-space joint positions in meters, shape
            `[T, J, 3]`. Pass the 22 SMPL body joints; including the SMPL-H
            hand joints roughly doubles the value, since finger joints carry
            their own high-frequency motion.
        fps: Frame rate of the motion, in frames per second.

    Returns:
        Mean jerk magnitude over all joints and frames, in units of
        10^2 m/s^3. NaN if the motion has fewer than 4 frames, which is the
        minimum the stencil needs.
    """
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    t = joint_positions.shape[0]
    if t < 4:
        return float("nan")
    dt = 1.0 / fps
    jerk = (
        joint_positions[3:]
        - 3 * joint_positions[2:-1]
        + 3 * joint_positions[1:-2]
        - joint_positions[:-3]
    ) / (dt**3)
    jerk_norm = np.linalg.norm(jerk, axis=-1)  # [T-3, J]
    return float(jerk_norm.mean() / 100.0)
