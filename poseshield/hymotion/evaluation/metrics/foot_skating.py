"""Foot Skating Ratio (FSR): how often a planted foot slides along the ground.

A frame transition counts as skating when a foot is in contact with the ground
(low enough in both adjacent frames) yet still translates horizontally faster
than a velocity threshold. FSR is the percentage of all frame transitions where
either foot skates. Collision resolution that displaces the whole body to
separate limbs tends to drag planted feet, which this catches.

Reference: Karunratanakul et al., "Guided Motion Diffusion for Controllable
Human Motion Synthesis" (GMD), ICCV 2023. This implementation follows GMD's
released `calculate_skating_ratio`, including two details that matter and are
easy to miss:

- Contact requires the foot to be below the height threshold in **both** frames
  of the transition, not just the later one.
- The instantaneous horizontal speed **and** its 5-frame moving average must
  both exceed the threshold, so a single-frame velocity spike does not count.

Two adaptations to this repo's data, neither a departure from GMD's definition:

- GMD's thresholds are stated at 20 fps against already floor-aligned input.
  The skating threshold is expressed here as a velocity in m/s and converted to
  a per-frame displacement using the motion's actual frame rate, so it means the
  same thing at any fps.
- Public canonical motions are not floor-aligned (each capture carries its own
  arbitrary height offset), so each motion is aligned to its own minimum joint
  height before the contact test. That reproduces GMD's precondition rather
  than changing it.
"""

from __future__ import annotations

import numpy as np

DEFAULT_FPS = 30.0
LEFT_FOOT_JOINT = 10
RIGHT_FOOT_JOINT = 11

GROUND_CONTACT_HEIGHT_M = 0.05  # foot height below which it counts as planted
SKATING_VELOCITY_M_PER_S = 0.5  # horizontal speed above which a planted foot is skating
MOVING_AVERAGE_WINDOW = 5  # frames, for the spike-suppressing velocity gate
UP_AXIS = 1  # canonical motion is Y-up


def compute_foot_skating_ratio(
    joint_positions: np.ndarray,
    fps: float = DEFAULT_FPS,
    foot_joint_indices=(LEFT_FOOT_JOINT, RIGHT_FOOT_JOINT),
    height_threshold: float = GROUND_CONTACT_HEIGHT_M,
    velocity_threshold: float = SKATING_VELOCITY_M_PER_S,
    up_axis: int = UP_AXIS,
    moving_average_window: int = MOVING_AVERAGE_WINDOW,
) -> float:
    """Compute the percentage of frame transitions exhibiting foot skating.

    Args:
        joint_positions: World-space joint positions in meters, shape
            `[T, J, 3]`. Must be world-space, not root-relative: a planted foot
            staying still while the root walks forward is correctly *not*
            skating, and root-relative coordinates would make it look like it is.
        fps: Frame rate of the motion, in frames per second.
        foot_joint_indices: Indices of the foot joints to test. Defaults to the
            two toe joints, matching GMD.
        height_threshold: Height in meters, after per-motion floor alignment,
            below which a foot counts as planted.
        velocity_threshold: Horizontal speed in m/s above which a planted foot
            counts as skating.
        up_axis: Index of the up axis in the last dimension.
        moving_average_window: Window in frames for the moving-average velocity
            gate.

    Returns:
        Percentage in `[0, 100]` of frame transitions where either foot skates.
        0.0 for motions shorter than 2 frames.
    """
    joint_positions = np.asarray(joint_positions, dtype=np.float64).copy()
    t = joint_positions.shape[0]
    if t < 2:
        return 0.0

    # Align this motion to its own floor so the absolute height test is meaningful.
    floor_height = joint_positions[:, :, up_axis].min()
    joint_positions[:, :, up_axis] -= floor_height

    horiz_axes = [a for a in range(3) if a != up_axis]
    displacement_threshold = velocity_threshold / fps

    skating = np.zeros(t - 1, dtype=bool)
    for joint_idx in foot_joint_indices:
        pos = joint_positions[:, joint_idx]  # [T, 3]
        horiz = pos[:, horiz_axes]
        disp = np.linalg.norm(horiz[1:] - horiz[:-1], axis=-1)  # [T-1]

        height = pos[:, up_axis]
        contact = (height[:-1] < height_threshold) & (height[1:] < height_threshold)

        disp_avg = _uniform_filter1d(disp, moving_average_window)
        skating |= contact & (disp > displacement_threshold) & (disp_avg > displacement_threshold)

    # Denominator is every frame transition, not only the in-contact ones --
    # this is a ratio over the whole motion, matching GMD.
    return float(skating.mean() * 100.0)


def _uniform_filter1d(values: np.ndarray, size: int) -> np.ndarray:
    """Apply a 1D moving average, matching `scipy.ndimage.uniform_filter1d`.

    Reimplemented rather than imported so this metric has no scipy dependency;
    equivalent to scipy's `mode="constant", origin=0`.

    Args:
        values: 1D array to smooth.
        size: Window size in samples.

    Returns:
        Smoothed array, same shape as `values`.
    """
    if size <= 1 or values.size == 0:
        return values
    left = size // 2
    right = size - left - 1
    padded = np.concatenate([np.zeros(left), values, np.zeros(right)])
    kernel = np.ones(size) / size
    return np.convolve(padded, kernel, mode="valid")
