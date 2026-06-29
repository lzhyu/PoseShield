"""Frozen Stage 2 motion-fidelity metrics for public canonical motions."""

from __future__ import annotations

from typing import Optional

import numpy as np

from poseshield.hymotion.utils.motion_format import validate_motion_array


def compute_motion_metrics(
    optimized_motion: np.ndarray,
    reference_motion: np.ndarray,
    ground_truth_motion: Optional[np.ndarray],
) -> dict[str, Optional[float]]:
    """Return motion retention and decomposed ground-truth distances."""
    validate_motion_array(optimized_motion, name="optimized_motion")
    validate_motion_array(reference_motion, name="reference_motion")
    if reference_motion.shape[0] < 2:
        raise ValueError("Motion metrics require at least two frames")
    if optimized_motion.shape != reference_motion.shape:
        raise ValueError(
            "Optimized/reference shape mismatch: "
            f"{optimized_motion.shape} != {reference_motion.shape}"
        )
    if ground_truth_motion is not None:
        validate_motion_array(ground_truth_motion, name="ground_truth_motion")
        if ground_truth_motion.shape != reference_motion.shape:
            raise ValueError(
                "Ground-truth/reference shape mismatch: "
                f"{ground_truth_motion.shape} != {reference_motion.shape}"
            )

    eps = np.finfo(np.float32).eps
    optimized_rot_delta = np.abs(np.diff(optimized_motion[:, :132], axis=0)).mean()
    reference_rot_delta = np.abs(np.diff(reference_motion[:, :132], axis=0)).mean()
    dyn_ratio = optimized_rot_delta / max(reference_rot_delta, eps)
    optimized_hand_var = np.var(optimized_motion[:, 15 * 6 : 22 * 6], axis=0).mean()
    reference_hand_var = np.var(reference_motion[:, 15 * 6 : 22 * 6], axis=0).mean()
    hand_var_ratio = optimized_hand_var / max(reference_hand_var, eps)
    metrics: dict[str, Optional[float]] = {
        "dyn_ratio": float(dyn_ratio),
        "hand_var_ratio": float(hand_var_ratio),
        "gt_pose_dist": None,
        "gt_rel_trans_dist": None,
        "gt_abs_trans_dist": None,
    }
    if ground_truth_motion is None:
        return metrics

    gt_pose_dist = np.mean(
        (optimized_motion[:, :132] - ground_truth_motion[:, :132]) ** 2
    )
    optimized_translation = optimized_motion[:, 132:135]
    ground_truth_translation = ground_truth_motion[:, 132:135]
    gt_rel_trans_dist = np.mean(
        (
            np.diff(optimized_translation, axis=0)
            - np.diff(ground_truth_translation, axis=0)
        )
        ** 2
    )
    optimized_from_first = optimized_translation - optimized_translation[:1]
    ground_truth_from_first = ground_truth_translation - ground_truth_translation[:1]
    gt_abs_trans_dist = np.mean(
        (optimized_from_first - ground_truth_from_first) ** 2
    )
    metrics.update(
        {
            "gt_pose_dist": float(gt_pose_dist),
            "gt_rel_trans_dist": float(gt_rel_trans_dist),
            "gt_abs_trans_dist": float(gt_abs_trans_dist),
        }
    )
    return metrics
