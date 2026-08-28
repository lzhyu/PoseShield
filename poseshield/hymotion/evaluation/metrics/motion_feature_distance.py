"""Motion Feature Distance (MFD): semantic drift from the input motion.

MFD measures how far collision resolution moved the motion in a learned motion
feature space: both the resolved and the original motion are embedded with a
pretrained motion encoder, and MFD is the L2 distance between the two
embeddings. It is the counterpart to RPD -- a method can always drive
penetration to zero by producing a completely different motion, and MFD is what
makes that visible.

Because the encoder produces L2-normalized embeddings, the plain Euclidean
distance taken here is already the normalized feature distance, bounded in
`[0, 2]`.

The encoder is injected as `encode_fn` rather than imported, so this module has
no dependency on the encoder's weights or on the vendored feature-conversion
code. That keeps the other three metrics runnable without the optional
third-party download that MFD needs; see
`poseshield/hymotion/evaluation/motion_features/motionclip_encoder.py`.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class MotionEncoderUnavailable(RuntimeError):
    """Raised when MFD is requested without a usable motion encoder."""


def _missing_encoder(_motion_135: np.ndarray) -> np.ndarray:
    """Placeholder `encode_fn` that always raises with setup instructions."""
    raise MotionEncoderUnavailable(
        "No motion feature encoder configured. MFD needs the pretrained motion "
        "encoder described in the Evaluation section of the README (a "
        "third-party download that is not distributed with PoseShield). Pass "
        "--no_mfd to report Jitter, FSR, and RPD without it."
    )


def compute_mfd(
    optimized_motion_135: np.ndarray,
    original_motion_135: np.ndarray,
    encode_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> float:
    """Compute the feature-space distance between a resolved motion and its input.

    Args:
        optimized_motion_135: Collision-resolved motion, shape `[T, 135]`.
        original_motion_135: The motion it was resolved from, shape `[T, 135]`.
        encode_fn: Maps a `[T, 135]` motion to a 1D feature vector. If None,
            raises `MotionEncoderUnavailable`.

    Returns:
        L2 distance between the two embeddings. 0.0 when the two motions are
        identical.

    Raises:
        MotionEncoderUnavailable: If `encode_fn` is None.
        ValueError: If the two motions have different frame counts. The encoder
            pools over time, so a mismatch would otherwise silently produce a
            plausible-looking distance between motions of different durations.
    """
    optimized_motion_135 = np.asarray(optimized_motion_135)
    original_motion_135 = np.asarray(original_motion_135)
    if optimized_motion_135.shape[0] != original_motion_135.shape[0]:
        raise ValueError(
            f"compute_mfd: frame count mismatch, optimized has "
            f"{optimized_motion_135.shape[0]} frames but original has "
            f"{original_motion_135.shape[0]}."
        )
    encode_fn = encode_fn or _missing_encoder
    feat_opt = np.asarray(encode_fn(optimized_motion_135), dtype=np.float64).reshape(-1)
    feat_orig = np.asarray(encode_fn(original_motion_135), dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(feat_opt - feat_orig))
