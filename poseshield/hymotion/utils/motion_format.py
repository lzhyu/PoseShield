"""Utilities for the public PoseShield motion format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


PUBLIC_MOTION_DIM = 135
ROTATION_DIM = 132
TRANSLATION_SLICE = slice(132, 135)


def validate_motion_array(motion: np.ndarray | torch.Tensor, *, name: str = "motion") -> None:
    """Validate a canonical HY-Motion-compatible motion array."""
    shape = tuple(motion.shape)
    if len(shape) != 2 or shape[1] != PUBLIC_MOTION_DIM:
        raise ValueError(f"Expected {name} shape [frames, 135], got {shape}")
    if isinstance(motion, np.ndarray):
        if not np.isfinite(motion).all():
            raise ValueError(f"{name} contains non-finite values")
    elif torch.is_tensor(motion):
        if not torch.isfinite(motion).all():
            raise ValueError(f"{name} contains non-finite values")
    else:
        raise TypeError(f"Unsupported {name} type: {type(motion)}")


def load_motion(path: str | Path) -> np.ndarray:
    """Load a canonical motion array from disk."""
    motion = np.load(Path(path))
    validate_motion_array(motion, name=str(path))
    return motion.astype(np.float32, copy=False)


def get_absolute_translation(motion: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return the absolute global XYZ translation block."""
    validate_motion_array(motion)
    return motion[..., TRANSLATION_SLICE]


def latent_to_public_motion(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    reference_translation: torch.Tensor,
) -> torch.Tensor:
    """Convert HY-Motion latent output to public 135-D format."""
    if x.shape[0] != 1 or x.shape[-1] != 201:
        raise ValueError(f"Expected optimized latent motion [1, frames, 201], got {tuple(x.shape)}")
    if reference_translation.shape != (x.shape[1], 3):
        raise ValueError(
            "Reference translation does not match optimized motion: "
            f"{tuple(reference_translation.shape)} != {(x.shape[1], 3)}"
        )
    x_unnorm = x * std + mean
    root_rotation = x_unnorm[0, :, 3:9]
    body_rotation = x_unnorm[0, :, 9:135]
    motion_135 = torch.cat(
        [
            root_rotation,
            body_rotation,
            reference_translation.to(device=x.device, dtype=x.dtype),
        ],
        dim=-1,
    )
    if motion_135.shape[-1] != PUBLIC_MOTION_DIM:
        raise AssertionError(f"Unexpected public motion shape: {tuple(motion_135.shape)}")
    return motion_135
