"""Utilities for the public PoseShield motion format."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch


PUBLIC_MOTION_DIM = 135
ROTATION_DIM = 132
TRANSLATION_SLICE = slice(132, 135)
TranslationLayout = Literal["auto", "xyz", "xzy"]
RotationJointLayout = Literal["auto", "root_first", "root_last"]
INTERNAL_TRANSLATION_LAYOUT: Literal["xyz"] = "xyz"
INTERNAL_ROTATION_JOINT_LAYOUT: Literal["root_first"] = "root_first"
_TRANSLATION_STABILITY_EPS = 0.05


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


def infer_translation_layout(
    motion: np.ndarray | torch.Tensor,
) -> Literal["xyz", "xzy"]:
    """Infer whether absolute translation looks like ``xyz`` or legacy ``xzy``.

    Public release motion stores ``[x, y_up, z_forward]`` and all release-facing
    loaders use that layout by default. This heuristic exists only for explicit
    ``auto`` compatibility with legacy/source-layout artifacts. The up/height
    channel is usually high and temporally stable, so auto mode scores candidate
    channels by height-like magnitude divided by temporal range.
    """
    validate_motion_array(motion)
    translation = motion[..., TRANSLATION_SLICE]
    if torch.is_tensor(translation):
        median_abs = torch.median(torch.abs(translation), dim=0).values
        channel_range = torch.max(translation, dim=0).values - torch.min(
            translation, dim=0
        ).values
        scores = median_abs / (channel_range + _TRANSLATION_STABILITY_EPS)
        return "xzy" if float(scores[2]) > float(scores[1]) else "xyz"
    median_abs = np.median(np.abs(translation), axis=0)
    channel_range = np.ptp(translation, axis=0)
    scores = median_abs / (channel_range + _TRANSLATION_STABILITY_EPS)
    return "xzy" if float(scores[2]) > float(scores[1]) else "xyz"


def _resolve_translation_layout(layout: TranslationLayout) -> Literal["auto", "xyz", "xzy"]:
    """Validate and return a translation layout name."""
    if layout in ("auto", "xyz", "xzy"):
        return layout
    raise ValueError(f"Unknown translation layout: {layout}")


def _resolve_rotation_joint_layout(
    layout: RotationJointLayout,
) -> Literal["auto", "root_first", "root_last"]:
    """Validate and return a rotation joint layout name."""
    if layout in ("auto", "root_first", "root_last"):
        return layout
    raise ValueError(f"Unknown rotation joint layout: {layout}")


def infer_rotation_joint_layout(
    motion: np.ndarray | torch.Tensor,
    *,
    translation_layout: TranslationLayout = "auto",
) -> Literal["root_first", "root_last"]:
    """Infer whether rotations look root-first or legacy root-last.

    Public release motions use ``root_first``: ``[root, body0, ..., body20]``.
    This heuristic exists only for explicit ``auto`` compatibility with legacy
    artifacts where the source convention is paired with xzy translation, so
    auto mode uses the resolved translation layout as the discriminator.
    """
    validate_motion_array(motion)
    if translation_layout == "auto":
        resolved_translation = infer_translation_layout(motion)
    else:
        resolved_translation = _resolve_translation_layout(translation_layout)
        if resolved_translation == "auto":
            raise ValueError("resolved translation layout cannot be auto")
    return "root_last" if resolved_translation == "xzy" else "root_first"


def normalize_rotation_joint_layout(
    motion: np.ndarray | torch.Tensor,
    *,
    source: RotationJointLayout = INTERNAL_ROTATION_JOINT_LAYOUT,
    target: RotationJointLayout = INTERNAL_ROTATION_JOINT_LAYOUT,
    translation_layout: TranslationLayout = INTERNAL_TRANSLATION_LAYOUT,
) -> np.ndarray | torch.Tensor:
    """Return motion whose 22 rotation blocks use the target joint order."""
    validate_motion_array(motion)
    resolved_target = _resolve_rotation_joint_layout(target)
    if resolved_target == "auto":
        raise ValueError("target rotation joint layout cannot be auto")
    if source == "auto":
        resolved_source = infer_rotation_joint_layout(
            motion,
            translation_layout=translation_layout,
        )
    else:
        resolved_source = _resolve_rotation_joint_layout(source)
    if resolved_source == resolved_target:
        return motion

    if resolved_source == "root_last" and resolved_target == "root_first":
        order = [21, *range(21)]
    elif resolved_source == "root_first" and resolved_target == "root_last":
        order = [*range(1, 22), 0]
    else:
        raise ValueError(
            f"Unsupported rotation joint layout conversion: {resolved_source} -> {resolved_target}"
        )

    if torch.is_tensor(motion):
        normalized = motion.clone()
        rotations = normalized[..., :ROTATION_DIM].reshape(*normalized.shape[:-1], 22, 6)
        normalized[..., :ROTATION_DIM] = rotations[..., order, :].reshape(
            *normalized.shape[:-1],
            ROTATION_DIM,
        )
        return normalized
    normalized = motion.copy()
    rotations = normalized[..., :ROTATION_DIM].reshape(*normalized.shape[:-1], 22, 6)
    normalized[..., :ROTATION_DIM] = rotations[..., order, :].reshape(
        *normalized.shape[:-1],
        ROTATION_DIM,
    )
    return normalized


def normalize_translation_layout(
    motion: np.ndarray | torch.Tensor,
    *,
    source: TranslationLayout = INTERNAL_TRANSLATION_LAYOUT,
    target: TranslationLayout = INTERNAL_TRANSLATION_LAYOUT,
) -> np.ndarray | torch.Tensor:
    """Return motion whose translation channels use the target layout.

    ``xyz`` is the public layout ``[x, y_up, z_forward]``. ``xzy`` is supported
    only for explicit legacy/source-layout conversion.
    """
    validate_motion_array(motion)
    resolved_target = _resolve_translation_layout(target)
    if resolved_target == "auto":
        raise ValueError("target translation layout cannot be auto")
    if source == "auto":
        resolved_source = infer_translation_layout(motion)
    else:
        resolved_source = _resolve_translation_layout(source)
    if resolved_source == resolved_target:
        return motion

    if torch.is_tensor(motion):
        normalized = motion.clone()
        swapped = normalized[..., [134, 133]].clone()
        normalized[..., [133, 134]] = swapped
        return normalized
    normalized = motion.copy()
    normalized[..., [133, 134]] = normalized[..., [134, 133]]
    return normalized


def load_motion(
    path: str | Path,
    *,
    translation_layout: TranslationLayout = INTERNAL_TRANSLATION_LAYOUT,
    target_translation_layout: TranslationLayout = INTERNAL_TRANSLATION_LAYOUT,
    rotation_joint_layout: RotationJointLayout = INTERNAL_ROTATION_JOINT_LAYOUT,
    target_rotation_joint_layout: RotationJointLayout = INTERNAL_ROTATION_JOINT_LAYOUT,
) -> np.ndarray:
    """Load a public-format motion array and normalize it to internal layout."""
    motion = np.load(Path(path))
    validate_motion_array(motion, name=str(path))
    motion = motion.astype(np.float32, copy=False)
    if translation_layout == "auto":
        resolved_translation_layout = infer_translation_layout(motion)
    else:
        resolved_translation_layout = _resolve_translation_layout(translation_layout)
        if resolved_translation_layout == "auto":
            raise ValueError("resolved translation layout cannot be auto")
    normalized = normalize_rotation_joint_layout(
        motion,
        source=rotation_joint_layout,
        target=target_rotation_joint_layout,
        translation_layout=resolved_translation_layout,
    )
    normalized = normalize_translation_layout(
        normalized,
        source=resolved_translation_layout,
        target=target_translation_layout,
    )
    return normalized.astype(np.float32, copy=False)


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
