"""Forward kinematics and exact-collision helpers for public 135-D motions.

Thin wrappers that take a public canonical motion (`[T, 135]`, as produced by
Stage 2 and documented in `data/motion_canonical/README.md`) and return SMPL-H
joints, SMPL-H vertices, or per-frame self-penetration depth. Everything here
delegates to the existing PoseShield implementations rather than re-deriving
them, so metrics computed on top of this module use exactly the same forward
kinematics and the same exact-FCL collision check as the rest of the repo.

6D rotation convention
----------------------
The public 135-D format stores each joint's rotation as a **column-interleaved**
6D vector `[c1x, c2x, c1y, c2y, c1z, c2z]` (the HY-Motion layout). This module
therefore uses the pair in `poseshield.hymotion.dno.dno_loss`, which parses that
layout.

Note that `poseshield/common/utils.py` exposes functions with the *same names*
(`rotation_6d_to_matrix_torch`, `matrix_to_axis_angle_torch`) implementing the
**row-major** convention used by the pose-level model. The two are not
interchangeable: feeding a column-interleaved vector to the row-major parser
silently yields a different rotation. Import the `dno_loss` pair for anything
touching 135-D motions, and the `common.utils` pair for the pose-level path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from poseshield.common.collision import (
    DEFAULT_MOTION_TOPOLOGY_THRESHOLD,
    load_topology_distances,
    self_collision_status,
)
from poseshield.hymotion.dno.dno_loss import (
    BODY_MODEL_PATH_,
    matrix_to_axis_angle_torch,
    parameters_to_joints,
    rotation_6d_to_matrix_torch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISTANCES_PATH = PROJECT_ROOT / "deps" / "topology_distances_30_60.npz"

_SMPL_MODEL_CACHE: dict[tuple[int, str], object] = {}
_DISTANCES_CACHE = None


def get_smpl_model(batch_size: int, device):
    """Return a cached SMPL-H neutral body model sized for `batch_size` frames.

    `smplx.create` bakes the batch size into the model, so a separate instance
    is needed per frame count; building one is expensive enough that scoring a
    directory of motions is dominated by it without this cache.
    """
    import smplx

    key = (batch_size, str(device))
    model = _SMPL_MODEL_CACHE.get(key)
    if model is None:
        model = smplx.create(
            BODY_MODEL_PATH_,
            model_type="smplh",
            gender="neutral",
            ext="npz",
            use_pca=False,
            batch_size=batch_size,
        ).to(device)
        _SMPL_MODEL_CACHE[key] = model
    return model


def motion_to_joints_torch(motion_135: torch.Tensor, device) -> torch.Tensor:
    """`[T, 135]` motion (torch, may require grad) -> `[T, 73, 3]` SMPL-H joints."""
    smpl_model = get_smpl_model(motion_135.shape[0], device)
    return parameters_to_joints(motion_135, smpl_model, device)


def motion_to_joints(motion_135: np.ndarray, device) -> np.ndarray:
    """`[T, 135]` motion (numpy) -> `[T, 73, 3]` SMPL-H joint positions (numpy).

    The first 22 joints are the SMPL body joints; metrics that operate on body
    articulation slice `[:, :22]`.
    """
    with torch.no_grad():
        motion_t = torch.from_numpy(np.asarray(motion_135)).float().to(device)
        joints = motion_to_joints_torch(motion_t, device)
    return joints.detach().cpu().numpy()


def motion_to_vertices(motion_135: np.ndarray, device):
    """`[T, 135]` motion -> (`[T, V, 3]` SMPL-H vertices, faces).

    Returns the posed surface mesh, which is what the exact-FCL collision check
    consumes (joints are not enough to detect surface interpenetration).
    """
    motion_135 = np.asarray(motion_135)
    rotations = torch.from_numpy(motion_135[:, :132]).float().to(device)
    root_rotation = rotations[:, :6]
    body_rotation = rotations[:, 6:].reshape(-1, 21, 6)
    global_axis_angles = matrix_to_axis_angle_torch(rotation_6d_to_matrix_torch(root_rotation))
    body_axis_angles = matrix_to_axis_angle_torch(rotation_6d_to_matrix_torch(body_rotation)).flatten(1)
    translation = torch.from_numpy(motion_135[:, 132:135]).float().to(device)

    smpl_model = get_smpl_model(motion_135.shape[0], device)
    with torch.no_grad():
        smpl_output = smpl_model(
            global_orient=global_axis_angles,
            body_pose=body_axis_angles,
            transl=translation,
            return_verts=True,
        )
    return smpl_output.vertices.cpu().numpy(), smpl_model.faces


def get_topology_distances(distances_path=DEFAULT_DISTANCES_PATH):
    """Return the cached face-pair topology distance table used to ignore
    self-contacts between mesh regions that are adjacent on the body surface."""
    global _DISTANCES_CACHE
    if _DISTANCES_CACHE is None:
        _DISTANCES_CACHE = load_topology_distances(Path(distances_path))
    return _DISTANCES_CACHE


def per_frame_penetration_depths(
    motion_135: np.ndarray,
    device,
    distances_path=DEFAULT_DISTANCES_PATH,
    topology_threshold: int = DEFAULT_MOTION_TOPOLOGY_THRESHOLD,
) -> np.ndarray:
    """Exact per-frame self-penetration depth for a motion, in meters.

    Runs the same python-fcl check as `tools/evaluate_exact_fcl.py`: pose the
    SMPL-H mesh at every frame and measure the total penetration depth of
    topologically distant self-intersections.

    Returns:
        `[T]` array of per-frame penetration depths.
    """
    vertices, faces = motion_to_vertices(motion_135, device)
    distances = get_topology_distances(distances_path)
    depths = np.empty(vertices.shape[0], dtype=np.float64)
    for i, frame_vertices in enumerate(vertices):
        _, depth = self_collision_status(
            frame_vertices, faces, distances, topology_threshold=topology_threshold
        )
        depths[i] = depth
    return depths
