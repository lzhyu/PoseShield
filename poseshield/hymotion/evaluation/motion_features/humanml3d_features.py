"""Convert SMPL-H joint positions into HumanML3D's 67-D motion feature.

The conversion itself is third-party: it runs the vendored HumanML3D
`process_file` (see `humanml3d/__init__.py`), and the joint indices and
thresholds configured below are the values HumanML3D's own
`motion_representation.ipynb` passes it. They are reproduced here rather than
chosen by us, because they define the feature the pretrained encoder expects.

This is the input representation the MFD motion encoder consumes. It is the
first 67 channels of HumanML3D's 263-D "redundant representation":

    [0:1]   root angular velocity about the Y (up) axis
    [1:3]   root linear velocity in the XZ (ground) plane
    [3:4]   root height
    [4:67]  the other 21 joints' positions, relative to the root and rotated
            into the root's facing frame (21 x 3)

The conversion is delegated to the vendored HumanML3D `process_file` (see
`humanml3d/__init__.py`) rather than re-derived, so the features match what the
pretrained encoder was trained on. A hand-written re-derivation is easy to get
subtly wrong: `process_file` smooths the body's facing direction with a
Gaussian (`sigma=20`) before deriving the root rotation, and omitting that step
alone shifts the root-angular-velocity channel by most of its own standard
deviation.

Skeleton retargeting note
-------------------------
Upstream HumanML3D rescales every clip's bone lengths to one fixed reference
skeleton so all motions share a scale. Here each motion is retargeted to *its
own* frame-0 skeleton instead, which makes `uniform_skeleton`'s scale factor
exactly 1.0 (an identity map). For this pipeline the two are equivalent: all
motions are posed with the same SMPL-H neutral body and no per-clip shape
parameters, so there is only ever one skeleton in play.
"""

from __future__ import annotations

import numpy as np
import torch

from .humanml3d.common.skeleton import Skeleton
from .humanml3d.utils import motion_process as _MP
from .humanml3d.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets

NUM_JOINTS = 22

# SMPL/SMPL-H joint order: 0 pelvis, 1 l_hip, 2 r_hip, ..., 16 l_shoulder,
# 17 r_shoulder. These pick the four joints whose cross-body vectors define the
# body's facing direction, matching HumanML3D's `face_joint_indx`.
FACE_JOINT_INDEX = [2, 1, 17, 16]  # r_hip, l_hip, r_shoulder, l_shoulder
FID_L = [7, 10]  # left ankle, left foot/toe
FID_R = [8, 11]  # right ankle, right foot/toe
L_IDX1, L_IDX2 = 5, 8  # lower-leg joints, used for the leg-length scale
FEET_THRE = 0.002  # ground-contact velocity threshold used by process_file

_N_RAW_OFFSETS = torch.from_numpy(t2m_raw_offsets)
_TGT_SKEL = Skeleton(_N_RAW_OFFSETS, t2m_kinematic_chain, "cpu")

# `process_file` and `uniform_skeleton` read their skeleton configuration from
# module-level globals. That mutable-global pattern comes from the upstream
# source; it is confined here so callers never have to know about it.
_MP.n_raw_offsets = _N_RAW_OFFSETS
_MP.kinematic_chain = t2m_kinematic_chain
_MP.face_joint_indx = FACE_JOINT_INDEX
_MP.fid_l = FID_L
_MP.fid_r = FID_R
_MP.l_idx1 = L_IDX1
_MP.l_idx2 = L_IDX2


def joints_to_67d(joints: np.ndarray) -> np.ndarray:
    """Convert world-space joint positions to the 67-D HumanML3D feature.

    Args:
        joints: World-space SMPL-H joint positions in meters, Y-up, shape
            `[T, 22, 3]`.

    Returns:
        `[T - 1, 67]` feature array. One frame shorter than the input, because
        the first four channels encode frame-to-frame root deltas.
    """
    positions = np.asarray(joints, dtype=np.float64).copy()
    _MP.tgt_offsets = _TGT_SKEL.get_offsets_joints(torch.from_numpy(positions[0]))
    data, _ground_positions, _positions, _l_velocity = _MP.process_file(positions, FEET_THRE)
    return data[:, :67].astype(np.float32)
