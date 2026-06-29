from poseshield.common.network import ResidualMLP
from poseshield.pose.dataset import PosesDataset
import torch
import numpy as np
from poseshield.common.utils import (
    rotation_6d_to_matrix, matrix_to_axis_angle,
    rotation_6d_to_matrix_torch, matrix_to_axis_angle_torch, aa_to_mesh_torch,
)

import os
import pickle

def constraint_function(model: ResidualMLP, x: torch.Tensor) -> torch.Tensor:
    """
    Apply the model to input x and return its first output element as the constraint value.
    """
    output = model(x.unsqueeze(0))
    return output.squeeze(0).squeeze(0)

# def cost_function(x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the L2 distance between x and a reference tensor x_ref.
#     """
#     return torch.linalg.norm(x - x_ref)

def cost_function(x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """
    Weighted pose distance between x and x_ref (both shape (126,)).

    For each of the 21 joints we compute the L2 norm of its 6D rotation
    difference, then take a weighted sum using per-joint weights.

    weights (21,): one weight per joint.
    Defaults to SMPLH_POSE_WEIGHTS — subtree-size weights normalized to 1,
    so joints higher in the kinematic chain are penalised more.
    """
    return cost_function_weighted(x, x_ref)

# ── SMPLH kinematic-chain weights ─────────────────────────────────────────────
# Weight for each of the 21 body joints (indices 1-21, root excluded) =
# size of its downstream subtree in the SMPLH kinematic tree.
# The bigger the subtree, the more joints are affected by that rotation,
# so we penalise deviations there more in the cost function.
#
# SMPLH body kinematic tree (joints 1-21, parent → children):
#   spine1(3) → spine2(6) → spine3(9) → neck(12) → head(15)
#                                       → left_collar(13)  → left_shoulder(16)  → left_elbow(18)  → left_wrist(20)
#                                       → right_collar(14) → right_shoulder(17) → right_elbow(19) → right_wrist(21)
#   left_hip(1)  → left_knee(4)  → left_ankle(7)  → left_foot(10)
#   right_hip(2) → right_knee(5) → right_ankle(8) → right_foot(11)
#
# Subtree sizes  (joint 1 … 21):
_SUBTREE_SIZES = [
#  j1   j2   j3   j4  j5   j6  j7  j8   j9  j10 j11  j12  j13  j14 j15  j16  j17 j18 j19 j20 j21
    4,   4,  13,   3,  3,  12,  2,  2,  11,   1,  1,   2,   4,   4,  1,   3,   3,  2,  2,  1,  1
]
# Expand to 126 dims (×6 per joint), normalize to sum = 1
# Per-joint weight vector, shape (21,).  One entry per local rotation.
SMPLH_POSE_WEIGHTS: torch.Tensor = torch.tensor(_SUBTREE_SIZES, dtype=torch.float32)
SMPLH_POSE_WEIGHTS /= SMPLH_POSE_WEIGHTS.sum()

def cost_function_weighted(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Weighted pose distance between x and x_ref (both shape (126,)).

    For each of the 21 joints we compute the L2 norm of its 6D rotation
    difference, then take a weighted sum using per-joint weights.

    weights (21,): one weight per joint.
    Defaults to SMPLH_POSE_WEIGHTS — subtree-size weights normalized to 1,
    so joints higher in the kinematic chain are penalised more.
    """
    if weights is None:
        weights = SMPLH_POSE_WEIGHTS.to(x.device)

    diff = (x - x_ref).reshape(21, 6)           # (21, 6)
    per_joint_norm = torch.linalg.norm(diff, dim=-1)  # (21,)
    return (per_joint_norm * weights).sum()

def _local_pose_to_vertices(x: torch.Tensor, smpl_model) -> torch.Tensor:
    """
    Convert a 126-dim SMPL local pose (21 joints × 6D rotation) to SMPL mesh
    vertices in a fully differentiable manner.

    Args:
        x: torch.Tensor of shape (126,), the 21-joint 6D rotation local pose.
        smpl_model: an smplx model instance (smplh / smpl, already on the right device).

    Returns:
        vertices: torch.Tensor of shape (V, 3) with gradients preserved.
    """
    r6d = x.reshape(21, 6)                                   # (21, 6)
    rot_mats = rotation_6d_to_matrix_torch(r6d)              # (21, 3, 3)
    axis_angles = matrix_to_axis_angle_torch(rot_mats)       # (21, 3)
    joints = aa_to_mesh_torch(smpl_model, axis_angles)  # (V, 3)
    return joints

def find_datum(model: ResidualMLP, dataset: PosesDataset, n_samples: int, device: torch.device, thresh: float = -0.1):
    """
    Search the dataset for a sample where the model output is below thresh and label == -1.
    Returns (sample, index) or (None, None) if not found.
    """
    data, labels = dataset.sample_from_dataset(n_samples)
    for idx, (sample, label) in enumerate(zip(data, labels)):
        x = torch.from_numpy(sample.reshape(-1)).float().to(device)
        if float(constraint_function(model, x)) < thresh and label == -1:
            return sample, idx
    return None, None

def save_sample_as_param_pickle(
    output_path: str,
    r6d: np.ndarray,
    model_type: str = "smplh",
    betas: np.ndarray= None,
    global_orient: np.ndarray = None,
    transl: np.ndarray= None,
):
    """
    Save a pose sample (in 6D rotation format) to a pickle compatible with scripts
    that expect keys: 'betas', 'global_pose', and 'pose'.

    Args:
        output_path: Destination .pkl file path.
        r6d: Pose in 6D rotation representation. Shape (N*6,) or (N, 6).
             For smplh, N is typically 21 body joints (63 AA dims).
             For smpl, N is typically 23 body joints (69 AA dims).
        model_type: One of {'smpl', 'smplx', 'smplh'}. Determines expected body joints count.
        betas: Optional shape parameters (10,). Defaults to zeros if None.
        global_orient: Optional root orientation in axis-angle (3,). Defaults to zeros if None.
        transl: Optional translation (3,). Stored for completeness if provided.
    """
    # Normalize input shape to (N, 6)
    r6d = np.asarray(r6d)
    if r6d.ndim == 1:
        assert r6d.size % 6 == 0, "Input r6d must have a multiple of 6 elements"
        r6d = r6d.reshape(-1, 6)
    elif r6d.ndim == 2 and r6d.shape[1] == 6:
        pass
    else:
        raise ValueError("r6d must have shape (N,6) or (N*6,)")

    # Determine expected joint count by model type
    expected_joints = {
        'smpl': 23,   # body joints → 69 dims AA
        'smplh': 21,  # body joints (hands handled separately) → 63 dims AA
        'smplx': 21,  # body joints (face/hands separate); keep body part only
    }.get(model_type, 21)

    # Pad or trim to expected size conservatively
    if r6d.shape[0] < expected_joints:
        pad = np.zeros((expected_joints - r6d.shape[0], 6), dtype=r6d.dtype)
        r6d_use = np.concatenate([r6d, pad], axis=0)
    else:
        r6d_use = r6d[:expected_joints]

    # Convert 6D → rotation matrices → axis-angle
    rot_mats = rotation_6d_to_matrix(r6d_use)            # (N, 3, 3)
    axis_angles = matrix_to_axis_angle(rot_mats)         # (N, 3)
    body_pose_aa = axis_angles.reshape(-1).astype(np.float32)

    data = {
        'betas': (np.zeros(10, dtype=np.float32) if betas is None else betas.astype(np.float32)),
        'global_pose': (np.zeros(3, dtype=np.float32) if global_orient is None else global_orient.astype(np.float32)),
        'pose': body_pose_aa,  # body joints only, in axis-angle， 21*3
    }
    if transl is not None:
        data['transl'] = transl.astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
