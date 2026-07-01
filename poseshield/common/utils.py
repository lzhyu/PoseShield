import numpy as np
import fcl
import torch
import matplotlib
matplotlib.use('Agg')  # For headless (no screen) environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import smplx
import trimesh
import os
import yaml

# Load global configuration paths
global_config_path_ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config_files",
    "global_path.yaml"
)
with open(global_config_path_, "r") as f:
    global_config_ = yaml.safe_load(f)
MODEL_PATH_ = global_config_["BODY_MODEL_PATH"]
def sample_6d(num_joints=23, random_state=None):
    """
    Samples random 6D rotations for each joint and returns them in axis-angle form.
    (Does NOT include global orientation; only the body joints.)
    Args:
        num_joints (int): Number of body joints to sample rotations for.
        random_state (np.random.RandomState or None): Random state for reproducibility.
    Returns:
        np.ndarray: A (num_joints, 3) array of axis-angle vectors.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    # Sample random 6D for each joint
    r_6d = random_state.randn(num_joints, 6)  # shape (23, 6)
    
    # Convert 6D -> rotation matrices -> axis-angle
    return r_6d

def normalize(x, axis=-1, eps=1e-8):
    """
    Normalize the input array along the given axis.
    
    Args:
        x (np.ndarray): Input array.
        axis (int): Axis along which to compute the norm.
        eps (float): Small constant to avoid division by zero.
    
    Returns:
        np.ndarray: Normalized array.
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / norm

def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation to a batch of 3x3 rotation matrices using
    Gram-Schmidt orthogonalisation.
    
    Args:
        d6 (np.ndarray): 6D rotation representation with shape (..., 6)
    
    Returns:
        np.ndarray: Batch of rotation matrices with shape (..., 3, 3)
    
    This implementation follows the approach in:
    Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    "On the Continuity of Rotation Representations in Neural Networks."
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # Split the input into two 3D vectors.
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    
    # Normalize the first vector.
    b1 = normalize(a1, axis=-1)
    
    # Compute the projection of a2 onto b1 and remove it to make b2 orthogonal.
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = normalize(b2, axis=-1)
    
    # Compute the third vector as the cross product of b1 and b2.
    b3 = np.cross(b1, b2, axis=-1)
    
    # Stack b1, b2, and b3 as the columns of the rotation matrix.
    # The new axis is inserted at position -2 so that the resulting shape is (..., 3, 3)
    rotation_mats = np.stack((b1, b2, b3), axis=-2)
    return rotation_mats

def matrix_to_axis_angle(R):
    """
    Convert a batch of rotation matrices R (shape (N, 3, 3))
    into axis-angle vectors (shape (N, 3)).
    """
    # Trace of each matrix
    trace = np.trace(R, axis1=1, axis2=2)
    # Clip trace to avoid numerical errors outside [-1, 3]
    trace = np.clip(trace, -1.0, 3.0)
    
    # Angle
    angles = np.arccos((trace - 1.0)/2.0)  # shape (N,)

    # For numerical stability when angle is close to 0
    # we can handle that carefully, but for random sampling
    # it is rare to get exactly 0 or pi. We'll do a simple approach here.
    # axis = (1 / (2 sin(theta))) * [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]
    # We'll handle small-angle cases by max-ing with a small eps in denominator.

    # Compute axis via the "off-diagonal trick"
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axes = np.stack([rx, ry, rz], axis=1)  # (N, 3)

    # Norm of each axis
    sin_angles = np.linalg.norm(axes, axis=1, keepdims=True) / 2.0
    # Normalize axis
    axes = axes / (2.0 * (sin_angles + 1e-8))

    # For angles near 0, the direction of the axis is not well-defined.
    # For random sampling, we can skip a special fix, but you could clamp angles etc.

    # Multiply each axis by the angle
    axis_angle = axes * angles[:, None]  # shape (N, 3)
    return axis_angle

def rotation_6d_to_matrix_torch(d6: torch.Tensor) -> torch.Tensor:
    """
    Torch (differentiable) version of rotation_6d_to_matrix.
    Converts 6D rotation representation to rotation matrices via Gram-Schmidt.

    Args:
        d6 (torch.Tensor): shape (..., 6)

    Returns:
        torch.Tensor: shape (..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_axis_angle_torch(R: torch.Tensor) -> torch.Tensor:
    """
    Torch (differentiable) version of matrix_to_axis_angle.
    Converts a batch of rotation matrices (N, 3, 3) to axis-angle vectors (N, 3)
    using Rodrigues' formula.

    Args:
        R (torch.Tensor): shape (N, 3, 3)

    Returns:
        torch.Tensor: shape (N, 3)
    """
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # (N,)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # (N,)

    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis_raw = torch.stack([rx, ry, rz], dim=1)  # (N, 3)

    safe_sin = torch.sin(angle).abs().clamp(min=1e-7).unsqueeze(1)  # (N, 1)
    unit_axis = axis_raw / (2.0 * safe_sin)
    return unit_axis * angle.unsqueeze(1)  # (N, 3)

def aa_to_mesh_torch(model, r_aa: torch.Tensor) -> torch.Tensor:
    """
    Torch (differentiable) version of aa_to_mesh.
    Takes axis-angle body joints as a torch Tensor and returns SMPL vertices
    with gradients preserved.

    Args:
        model:  smplx model instance (already on the correct device).
        r_aa:   torch.Tensor of shape (21, 3) or (63,) — body joint axis-angles.

    Returns:
        vertices: torch.Tensor of shape (V, 3), still in the compute graph.
        faces:    np.ndarray of shape (F, 3).
    """
    body_pose = r_aa.reshape(1, -1)  # (1, 63)
    output = model(
        global_orient=None,
        body_pose=body_pose,
        betas=None,
        transl=None,
        return_verts=True,
    )
    return output.joints[0][:,:22]# exclude fingers and face

def sixd_to_mesh(model, r_6d, device=torch.device('cpu'), betas=None, transl=None):
    """
    Takes a set of axis-angle body joints (from 6D) and produces SMPL mesh vertices and faces.
    Args:
        model: SMPL model object with a call signature like model(global_orient, body_pose, betas, transl).
        r_6d: 6D rotation representation (numpy array or tensor).
        device (torch.device): Device on which to run the model.
        betas (torch.Tensor or None): SMPL shape parameters.
        transl (torch.Tensor or None): Global translation.
    Returns:
        vertices (np.ndarray): The SMPL vertices (V x 3).
        faces (np.ndarray): The SMPL faces (F x 3).
    """
    rot_mats = rotation_6d_to_matrix(r_6d)     # (21, 3, 3)
    axis_angles = matrix_to_axis_angle(rot_mats)  # (21, 3)
    return aa_to_mesh(model, axis_angles, device=device, betas=betas, transl=transl)

def aa_to_mesh(model, r_aa, device=None, betas=None, transl=None):
    """
    Takes a set of axis-angle body joints (from 6D) and produces SMPL mesh vertices and faces.
    Args:
        model: SMPL model object with a call signature like model(global_orient, body_pose, betas, transl).
        axis_angles (np.ndarray): An array of shape (23, 3) for the body joints.
        device (torch.device or None): Device on which to run the model. If None, inferred from model.
        betas (torch.Tensor or None): SMPL shape parameters (1 x num_betas). Defaults to zeros.
        transl (torch.Tensor or None): Global translation (1 x 3). Defaults to zeros.
    Returns:
        vertices (np.ndarray): The SMPL vertices (V x 3).
        faces (np.ndarray): The SMPL faces (F x 3).
    """
    # Infer device from model to avoid CPU/CUDA mismatch when caller omits device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    # Default betas
    if betas is None:
        betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
    # Default translation
    if transl is None:
        transl = torch.zeros((1, 3), dtype=torch.float32, device=device)

    # SMPL's full pose has 72 dims: [3 (global) + 69 (23 joints × 3)]
    # We'll keep the global orientation at zero
    pose = torch.zeros((1, 66), dtype=torch.float32, device=device)
    # Flatten the (23, 3) -> (69,) and place it into pose starting at index 3
    body_pose_axis_angle = r_aa.reshape(-1)  # shape (69,)
    pose[:, 3:] = torch.from_numpy(body_pose_axis_angle).to(device)

    # Forward pass through SMPL
    output = model(
        # global_orient=pose[:, :3],    # zero global orientation
        global_orient=None,
        body_pose=pose[:, 3:],        # 23 axis-angle joints
        betas=None,
        transl=None,
        return_verts=True
    )
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int32)
    return vertices, faces

def build_per_triangle_bvh(vertices, faces):
    """
    Build a BVH where each triangle is added as a submodel with a unique ID.
    That way, in collisions, you can distinguish which triangles collided.
    """
    bvh = fcl.BVHModel()
    # The total number of triangles = len(faces)
    bvh.beginModel(len(vertices), len(faces))
    
    # Instead of addSubModel(...) once, do addTriangle(...) per face
    for f_idx, face in enumerate(faces):
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]
        bvh.addTriangle(v1, v2, v3)
        
    bvh.endModel()
    return bvh

def visualize_smpl(vertices, faces, save_path="smpl_pose.png", color='#54a0ff'):
    """
    Visualize SMPL mesh in 3D and save to a PNG file (no display required).
    """
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Map SMPL coordinates to Matplotlib 3D space:
    # SMPL has Y-up, Z-forward, X-right.
    # Matplotlib 3D has Z-vertical. So we map SMPL Y -> Matplotlib Z, and SMPL Z -> Matplotlib Y.
    x_plt = vertices[:, 0]     # Right-Left
    y_plt = vertices[:, 2]     # Forward-Backward (Depth)
    z_plt = vertices[:, 1]     # Up-Down (Height)

    # Tri-surface plot of the mesh with a premium color
    ax.plot_trisurf(
        x_plt,
        y_plt,
        z_plt,
        triangles=faces,
        shade=True,
        color=color,      # Custom color
        edgecolor='none',
        alpha=0.9
    )

    # Make axes roughly equal so the mesh isn't distorted
    all_coords = np.stack([x_plt, y_plt, z_plt], axis=-1)
    min_vals = np.min(all_coords, axis=0)
    max_vals = np.max(all_coords, axis=0)
    ranges = max_vals - min_vals
    max_range = max(ranges)
    mid = (max_vals + min_vals) / 2
    
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    # Set camera view angle to make the body stand upright and face the screen
    ax.view_init(elev=15, azim=90)

    # Hide grid and axes for a clean render
    ax.axis('off')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved SMPL visualization to {save_path}")

def make_6d_rotation_valid(rot_6d: np.ndarray) -> np.ndarray:
    """
    Given a 6D rotation vector (two raw 3D vectors), orthonormalize them
    so they represent the first two columns of a valid 3x3 rotation matrix.
    Returns a 6D vector (still two 3D columns).
    
    Args:
        rot_6d (np.ndarray): shape (6,) or (N, 6) if batching.
    
    Returns:
        np.ndarray: shape (6,) or (N, 6), an orthonormalized 6D rotation.
    """
    # If we get a single 6D vector, reshape to (1, 6) for consistent processing
    single_input = False
    if rot_6d.ndim == 1:
        rot_6d = rot_6d[np.newaxis, :]  # shape (1, 6)
        single_input = True

    # Orthonormalize each 6D vector row by row
    out = []
    for row in rot_6d:
        # Split into x (first 3D vector) and y (second 3D vector)
        x_raw = row[:3]
        y_raw = row[3:]
        
        # 1) Normalize x
        x_norm = x_raw / np.linalg.norm(x_raw)
        
        # 2) Make y orthonormal to x
        #    remove component of y in direction x, then normalize
        y_perp = y_raw - np.dot(x_norm, y_raw) * x_norm
        y_norm = y_perp / np.linalg.norm(y_perp)
        
        # Concatenate back to a 6D vector
        valid_6d = np.concatenate([x_norm, y_norm], axis=0)
        out.append(valid_6d)

    out = np.stack(out, axis=0)  # shape (batch_size, 6)
    
    # If input was single vector, squeeze back to (6,)
    if single_input:
        out = out[0]
    return out

_cached_smpl_model = None

def quick_viz_6d(r_6d, save_path="smpl_pose.png", mesh_path=None, smpl_model=None, color='#54a0ff'):
    """
    Visualize a 6D rotation vector as a mesh and save to a PNG file.
    r_6d: 21*6
    """
    # Convert 6D to mesh
    assert r_6d.shape[0] == 21, "Expected 6D rotation vector with 21 joints."

    if smpl_model is None:
        global _cached_smpl_model
        device = torch.device('cpu')
        if _cached_smpl_model is None:
            model_path = MODEL_PATH_  # Folder containing SMPL .pkl files
            _cached_smpl_model = smplx.create(
                model_path,
                model_type='smplh',
                gender='neutral',
                ext='npz',
                use_pca=False
            ).to(device)
        smpl_model = _cached_smpl_model
    else:
        # Infer device from externally provided model
        try:
            device = next(smpl_model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    vertices, faces = sixd_to_mesh(smpl_model, r_6d, device=device)
        
    # Visualize the mesh
    visualize_smpl(vertices, faces, save_path=save_path, color=color)
    if mesh_path!=None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # Export as OBJ
        mesh.export(mesh_path)

import numpy as np

def axis_angle_to_matrix(axis_angle):
    """
    Convert a batch of axis-angle vectors (shape (N, 3))
    into rotation matrices (shape (N, 3, 3)).
    Uses Rodrigues' rotation formula.
    """
    aa = np.asarray(axis_angle, dtype=np.float64)
    assert aa.ndim == 2 and aa.shape[1] == 3, "axis_angle must have shape (N, 3)"
    N = aa.shape[0]

    # angles: (N, 1)
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    eps = 1e-8

    # unit axes: (N, 3)  (safe for small angles)
    k = aa / (theta + eps)

    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]

    # Skew-symmetric cross-product matrices K: (N, 3, 3)
    K = np.zeros((N, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -kz
    K[:, 0, 2] =  ky
    K[:, 1, 0] =  kz
    K[:, 1, 2] = -kx
    K[:, 2, 0] = -ky
    K[:, 2, 1] =  kx

    I = np.eye(3, dtype=np.float64)[None, :, :]  # (1, 3, 3)

    # Rodrigues: R = I + sinθ K + (1-cosθ) K^2
    sin_t = np.sin(theta)[:, None]   # (N, 1, 1) after [:, None]
    cos_t = np.cos(theta)[:, None]

    sin_t = sin_t.reshape(N, 1, 1)
    cos_t = cos_t.reshape(N, 1, 1)

    K2 = K @ K

    # For tiny angles, use series expansions to avoid 0/0-ish behavior:
    # sinθ ~ θ, (1-cosθ) ~ θ^2/2
    small = (theta.reshape(N) < 1e-4)
    A = np.empty((N, 1, 1), dtype=np.float64)  # sinθ
    B = np.empty((N, 1, 1), dtype=np.float64)  # (1-cosθ)

    # normal
    A[~small] = sin_t[~small]
    B[~small] = (1.0 - cos_t[~small])

    # series
    th = theta.reshape(N, 1, 1)
    A[small] = th[small] - (th[small]**3)/6.0
    B[small] = (th[small]**2)/2.0 - (th[small]**4)/24.0

    R = I + A * K + B * K2
    return R

import numpy as np

def matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    """
    Inverse of rotation_6d_to_matrix (as implemented above).

    Args:
        R (np.ndarray): rotation matrices with shape (..., 3, 3)

    Returns:
        np.ndarray: 6D rotation representation with shape (..., 6)
    """
    R = np.asarray(R)
    assert R.shape[-2:] == (3, 3), "R must have shape (..., 3, 3)"

    b1 = R[..., 0, :]  # first row
    b2 = R[..., 1, :]  # second row

    d6 = np.concatenate([b1, b2], axis=-1)
    return d6

def load_model(cfg, model_path=None, device='cuda'):
    """Load a ResidualMLP model from config and optional checkpoint."""
    from poseshield.common.network import ResidualMLP
    model = ResidualMLP(
        in_dim=cfg.MODEL.IN_DIM,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_layers=cfg.MODEL.NUM_LAYERS,
        activation=cfg.MODEL.ACTIVATION
    ).to(device)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    return model

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 10
    aa = rng.standard_normal((N, 3)) * 0.5
    R = axis_angle_to_matrix(aa)
    aa2 = matrix_to_axis_angle(R)
    assert np.max(np.abs(aa - aa2)) < 1e-6

    d6 = rng.standard_normal((1000, 6))
    R = rotation_6d_to_matrix(d6)
    d6_rec = matrix_to_rotation_6d(R)

    R_rec = rotation_6d_to_matrix(d6_rec)
    err_R = np.max(np.abs(R - R_rec))
    assert err_R < 1e-5
