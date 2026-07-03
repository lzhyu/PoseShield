"""Standalone Demo Script for Shape-conditioned SAField.

This script demonstrates that when starting from the SAME colliding pose, the
network optimizes the pose differently depending on the shape parameters (beta)
to resolve the collision.

Features:
- Loads a pre-trained SAFieldNetwork checkpoint.
- Runs SLSQP optimization for a colliding pose under two distinct shape conditions.
- Outputs the field values, distance of optimized poses, and MVD (Mean Vertex Displacement) between them.
- (Optional) Runs SMPL+H forward pass to export resolved meshes as `.obj` files.
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml

from network import SAFieldNetwork
from inference import optimize_pose


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = DEMO_DIR / "config.yaml"


def load_config(path: Path) -> dict:
    """Load the SAField demo configuration."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def resolve_config_path(config_path: str) -> Path:
    """Resolve a config path relative to the demo directory when needed."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return DEMO_DIR / path


def checkpoint_from_config(config: dict, config_path: Path) -> Path:
    """Return the checkpoint path declared by the config."""
    checkpoint = config.get("checkpoint", {})
    filename = checkpoint.get("filename", "best_scc_model.pth")
    path = Path(filename)
    if path.is_absolute():
        return path
    return config_path.parent / path


# --- SMPL-H reconstruction helpers (for OBJ export) ---

def normalize(x, axis=-1, eps=1e-8):
    norm = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / norm

def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = normalize(a1, axis=-1)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = normalize(b2, axis=-1)
    b3 = np.cross(b1, b2, axis=-1)
    rotation_mats = np.stack((b1, b2, b3), axis=-2)
    return rotation_mats

def matrix_to_axis_angle(R):
    trace = np.trace(R, axis1=1, axis2=2)
    trace = np.clip(trace, -1.0, 3.0)
    angles = np.arccos((trace - 1.0)/2.0)
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axes = np.stack([rx, ry, rz], axis=1)
    sin_angles = np.linalg.norm(axes, axis=1, keepdims=True) / 2.0
    axes = axes / (2.0 * (sin_angles + 1e-8))
    axis_angle = axes * angles[:, None]
    return axis_angle

def sixd_to_mesh(model, r_6d, device=torch.device('cpu'), betas=None):
    if betas is None:
        betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
    pose = torch.zeros((1, 66), dtype=torch.float32, device=device)
    rot_mats = rotation_6d_to_matrix(r_6d)
    axis_angles = matrix_to_axis_angle(rot_mats)
    body_pose_axis_angle = axis_angles.reshape(-1)
    pose[:, 3:] = torch.from_numpy(body_pose_axis_angle).to(device)

    output = model(
        global_orient=None,
        body_pose=pose[:, 3:],
        betas=betas,
        transl=None,
        return_verts=True
    )
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = model.faces.astype(np.int32)
    return vertices, faces

def save_obj(vertices, faces, path):
    """Save vertices and faces as a wavefront OBJ file."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ files are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def main():
    parser = argparse.ArgumentParser(description="Shape-Conditioned SAField Demo")
    parser.add_argument(
        "--config_path", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to the SAField demo config YAML"
    )
    parser.add_argument(
        "--model_path", type=str, default="",
        help="Optional path to the trained SAField model checkpoint; overrides config"
    )
    parser.add_argument(
        "--examples_path", type=str, default=str(DEMO_DIR / "demo_examples.json"),
        help="Path to the demo_examples.json file containing the poses"
    )
    parser.add_argument(
        "--example_idx", type=int, default=0,
        help="Index of the example to run (0 to 4)"
    )
    parser.add_argument(
        "--smpl_model_path", type=str, default="",
        help="Path to the SMPL+H body model folder (e.g. containing 'smplh' folder) for OBJ exporting"
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override the configured field threshold")
    parser.add_argument("--max_itr", type=int, default=None, help="Override the configured SLSQP iteration count")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Running on device: {device}")
    config_path = resolve_config_path(args.config_path)
    config = load_config(config_path)
    model_path = Path(args.model_path) if args.model_path else checkpoint_from_config(config, config_path)
    model_cfg = config.get("model", {})
    optim_cfg = config.get("optimization", {})
    threshold = float(args.threshold if args.threshold is not None else optim_cfg.get("threshold", 0.05))
    max_itr = int(args.max_itr if args.max_itr is not None else optim_cfg.get("max_itr", 100))
    resolved_tol = float(optim_cfg.get("resolved_tolerance", 0.005))

    # 1. Load SAField model
    if not model_path.is_file():
        raise FileNotFoundError(
            f"SAField checkpoint not found: {model_path}. "
            "Download and extract the SAField demo asset package first."
        )
    print(f"Loading SAField config from {config_path}...")
    print(f"Loading SAField model from {model_path}...")
    model = SAFieldNetwork(
        theta_dim=int(model_cfg.get("theta_dim", 126)),
        beta_dim=int(model_cfg.get("beta_dim", 10)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        K=int(model_cfg.get("K", 8)),
        num_layers_g0=int(model_cfg.get("num_layers_g0", 12)),
        num_layers_phi=int(model_cfg.get("num_layers_phi", 6)),
        num_layers_shape=int(model_cfg.get("num_layers_shape", 4)),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("SAField model loaded successfully.")

    # 2. Load examples
    if not os.path.exists(args.examples_path):
        print(f"Error: examples path '{args.examples_path}' does not exist.")
        return
    with open(args.examples_path, "r") as f:
        examples = json.load(f)
    
    if args.example_idx < 0 or args.example_idx >= len(examples):
        print(f"Error: example_idx must be between 0 and {len(examples)-1}.")
        return
    
    ex = examples[args.example_idx]
    print(f"\n--- Running Example index {args.example_idx} (Dataset Pose Index {ex['idx']}) ---")

    pose_init = np.array(ex["pose_init"], dtype=np.float32)
    # Keep the historical JSON keys for compatibility; the selected shapes may
    # now use any beta values in [-2, 2], not only thin/fat beta0 endpoints.
    beta_shape_a = np.array(ex["beta_thin"], dtype=np.float32)
    beta_shape_b = np.array(ex["beta_fat"], dtype=np.float32)

    # 3. Perform collision resolution optimization for both shapes
    print(f"\nRunning pose optimization for Shape A (beta = {np.round(beta_shape_a, 3).tolist()})...")
    opt_theta_a, success_a = optimize_pose(
        pose_init, beta_shape_a, model, device, max_itr=max_itr, threshold=threshold
    )

    print(f"Running pose optimization for Shape B (beta = {np.round(beta_shape_b, 3).tolist()})...")
    opt_theta_b, success_b = optimize_pose(
        pose_init, beta_shape_b, model, device, max_itr=max_itr, threshold=threshold
    )

    # 4. Evaluate fields
    with torch.no_grad():
        t_init_t = torch.tensor(pose_init, dtype=torch.float32).unsqueeze(0)
        t_a_t = torch.tensor(opt_theta_a, dtype=torch.float32).unsqueeze(0)
        t_b_t = torch.tensor(opt_theta_b, dtype=torch.float32).unsqueeze(0)
        b_a_t = torch.tensor(beta_shape_a, dtype=torch.float32).unsqueeze(0)
        b_b_t = torch.tensor(beta_shape_b, dtype=torch.float32).unsqueeze(0)

        g_init_a = model(t_init_t, b_a_t).item()
        g_init_b = model(t_init_t, b_b_t).item()
        g_opt_a = model(t_a_t, b_a_t).item()
        g_opt_b = model(t_b_t, b_b_t).item()

    pose_diff_l2 = np.linalg.norm(opt_theta_a - opt_theta_b)

    print("\n================== RESULTS ==================")
    accepted_a = g_opt_a >= threshold - resolved_tol
    accepted_b = g_opt_b >= threshold - resolved_tol
    print("Shape A:")
    print(f"  Initial Field: {g_init_a:.4f} (Colliding)")
    print(f"  Optimized Field: {g_opt_a:.4f} (Accepted by field threshold: {accepted_a}, solver_success: {success_a})")
    print("Shape B:")
    print(f"  Initial Field: {g_init_b:.4f} (Colliding)")
    print(f"  Optimized Field: {g_opt_b:.4f} (Accepted by field threshold: {accepted_b}, solver_success: {success_b})")
    print(f"Pose parameters L2 difference: {pose_diff_l2:.4f}")

    # 5. Optional SMPL-H vertex reconstruction & OBJ export
    if args.smpl_model_path:
        if not os.path.exists(args.smpl_model_path):
            print(f"\nWarning: SMPL-H model path '{args.smpl_model_path}' does not exist. Skipping OBJ export.")
            return

        import smplx
        print(f"\nLoading SMPL-H model from {args.smpl_model_path} for exporting meshes...")
        smpl_model = smplx.create(
            args.smpl_model_path,
            model_type="smplh",
            gender="male",
            ext="npz",
            use_pca=False
        ).to(device)

        # Reconstruct meshes
        # Note: we can reconstruct them on the SAME body shape to measure the PURE pose-induced vertex displacement
        verts_a_same, faces = sixd_to_mesh(smpl_model, opt_theta_a.reshape(21, 6), device=device, betas=b_a_t)
        verts_b_same, _ = sixd_to_mesh(smpl_model, opt_theta_b.reshape(21, 6), device=device, betas=b_a_t)
        pure_pose_mvd = np.mean(np.linalg.norm(verts_a_same - verts_b_same, axis=-1))

        # Reconstruct them on their OWN respective shapes
        verts_a_own, _ = sixd_to_mesh(smpl_model, opt_theta_a.reshape(21, 6), device=device, betas=b_a_t)
        verts_b_own, _ = sixd_to_mesh(smpl_model, opt_theta_b.reshape(21, 6), device=device, betas=b_b_t)

        print(f"Mean Vertex Displacement (MVD) purely from pose change: {pure_pose_mvd * 100:.2f} cm")

        # Export meshes
        out_a_path = f"resolved_pose_idx{args.example_idx}_shape_a.obj"
        out_b_path = f"resolved_pose_idx{args.example_idx}_shape_b.obj"
        save_obj(verts_a_own, faces, out_a_path)
        save_obj(verts_b_own, faces, out_b_path)
        print(f"Exported OBJ: {out_a_path}")
        print(f"Exported OBJ: {out_b_path}")
        print("You can open these OBJ files in Blender or MeshLab to visualize the shape-conditioned collision resolution!")
    else:
        print("\nNote: Provide --smpl_model_path to export optimized meshes as OBJ files.")

if __name__ == "__main__":
    main()
