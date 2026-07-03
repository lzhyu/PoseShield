"""Export current SAField experimental demo meshes as release OBJ assets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import smplx
import torch

from poseshield.common.utils import sixd_to_mesh


DEFAULT_OUTPUT_DIR = DEMO_DIR / "artifacts/release_example/release_objs"
DEFAULT_SMPL_MODEL_PATH = REPO_ROOT / "deps/body_models"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-path", type=Path, default=DEMO_DIR / "demo_examples.json")
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument(
        "--smpl-model-path",
        type=Path,
        default=None,
        help=(
            "Path to the parent folder containing the SMPL-H 'smplh' directory. "
            "Defaults to $SMPL_MODEL_PATH or deps/body_models."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.smpl_model_path is None:
        args.smpl_model_path = Path(os.environ.get("SMPL_MODEL_PATH", DEFAULT_SMPL_MODEL_PATH))
    return args


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write one mesh as a Wavefront OBJ file."""
    with path.open("w", encoding="utf-8") as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def main() -> None:
    """Export A/B input and resolved meshes for Blender rendering."""
    args = parse_args()
    if not args.smpl_model_path.exists():
        raise FileNotFoundError(
            f"SMPL-H model path does not exist: {args.smpl_model_path}. "
            "Pass --smpl-model-path or set SMPL_MODEL_PATH."
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = json.loads(args.examples_path.read_text(encoding="utf-8"))
    if args.example_idx < 0 or args.example_idx >= len(examples):
        raise IndexError(f"example_idx {args.example_idx} out of range for {len(examples)} examples")
    example = examples[args.example_idx]
    device = torch.device("cpu")
    smpl_model = smplx.create(
        args.smpl_model_path,
        model_type="smplh",
        gender="male",
        ext="npz",
        use_pca=False,
    ).to(device)

    pose_init = np.asarray(example["pose_init"], dtype=np.float32).reshape(21, 6)
    pose_a = np.asarray(example["opt_theta_thin"], dtype=np.float32).reshape(21, 6)
    pose_b = np.asarray(example["opt_theta_fat"], dtype=np.float32).reshape(21, 6)
    beta_a = torch.tensor(example["beta_thin"], dtype=torch.float32, device=device).view(1, 10)
    beta_b = torch.tensor(example["beta_fat"], dtype=torch.float32, device=device).view(1, 10)

    meshes = {
        "shape_a_input.obj": sixd_to_mesh(smpl_model, pose_init, device=device, betas=beta_a),
        "shape_b_input.obj": sixd_to_mesh(smpl_model, pose_init, device=device, betas=beta_b),
        "shape_a_resolved.obj": sixd_to_mesh(smpl_model, pose_a, device=device, betas=beta_a),
        "shape_b_resolved.obj": sixd_to_mesh(smpl_model, pose_b, device=device, betas=beta_b),
    }
    for filename, (vertices, faces) in meshes.items():
        save_obj(output_dir / filename, vertices, faces)
    print(output_dir)


if __name__ == "__main__":
    main()
