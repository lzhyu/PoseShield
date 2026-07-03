"""Generate an interactive original-versus-optimized motion visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.hymotion.utils.motion_format import load_motion


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""
    parser = argparse.ArgumentParser(description="Generate a static HTML motion comparison")
    parser.add_argument("--sequence", required=True, help="Sequence label used in output filenames")
    parser.add_argument("--original", type=Path, required=True, help="Original canonical motion")
    parser.add_argument("--optimized", type=Path, required=True, help="Optimized canonical motion")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the generated HTML")
    parser.add_argument("--hide-captions", action="store_true", help="Hide overlay labels")
    return parser.parse_args()


def load_public_motion(path: Path) -> dict:
    """Convert public canonical motion to visualizer SMPL data."""
    from poseshield.hymotion.pipeline.body_model import construct_smpl_data_dict

    motion = load_motion(path)
    rotations = torch.from_numpy(motion[:, :132]).float().reshape(-1, 22, 6)
    translation = torch.from_numpy(motion[:, 132:135]).float()
    return construct_smpl_data_dict(rotations, translation)


def save_npz(path: Path, smpl_data: dict) -> None:
    """Save one visualizer sample."""
    payload = {"gender": np.array([smpl_data.get("gender", "neutral")], dtype=str)}
    for key in ("Rh", "trans", "poses", "betas"):
        if key in smpl_data:
            value = smpl_data[key]
            payload[key] = value.cpu().numpy() if torch.is_tensor(value) else value
    np.savez_compressed(path, **payload)


def to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a tensor-like value to NumPy."""
    return value.cpu().numpy() if torch.is_tensor(value) else value


def main() -> None:
    """Create an original-vs-optimized HTML visualization."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original = load_public_motion(args.original.resolve())
    optimized = load_public_motion(args.optimized.resolve())
    if original["poses"].shape != optimized["poses"].shape:
        raise ValueError(
            "Original and optimized pose shapes differ: "
            f"{original['poses'].shape} != {optimized['poses'].shape}"
        )
    if not np.array_equal(to_numpy(original["trans"]), to_numpy(optimized["trans"])):
        raise AssertionError("Optimized translation does not exactly match the original")

    base_name = f"{args.sequence}_original_vs_optimized"
    cache_folder = f"output/visualizations/{args.sequence}"
    cache_dir = PROJECT_ROOT / "poseshield" / cache_folder
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_npz(cache_dir / f"{base_name}_000.npz", original)
    save_npz(cache_dir / f"{base_name}_001.npz", optimized)

    metadata = {
        "timestamp": f"visualization_{args.sequence}",
        "text": f"Sequence {args.sequence} collision-resolution comparison",
        "text_rewrite": [
            "Red / left: original motion",
            "Green / right: optimized pose with the original translation trajectory",
        ],
        "num_samples": 2,
        "base_filename": base_name,
        "source_original": str(args.original.resolve()),
        "source_optimized": str(args.optimized.resolve()),
    }
    with (cache_dir / f"{base_name}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    from poseshield.hymotion.utils.visualize_mesh_web import generate_static_html

    output_path = generate_static_html(
        folder_name=cache_folder,
        file_name=base_name,
        output_dir=str(output_dir),
        hide_captions=args.hide_captions,
    )
    print(f"Visualization written to {output_path}")


if __name__ == "__main__":
    main()
