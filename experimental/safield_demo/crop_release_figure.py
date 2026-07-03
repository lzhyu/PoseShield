"""Crop the full Blender SAField render into a README-friendly teaser image."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "assets/safield_experimental_shape_demo_blender_full.png"
OUTPUT_PATH = ROOT / "assets/safield_experimental_shape_demo_blender.png"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top", type=float, default=0.03, help="Fraction cropped from the top.")
    parser.add_argument("--bottom", type=float, default=1.0, help="Bottom fraction retained after cropping.")
    return parser.parse_args()


def crop_release_figure(input_path: Path, output_path: Path, top: float, bottom: float) -> None:
    """Write a full-body README crop with modest top whitespace removed."""
    if not 0.0 <= top < bottom <= 1.0:
        raise ValueError(f"Invalid crop fractions: top={top}, bottom={bottom}")
    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    crop_box = (
        0,
        int(height * top),
        width,
        int(height * bottom),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(output_path, quality=96)


def main() -> None:
    """Crop the full render into the final README image."""
    args = parse_args()
    input_path = args.input if args.input.exists() else args.output
    crop_release_figure(input_path, args.output, args.top, args.bottom)
    print(f"Cropped {input_path} -> {args.output}")


if __name__ == "__main__":
    main()
