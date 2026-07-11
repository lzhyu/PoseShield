#!/usr/bin/env python3
"""Validate that required PoseShield release assets are present."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AssetSpec:
    """Description of a file required by a PoseShield workflow."""

    label: str
    paths: tuple[Path, ...]
    workflow: str
    required: bool = True
    validator: str | None = None


SHARED_ASSETS = (
    AssetSpec(
        "Neutral SMPL-H body model",
        (Path("deps/body_models/smplh/SMPLH_NEUTRAL.npz"),),
        "pose and motion mesh generation",
    ),
    AssetSpec(
        "Pose collision field checkpoint",
        (Path("ckpts/poseshield/model.pth"),),
        "pose collision detection and correction",
    ),
    AssetSpec(
        "Pose collision field config",
        (Path("ckpts/poseshield/config.yaml"),),
        "pose collision detection and correction",
    ),
)

MOTION_ASSETS = (
    AssetSpec(
        "Motion collision field checkpoint",
        (Path("ckpts/poseshield/model_elu.pth"),),
        "motion collision resolution",
    ),
    AssetSpec(
        "Motion collision field config",
        (Path("ckpts/poseshield/config_elu.yaml"),),
        "motion collision resolution",
    ),
    AssetSpec(
        "HY-Motion checkpoint",
        (Path("ckpts/tencent/HY-Motion-1.0-Lite/latest.ckpt"),),
        "motion generation and DNO",
    ),
    AssetSpec(
        "HY-Motion config",
        (
            Path("ckpts/tencent/HY-Motion-1.0-Lite/config.yaml"),
            Path("ckpts/tencent/HY-Motion-1.0-Lite/config.yml"),
        ),
        "motion generation and DNO",
    ),
    AssetSpec(
        "HY-Motion normalization mean",
        (Path("ckpts/tencent/HY-Motion-1.0-Lite/stats/Mean.npy"),),
        "motion generation and DNO",
    ),
    AssetSpec(
        "HY-Motion normalization std",
        (Path("ckpts/tencent/HY-Motion-1.0-Lite/stats/Std.npy"),),
        "motion generation and DNO",
    ),
)

EXACT_FCL_ASSETS = (
    AssetSpec(
        "Exact-FCL mesh topology distances",
        (Path("deps/topology_distances_30_60.npz"),),
        "exact mesh self-collision validation",
    ),
)

POSE_DATA_ASSETS = (
    AssetSpec(
        "HwC pose training split list",
        (Path("data/dataset/train_list.csv"),),
        "HwC pose training",
    ),
    AssetSpec(
        "HwC pose test split list",
        (Path("data/dataset/test_list.csv"),),
        "HwC pose evaluation",
    ),
    AssetSpec(
        "HwC augmented pose files",
        (Path("data/dataset/augmented_data/*.npz"),),
        "HwC pose training",
    ),
    AssetSpec(
        "HwC ground-truth pose files",
        (Path("data/dataset/gt_data/*.npz"),),
        "HwC pose training and evaluation",
    ),
    AssetSpec(
        "HwC benchmark metadata",
        (Path("data/dataset_test/*.pkl"),),
        "pose collision-resolution benchmark evaluation",
    ),
    AssetSpec(
        "HwC benchmark mesh files",
        (Path("data/dataset_test/*.obj"),),
        "pose collision-resolution benchmark evaluation",
    ),
    AssetSpec(
        "HwC benchmark preview images",
        (Path("data/dataset_test/*.png"),),
        "pose collision-resolution benchmark evaluation",
    ),
)

MOTION_DATA_ASSETS = (
    AssetSpec(
        "Canonical MotionFix motion files",
        (Path("data/motion_canonical/motionfix_*_135.npy"),),
        "motion evaluation",
        validator="public_motion_format",
    ),
)


OPTIONAL_ASSETS = (
    AssetSpec(
        "Experimental SAField checkpoint",
        (Path("experimental/safield_demo/sa_model.pth"),),
        "optional shape-aware demo",
        required=False,
    ),
    AssetSpec(
        "Experimental SAField config",
        (Path("experimental/safield_demo/sa_config.yaml"),),
        "optional shape-aware demo",
        required=False,
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the namespace."""
    parser = argparse.ArgumentParser(
        description="Check PoseShield checkpoints, body models, and data assets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="PoseShield repository root. Defaults to this script's parent repo.",
    )
    parser.add_argument(
        "--mode",
        choices=("pose", "motion", "all"),
        default="all",
        help=(
            "Asset group to validate. Default 'all' checks the full release "
            "layout including released data and optional assets."
        ),
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Also check released data for the selected non-'all' workflow.",
    )
    return parser.parse_args()


def select_assets(mode: str, check_data: bool) -> list[AssetSpec]:
    """Return the asset checks required for the selected workflow."""
    assets = list(SHARED_ASSETS)
    if mode in {"pose", "motion", "all"}:
        assets.extend(EXACT_FCL_ASSETS)
    if mode in {"motion", "all"}:
        assets.extend(MOTION_ASSETS)

    include_data = check_data or mode == "all"
    if include_data:
        if mode in {"pose", "all"}:
            assets.extend(POSE_DATA_ASSETS)
        if mode in {"motion", "all"}:
            assets.extend(MOTION_DATA_ASSETS)
    if mode == "all":
        assets.extend(OPTIONAL_ASSETS)
    return assets


def format_size(path: Path) -> str:
    """Return a compact size string for an existing path."""
    if path.is_dir():
        return "directory"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def has_glob_pattern(path: Path) -> bool:
    """Return whether a relative path contains glob wildcard syntax."""
    return any(char in str(path) for char in "*?[")


def find_existing_paths(root: Path, spec: AssetSpec) -> list[Path]:
    """Return existing paths matching an asset spec."""
    existing: list[Path] = []
    for rel_path in spec.paths:
        if has_glob_pattern(rel_path):
            existing.extend(sorted(root.glob(str(rel_path))))
        else:
            path = root / rel_path
            if path.exists():
                existing.append(path)
    return existing


def check_asset(root: Path, spec: AssetSpec) -> tuple[bool, str]:
    """Validate one asset and return a status flag plus printable message."""
    existing = find_existing_paths(root, spec)
    if not existing:
        expected = " or ".join(str(path) for path in spec.paths)
        return False, f"MISSING {spec.label}: expected {expected} ({spec.workflow})"
    path = existing[0]
    if path.is_file() and path.stat().st_size == 0:
        return False, f"EMPTY {spec.label}: {path.relative_to(root)} ({spec.workflow})"
    if spec.validator == "public_motion_format":
        ok, detail = validate_public_motion_files(root, existing)
        if not ok:
            return False, f"INVALID {spec.label}: {detail} ({spec.workflow})"
    suffix = f"{len(existing)} match(es)" if len(existing) > 1 else format_size(path)
    return True, f"{spec.label}: {path.relative_to(root)} [{suffix}]"


def validate_public_motion_files(root: Path, paths: list[Path]) -> tuple[bool, str]:
    """Validate that motion files are readable canonical motion arrays."""
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT))
    from poseshield.hymotion.utils.motion_format import validate_motion_array

    failures: list[str] = []
    for path in paths:
        try:
            motion = np.load(path)
            validate_motion_array(motion, name=str(path))
            if not np.isfinite(motion).all():
                failures.append(f"{path.relative_to(root)} contains non-finite values")
        except Exception as error:  # pragma: no cover - defensive asset check
            failures.append(f"{path.relative_to(root)}: {error}")
    if failures:
        preview = "; ".join(failures[:3])
        if len(failures) > 3:
            preview += f"; ... {len(failures) - 3} more"
        return False, preview
    return True, f"{len(paths)} canonical motion file(s)"


def print_group(title: str, messages: list[str]) -> None:
    """Print a named group of asset check messages."""
    print(f"\n{title}")
    if not messages:
        print("  - None")
        return
    for message in messages:
        print(f"  - {message}")


def main() -> int:
    """Run asset checks and return a process exit code."""
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR repository root does not exist: {root}", file=sys.stderr)
        return 2

    present: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []

    print(f"PoseShield asset check\nRoot: {root}")
    for spec in select_assets(args.mode, args.check_data):
        ok, message = check_asset(root, spec)
        if ok:
            present.append(message)
        elif spec.required:
            missing_required.append(message)
        else:
            missing_optional.append(message)

    print_group("Present", present)
    print_group("Missing required", missing_required)
    print_group("Missing optional", missing_optional)

    if missing_required:
        print(f"\nFAILED: {len(missing_required)} required asset group(s) missing.")
        return 1

    print("\nAll required PoseShield assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
