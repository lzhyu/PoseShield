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
        (Path("deps/distances.pkl"),),
        "exact mesh self-collision validation",
    ),
)

DATA_ASSETS = (
    AssetSpec(
        "Pose training split list",
        (Path("data/dataset/train_list.csv"),),
        "pose training",
    ),
    AssetSpec(
        "Pose test split list",
        (Path("data/dataset/test_list.csv"),),
        "pose evaluation",
    ),
    AssetSpec(
        "Pose benchmark dataset",
        (Path("data/dataset_test"),),
        "pose benchmark evaluation",
    ),
    AssetSpec(
        "Canonical motion subset",
        (Path("data/motion_canonical"),),
        "motion evaluation",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the namespace."""
    parser = argparse.ArgumentParser(
        description="Check required PoseShield checkpoints, body models, and data assets."
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
        help="Asset group to validate. 'motion' includes pose-level shared assets.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Also check released training/evaluation data directories.",
    )
    return parser.parse_args()


def select_assets(mode: str, check_data: bool) -> list[AssetSpec]:
    """Return the asset checks required for the selected workflow."""
    assets = list(SHARED_ASSETS)
    if mode in {"pose", "motion", "all"}:
        assets.extend(EXACT_FCL_ASSETS)
    if mode in {"motion", "all"}:
        assets.extend(MOTION_ASSETS)
    if check_data:
        assets.extend(DATA_ASSETS)
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


def check_asset(root: Path, spec: AssetSpec) -> tuple[bool, str]:
    """Validate one asset and return a status flag plus printable message."""
    candidates = [root / path for path in spec.paths]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        expected = " or ".join(str(path.relative_to(root)) for path in candidates)
        return False, f"MISSING {spec.label}: expected {expected} ({spec.workflow})"
    path = existing[0]
    if path.is_file() and path.stat().st_size == 0:
        return False, f"EMPTY {spec.label}: {path.relative_to(root)} ({spec.workflow})"
    return True, f"OK {spec.label}: {path.relative_to(root)} [{format_size(path)}]"


def main() -> int:
    """Run asset checks and return a process exit code."""
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR repository root does not exist: {root}", file=sys.stderr)
        return 2

    failures = []
    for spec in select_assets(args.mode, args.check_data):
        ok, message = check_asset(root, spec)
        print(message)
        if not ok:
            failures.append(message)

    if failures:
        print(f"\nFAILED {len(failures)} asset check(s).")
        return 1
    print("\nAll requested PoseShield asset checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
