"""Convert tuple-key topology distances into a compact banded cache.

The compact cache stores exact face distances only inside a configurable band.
Distances below the lower bound are represented by one near marker; missing
pairs are treated as farther than the upper bound. This keeps threshold checks
exact for any threshold in the stored band, e.g. 40 for motion and 50 for pose
when using the default [30, 60] range.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.common.collision import BandedTopologyDistances  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "deps/distances.pkl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "deps/topology_distances_30_60.npz")
    parser.add_argument("--min-distance", type=int, default=30)
    parser.add_argument("--max-distance", type=int, default=60)
    parser.add_argument(
        "--fast-thresholds",
        type=int,
        nargs="*",
        default=[40, 50],
        help="Optional threshold-specific near-pair bitsets for fast exact checks.",
    )
    parser.add_argument("--num-faces", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.input.open("rb") as handle:
        distance_dict = pickle.load(handle)

    compact = BandedTopologyDistances.from_distance_dict(
        distance_dict,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        num_faces=args.num_faces,
    )
    compact.add_fast_thresholds(args.fast_thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compact.save(args.output)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "min_distance": compact.min_distance,
        "max_distance": compact.max_distance,
        "num_faces": int(len(compact.indptr) - 1),
        "num_stored_pairs": int(len(compact.indices)),
        "indptr_bytes": int(compact.indptr.nbytes),
        "indices_bytes": int(compact.indices.nbytes),
        "distances_bytes": int(compact.distances.nbytes),
        "fast_thresholds": [int(t) for t in sorted(compact.fast_bits)],
        "fast_bits_bytes": {str(int(t)): int(compact.fast_bits[int(t)].nbytes) for t in compact.fast_bits},
        "array_bytes_total": int(compact.indptr.nbytes + compact.indices.nbytes + compact.distances.nbytes),
        "array_and_fast_bits_bytes_total": int(
            compact.indptr.nbytes
            + compact.indices.nbytes
            + compact.distances.nbytes
            + sum(bits.nbytes for bits in compact.fast_bits.values())
        ),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
