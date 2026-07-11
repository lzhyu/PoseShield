"""Check compact topology cache equivalence against the legacy pickle dict."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from poseshield.common.collision import is_topologically_far, load_topology_distances, topological_distance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=PROJECT_ROOT / "deps/distances.pkl")
    parser.add_argument("--compact", type=Path, default=PROJECT_ROOT / "deps/topology_distances_30_60.npz")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[30, 40, 50, 60])
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--random-pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    with args.legacy.open("rb") as handle:
        legacy = pickle.load(handle)
    compact = load_topology_distances(args.compact)

    keys = list(legacy.keys())
    sampled_keys = [keys[rng.randrange(len(keys))] for _ in range(min(args.samples, len(keys)))]
    num_faces = len(compact.indptr) - 1
    random_pairs = [
        (rng.randrange(num_faces), rng.randrange(num_faces))
        for _ in range(args.random_pairs)
    ]
    pairs = sampled_keys + random_pairs

    mismatches = []
    distance_mismatches = []
    for face1, face2 in pairs:
        legacy_distance = topological_distance(legacy, face1, face2)
        compact_distance = topological_distance(compact, face1, face2)
        for threshold in args.thresholds:
            legacy_far = is_topologically_far(legacy, face1, face2, threshold)
            compact_far = is_topologically_far(compact, face1, face2, threshold)
            if legacy_far != compact_far:
                mismatches.append(
                    {
                        "face1": int(face1),
                        "face2": int(face2),
                        "threshold": int(threshold),
                        "legacy_distance": int(legacy_distance),
                        "compact_distance": int(compact_distance),
                        "legacy_far": bool(legacy_far),
                        "compact_far": bool(compact_far),
                    }
                )
                break
        if legacy_distance != compact_distance:
            if legacy_distance < compact.min_distance and compact_distance == compact.near_value:
                continue
            if legacy_distance > compact.max_distance and compact_distance == -1:
                continue
            if legacy_distance == -1 and compact_distance == -1:
                continue
            distance_mismatches.append(
                {
                    "face1": int(face1),
                    "face2": int(face2),
                    "legacy_distance": int(legacy_distance),
                    "compact_distance": int(compact_distance),
                }
            )

    result = {
        "legacy": str(args.legacy),
        "compact": str(args.compact),
        "thresholds": [int(t) for t in args.thresholds],
        "num_pairs_checked": int(len(pairs)),
        "num_threshold_mismatches": int(len(mismatches)),
        "num_distance_mismatches_outside_expected_banding": int(len(distance_mismatches)),
        "first_threshold_mismatches": mismatches[:10],
        "first_distance_mismatches": distance_mismatches[:10],
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 1 if mismatches or distance_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
