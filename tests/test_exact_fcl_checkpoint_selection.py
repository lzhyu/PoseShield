import json
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from poseshield.hymotion.dno.run_dno_stage2 import (
    get_args,
    _run_exact_fcl_for_checkpoint,
    _select_checkpoint_by_exact_fcl,
)


class ExactFCLCheckpointSelectionTest(unittest.TestCase):
    def test_release_stage2_defaults_match_exact_aware_preset(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_dno_stage2.py",
                "--motion_file",
                "motion.npy",
                "--stage1_z",
                "stage1_z.pt",
            ],
        ):
            args = get_args()

        self.assertEqual(args.ode_steps, 20)
        self.assertEqual(args.s2_steps, 100)
        self.assertEqual(args.save_checkpoint_motions_every, 10)
        self.assertTrue(args.exact_fcl_select_checkpoint)
        self.assertEqual(args.exact_fcl_selection_distances, "deps/topology_distances_30_60.npz")
        self.assertEqual(args.exact_fcl_selection_proxy_col_threshold, 1e-3)

    def test_exact_fcl_checkpoint_selection_can_be_disabled(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_dno_stage2.py",
                "--motion_file",
                "motion.npy",
                "--stage1_z",
                "stage1_z.pt",
                "--no_exact_fcl_select_checkpoint",
            ],
        ):
            args = get_args()

        self.assertFalse(args.exact_fcl_select_checkpoint)

    def _write_metadata(self, root: Path, records: list[dict]) -> Path:
        metadata_path = root / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return metadata_path

    def _record(self, step: int, score: float, col: Optional[float] = None) -> dict:
        return {
            "step": step,
            "motion": f"step_{step:04d}.npy",
            "x": f"step_{step:04d}_x.pt",
            "z": f"step_{step:04d}_z.pt",
            "checkpoint_score": score,
            "col": col,
        }

    def test_selects_lowest_score_exact_free_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = self._write_metadata(
                root,
                [self._record(10, 0.2), self._record(20, 0.1)],
            )

            def fake_exact(**kwargs):
                step = int(kwargs["motion_path"].stem.split("_")[1])
                return {
                    "returncode": 0,
                    "log": str(root / f"step_{step:04d}.log"),
                    "result_path": str(root / f"step_{step:04d}.json"),
                    "results": {
                        "exact_collision_free": True,
                        "num_collision_frames": 0,
                        "mean_penetration_depth": 0.0,
                    },
                    "reused": False,
                }

            with patch(
                "poseshield.hymotion.dno.run_dno_stage2._run_exact_fcl_for_checkpoint",
                side_effect=fake_exact,
            ):
                summary = _select_checkpoint_by_exact_fcl(
                    metadata_path=metadata_path,
                    output_dir=root / "selection",
                    distances_path=root / "distances.pkl",
                    device="cpu",
                    topology_threshold=40,
                )

            self.assertEqual(summary["selection_reason"], "exact_collision_free_lowest_checkpoint_score")
            self.assertEqual(summary["selected_step"], 20)
            self.assertEqual(summary["num_reused_exact_fcl_results"], 0)

    def test_falls_back_to_lowest_exact_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = self._write_metadata(
                root,
                [self._record(10, 0.1), self._record(20, 0.2)],
            )

            means = {10: 0.5, 20: 0.1}

            def fake_exact(**kwargs):
                step = int(kwargs["motion_path"].stem.split("_")[1])
                return {
                    "returncode": 1,
                    "log": str(root / f"step_{step:04d}.log"),
                    "result_path": str(root / f"step_{step:04d}.json"),
                    "results": {
                        "exact_collision_free": False,
                        "num_collision_frames": step,
                        "mean_penetration_depth": means[step],
                    },
                    "reused": False,
                }

            with patch(
                "poseshield.hymotion.dno.run_dno_stage2._run_exact_fcl_for_checkpoint",
                side_effect=fake_exact,
            ):
                summary = _select_checkpoint_by_exact_fcl(
                    metadata_path=metadata_path,
                    output_dir=root / "selection",
                    distances_path=root / "distances.pkl",
                    device="cpu",
                    topology_threshold=40,
                )

            self.assertEqual(summary["selection_reason"], "minimum_exact_fcl_mean_penetration")
            self.assertEqual(summary["selected_step"], 20)

    def test_filters_exact_fcl_candidates_by_proxy_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = self._write_metadata(
                root,
                [
                    self._record(10, 0.1, col=0.2),
                    self._record(20, 0.3, col=0.0),
                    self._record(30, 0.2, col=5e-5),
                ],
            )
            evaluated_steps = []

            def fake_exact(**kwargs):
                step = int(kwargs["motion_path"].stem.split("_")[1])
                evaluated_steps.append(step)
                return {
                    "returncode": 0,
                    "log": str(root / f"step_{step:04d}.log"),
                    "result_path": str(root / f"step_{step:04d}.json"),
                    "results": {
                        "exact_collision_free": True,
                        "num_collision_frames": 0,
                        "mean_penetration_depth": 0.0,
                    },
                    "reused": False,
                }

            with patch(
                "poseshield.hymotion.dno.run_dno_stage2._run_exact_fcl_for_checkpoint",
                side_effect=fake_exact,
            ):
                summary = _select_checkpoint_by_exact_fcl(
                    metadata_path=metadata_path,
                    output_dir=root / "selection",
                    distances_path=root / "distances.pkl",
                    device="cpu",
                    topology_threshold=40,
                    proxy_col_threshold=1e-4,
                )

            self.assertEqual(evaluated_steps, [20, 30])
            self.assertEqual(summary["candidate_source"], "proxy_collision_threshold")
            self.assertEqual(summary["num_total_checkpoints"], 3)
            self.assertEqual(summary["num_candidate_checkpoints"], 2)
            self.assertEqual(summary["selected_step"], 30)

    def test_resume_reuses_existing_exact_fcl_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "step_0010"
            output_dir.mkdir()
            result_path = output_dir / "exact_fcl_results.json"
            result_path.write_text(
                json.dumps(
                    {
                        "exact_collision_free": True,
                        "num_collision_frames": 0,
                        "mean_penetration_depth": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            with patch("subprocess.run") as run:
                result = _run_exact_fcl_for_checkpoint(
                    motion_path=root / "step_0010.npy",
                    output_dir=output_dir,
                    distances_path=root / "distances.pkl",
                    device="cpu",
                    topology_threshold=40,
                    log_path=root / "step_0010.log",
                    resume=True,
                )

            run.assert_not_called()
            self.assertTrue(result["reused"])
            self.assertTrue(result["results"]["exact_collision_free"])


if __name__ == "__main__":
    unittest.main()
