import tempfile
import unittest

try:  # optional ML extra (imported transitively by the modules below)
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc
from pathlib import Path

from lolo_agent.environment import Action
from lolo_agent.pixels import Frame
from lolo_agent.probe_returnability_import import (
    balanced_probe_sample,
    extract_probe_returnability,
    load_probe_returnability_corpus,
    probe_validation_sample,
)
from lolo_agent.run_logging import RunLogger


class ProbeReturnabilityImportTests(unittest.TestCase):
    @staticmethod
    def _frame(value: int) -> Frame:
        return Frame(2, 1, 1, bytes((value, value + 1)))

    def _run(
        self,
        root: Path,
        run_id: str,
        records,
        reward_track="strict_rule_free",
        restart_attempts=False,
    ):
        actions = ["left", "right", "noop"]
        logger = RunLogger(
            root,
            run_id=run_id,
            metadata={
                "reward_track": reward_track,
                "planning_config": {
                    "returnability_probe_depth": 2,
                    "returnability_probe_beam_width": 4,
                    "returnability_probe_pixel_l1_threshold": 0.002,
                    "actions": actions,
                },
            },
        )
        for index, (source, target, action, returned) in enumerate(records, 1):
            if restart_attempts:
                logger.start_attempt("test_restart")
            branch_id = (
                "decision-00000001-branch-01"
                if restart_attempts
                else f"decision-{index:08d}-branch-01"
            )
            common = {
                "decision": index,
                "branch_id": branch_id,
                "candidate_rank": 1,
                "initial_action": action,
                "initial_action_frames": 4,
                "maximum_depth": 2,
                "beam_width": 4,
                "pixel_l1_threshold": 0.002,
                "actions": actions,
                "source_frame": logger.store_frame(source),
                "endpoint_frame": logger.store_frame(target),
            }
            logger.log("bidirectional_probe_started", **common)
            logger.log(
                "bidirectional_probe_completed",
                **common,
                paths_evaluated=3 if returned else 15,
                returning_paths=1 if returned else 0,
                return_observed=returned,
                no_return_within_probe_budget=not returned,
                shortest_return_depth=1 if returned else None,
                best_matched_noop_l1=0.0 if returned else 0.01,
            )
            logger.log(
                "branch_verified",
                branch_id=branch_id,
                action=action,
                action_frames=4,
                frame=target.digest,
            )
        logger.close()
        return logger.run_dir

    def test_strict_import_validates_and_removes_partition_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlap = (
                self._frame(1),
                self._frame(3),
                Action.RIGHT,
                True,
            )
            training = self._run(
                root,
                "train",
                [
                    overlap,
                    (self._frame(5), self._frame(7), Action.LEFT, False),
                    (self._frame(9), self._frame(11), Action.RIGHT, True),
                ],
            )
            validation = self._run(
                root,
                "validation",
                [
                    overlap,
                    (
                        overlap[0],
                        self._frame(21),
                        Action.NOOP,
                        True,
                    ),
                    (self._frame(13), self._frame(15), Action.LEFT, False),
                    (self._frame(17), self._frame(19), Action.RIGHT, True),
                ],
            )

            corpus = load_probe_returnability_corpus([training], [validation])

            self.assertEqual(
                corpus.metadata["validation_transition_overlap_removed"], 1
            )
            self.assertEqual(corpus.metadata["validation_source_overlap_removed"], 1)
            self.assertEqual(corpus.metadata["validation_overlap_removed"], 2)
            self.assertEqual(len(corpus.training), 3)
            self.assertEqual(len(corpus.validation), 2)
            self.assertEqual({item.label for item in corpus.training}, {0, 1})
            self.assertEqual({item.label for item in corpus.validation}, {0, 1})
            self.assertTrue(all(item.target is not None for item in corpus.training))
            sample = balanced_probe_sample(corpus.training, 2, seed=3)
            self.assertEqual({item.label for item in sample}, {0, 1})
            validation_sample = probe_validation_sample(
                corpus.validation, 10, seed=3
            )
            self.assertEqual(len(validation_sample), 2)
            self.assertEqual({item.label for item in validation_sample}, {0, 1})

    def test_assisted_probes_cannot_enter_strict_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(
                Path(directory),
                "assisted",
                [
                    (
                        self._frame(21),
                        self._frame(23),
                        Action.RIGHT,
                        True,
                    )
                ],
                reward_track="human_prior_resume_observational",
            )

            with self.assertRaisesRegex(ValueError, "assisted.*strict"):
                extract_probe_returnability(run, required_reward_track="strict")

    def test_branch_ids_may_repeat_after_attempt_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(
                Path(directory),
                "restarted",
                [
                    (self._frame(31), self._frame(33), Action.RIGHT, True),
                    (self._frame(35), self._frame(37), Action.LEFT, False),
                ],
                restart_attempts=True,
            )

            examples, metadata = extract_probe_returnability(run)

            self.assertEqual(len(examples), 2)
            self.assertEqual(metadata["positives"], 1)
            self.assertEqual(metadata["negatives"], 1)


if __name__ == "__main__":
    unittest.main()
