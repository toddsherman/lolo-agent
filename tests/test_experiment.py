import json
import tempfile
import unittest
from pathlib import Path

import torch

from lolo_agent.bootstrap import BootstrapFixture, BootstrapStep
from lolo_agent.ensemble_world_model import (
    EnsembleVisualDynamicsModel,
    save_ensemble_checkpoint,
)
from lolo_agent.environment import Action
from lolo_agent.experiment import DurableExperiment, ExperimentConfig


class DurableExperimentTests(unittest.TestCase):
    def test_manifest_freezes_bootstrap_and_initial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / "host"
            core = root / "core"
            rom = root / "game.nes"
            host.write_bytes(b"host")
            core.write_bytes(b"core")
            rom.write_bytes(b"rom")

            model = EnsembleVisualDynamicsModel(
                latent_size=16,
                action_size=8,
                ensemble_size=2,
                duration_conditioned=True,
            )
            checkpoint = root / "initial.pt"
            expected_digest = save_ensemble_checkpoint(model, checkpoint, 1)
            fixture = BootstrapFixture(
                name="test-room",
                steps=(BootstrapStep(Action.START, 1),),
            )
            config = ExperimentConfig(
                roots_per_cycle=2,
                branches_per_root=1,
                horizon=1,
                action_durations=(1,),
                evaluation_decisions=1,
                verify_actions=1,
                validation_modulus=2,
                latent_size=16,
                action_size=8,
                ensemble_size=2,
            )
            experiment_dir = root / "experiment"
            experiment = DurableExperiment(
                experiment_dir,
                host,
                core,
                rom,
                config,
                bootstrap_fixture=fixture,
                initial_checkpoint=checkpoint,
            )

            manifest = json.loads((experiment_dir / "experiment.json").read_text())
            self.assertEqual(manifest["bootstrap"]["fixture"], "test-room")
            self.assertEqual(manifest["initial_checkpoint"]["name"], "initial.pt")
            loaded = experiment._load_model(torch.device("cpu"))
            self.assertEqual(loaded.checkpoint_digest, expected_digest)

            DurableExperiment(
                experiment_dir,
                host,
                core,
                rom,
                config,
                bootstrap_fixture=fixture,
                initial_checkpoint=checkpoint,
            )

            legacy_manifest = json.loads((experiment_dir / "experiment.json").read_text())
            legacy_manifest["config"].pop("learning_rate")
            (experiment_dir / "experiment.json").write_text(json.dumps(legacy_manifest))
            DurableExperiment(
                experiment_dir,
                host,
                core,
                rom,
                config,
                bootstrap_fixture=fixture,
                initial_checkpoint=checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "resume configuration"):
                DurableExperiment(
                    experiment_dir,
                    host,
                    core,
                    rom,
                    config,
                    initial_checkpoint=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
