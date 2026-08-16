from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .bootstrap import (
    BOOTSTRAP_FIXTURES,
    BootstrapFixture,
    apply_bootstrap_fixture,
    bootstrap_metadata,
    get_bootstrap_fixture,
    validate_bootstrap_rom,
)
from .ensemble_world_model import (
    EnsembleVisualDynamicsModel,
    collect_branched_sequences,
    load_ensemble_checkpoint,
    save_ensemble_checkpoint,
    split_sequence_groups,
    train_ensemble_model,
    validate_ensemble_model,
)
from .log_summary import build_run_summary
from .native_env import NativeLibretroEnv
from .neural_planner import NeuralPlanningConfig, VerifiedNeuralAgent
from .neural_world_model import ACTION_ORDER, choose_torch_device
from .run_logging import LoggedEnvironment, RunLogger, sha256_file, utc_now
from .sequence_store import SequenceStore


@dataclass(frozen=True)
class ExperimentConfig:
    roots_per_cycle: int = 20
    branches_per_root: int = 3
    horizon: int = 3
    action_durations: Tuple[int, ...] = (1, 2, 4, 8, 16)
    epochs_per_cycle: int = 1
    batch_size: int = 8
    learning_rate: float = 3e-4
    evaluation_decisions: int = 20
    verify_actions: int = 6
    validation_modulus: int = 5
    latent_size: int = 256
    action_size: int = 32
    ensemble_size: int = 3
    seed: int = 7

    def validate(self) -> None:
        positive = (
            self.roots_per_cycle,
            self.branches_per_root,
            self.horizon,
            self.epochs_per_cycle,
            self.batch_size,
            self.evaluation_decisions,
            self.verify_actions,
            self.latent_size,
            self.action_size,
            self.ensemble_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("experiment sizes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if len(set(self.action_durations)) != len(self.action_durations):
            raise ValueError("action durations must be unique")
        if not self.action_durations or any(value <= 0 for value in self.action_durations):
            raise ValueError("action durations must be positive")
        if self.validation_modulus < 2:
            raise ValueError("validation modulus must be at least two")


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _input_manifest(host: Path, core: Path, rom: Path) -> Dict[str, Any]:
    return {
        "host": {"name": host.name, "sha256": sha256_file(host)},
        "core": {"name": core.name, "sha256": sha256_file(core)},
        "rom": {"name": rom.name, "sha256": sha256_file(rom)},
    }


def _validation_dict(report: Any) -> Dict[str, Any]:
    return {
        "horizon_pixel_l1": list(report.horizon_pixel_l1),
        "horizon_uncertainty": list(report.horizon_uncertainty),
        "uncertainty_error_correlation": report.uncertainty_error_correlation,
    }


class DurableExperiment:
    def __init__(
        self,
        experiment_dir: Path,
        host: Path,
        core: Path,
        rom: Path,
        config: ExperimentConfig,
        store_collection_frames: bool = True,
        bootstrap_fixture: Optional[BootstrapFixture] = None,
        initial_checkpoint: Optional[Path] = None,
    ) -> None:
        config.validate()
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.host = Path(host).expanduser().resolve()
        self.core = Path(core).expanduser().resolve()
        self.rom = Path(rom).expanduser().resolve()
        self.config = config
        self.store_collection_frames = store_collection_frames
        self.bootstrap_fixture = bootstrap_fixture
        self.initial_checkpoint = (
            None
            if initial_checkpoint is None
            else Path(initial_checkpoint).expanduser().resolve()
        )
        self.manifest_path = self.experiment_dir / "experiment.json"
        self.state_path = self.experiment_dir / "state.json"
        self.dataset = SequenceStore(self.experiment_dir / "dataset")
        self.inputs = _input_manifest(self.host, self.core, self.rom)
        if self.bootstrap_fixture is not None:
            validate_bootstrap_rom(
                self.bootstrap_fixture, self.inputs["rom"]["sha256"]
            )
        initial_checkpoint_metadata = (
            None
            if self.initial_checkpoint is None
            else {
                "name": self.initial_checkpoint.name,
                "sha256": sha256_file(self.initial_checkpoint),
            }
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        requested = {
            "version": 1,
            "config": asdict(config),
            "inputs": self.inputs,
            "bootstrap": bootstrap_metadata(self.bootstrap_fixture),
            "initial_checkpoint": initial_checkpoint_metadata,
        }
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            existing_config = dict(existing["config"])
            # Version-1 manifests created before learning-rate configuration
            # implicitly used train_ensemble_model's 3e-4 default.
            existing_config.setdefault("learning_rate", 3e-4)
            comparable = {
                "version": existing["version"],
                "config": existing_config,
                "inputs": existing["inputs"],
                "bootstrap": existing.get("bootstrap"),
                "initial_checkpoint": existing.get("initial_checkpoint"),
            }
            if comparable != json.loads(json.dumps(requested)):
                raise ValueError("resume configuration or input hashes do not match experiment")
        else:
            _atomic_json(
                self.manifest_path,
                {
                    **requested,
                    "created_at": utc_now(),
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                },
            )
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "version": 1,
                "phase": "idle",
                "completed_cycles": 0,
                "current_cycle": None,
                "next_group": 0,
                "checkpoint": None,
                "evaluation_attempt": 0,
                "updated_at": utc_now(),
            }
            self._save_state()

    def _save_state(self, **changes: Any) -> None:
        self.state.update(changes)
        self.state["updated_at"] = utc_now()
        _atomic_json(self.state_path, self.state)

    def _load_model(self, device: torch.device) -> EnsembleVisualDynamicsModel:
        checkpoint = self.state.get("checkpoint")
        if checkpoint:
            model, horizon = load_ensemble_checkpoint(
                self.experiment_dir / checkpoint, device=device, frozen=False
            )
        elif self.initial_checkpoint is not None:
            model, horizon = load_ensemble_checkpoint(
                self.initial_checkpoint, device=device, frozen=False
            )
        else:
            return EnsembleVisualDynamicsModel(
                latent_size=self.config.latent_size,
                action_size=self.config.action_size,
                ensemble_size=self.config.ensemble_size,
                duration_conditioned=True,
                max_action_frames=max(32, max(self.config.action_durations)),
            ).to(device)
        if horizon != self.config.horizon:
            raise ValueError("checkpoint planning horizon does not match experiment")
        if not model.duration_conditioned:
            raise ValueError("durable experiment checkpoint lacks duration conditioning")
        architecture = (model.latent_size, model.action_size, model.ensemble_size)
        expected = (
            self.config.latent_size,
            self.config.action_size,
            self.config.ensemble_size,
        )
        if architecture != expected:
            raise ValueError("checkpoint architecture does not match experiment")
        return model

    def _train_cycle(
        self, cycle: int, model: EnsembleVisualDynamicsModel, device: torch.device
    ) -> Dict[str, Any]:
        sequences = self.dataset.load()
        training, validation = split_sequence_groups(
            sequences, self.config.validation_modulus
        )
        before = validate_ensemble_model(
            model, validation, device, self.config.batch_size
        )
        started = time.monotonic()
        history = train_ensemble_model(
            model,
            training,
            device,
            epochs=self.config.epochs_per_cycle,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            seed=self.config.seed + cycle,
        )
        after = validate_ensemble_model(
            model, validation, device, self.config.batch_size
        )
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        destination = checkpoint_dir / f"cycle-{cycle:06d}.pt"
        temporary = checkpoint_dir / f".cycle-{cycle:06d}.tmp"
        digest = save_ensemble_checkpoint(model, temporary, self.config.horizon)
        os.replace(temporary, destination)
        metrics = {
            "cycle": cycle,
            "completed_at": utc_now(),
            "training_sequences": len(training),
            "validation_sequences": len(validation),
            "updates": len(history),
            "learning_rate": self.config.learning_rate,
            "first_loss": history[0].loss,
            "final_loss": history[-1].loss,
            "duration_seconds": time.monotonic() - started,
            "validation_before": _validation_dict(before),
            "validation_after": _validation_dict(after),
            "checkpoint": str(destination.relative_to(self.experiment_dir)),
            "checkpoint_parameter_sha256": digest,
            "dataset": self.dataset.statistics(),
        }
        _atomic_json(self.experiment_dir / "metrics" / f"cycle-{cycle:06d}.json", metrics)
        self._save_state(
            phase="trained",
            checkpoint=str(destination.relative_to(self.experiment_dir)),
            checkpoint_parameter_sha256=digest,
        )
        return metrics

    def _evaluate_cycle(
        self, cycle: int, model: EnsembleVisualDynamicsModel, device: torch.device
    ) -> Dict[str, Any]:
        attempt = int(self.state.get("evaluation_attempt", 0)) + 1
        self._save_state(evaluation_attempt=attempt)
        checkpoint_path = self.experiment_dir / self.state["checkpoint"]
        run_id = f"cycle-{cycle:06d}-attempt-{attempt:03d}"
        logger = RunLogger(
            self.experiment_dir / "evaluations",
            run_id=run_id,
            metadata={
                "mode": "frozen_cycle_evaluation",
                "cycle": cycle,
                "experiment": self.experiment_dir.name,
                "inputs": self.inputs,
                "checkpoint": {
                    "name": checkpoint_path.name,
                    "file_sha256": sha256_file(checkpoint_path),
                    "parameter_sha256": model.checkpoint_digest,
                },
                "planning_config": {
                    "actions": ACTION_ORDER,
                    "durations": self.config.action_durations,
                    "horizon": self.config.horizon,
                },
                "bootstrap": bootstrap_metadata(self.bootstrap_fixture),
            },
        )
        agent: Optional[VerifiedNeuralAgent] = None
        before = model.checkpoint_digest
        try:
            with NativeLibretroEnv(self.host, self.core, self.rom) as native:
                env = LoggedEnvironment(native, logger)
                agent = VerifiedNeuralAgent(
                    env,
                    model,
                    device,
                    NeuralPlanningConfig(
                        actions=ACTION_ORDER,
                        planning_depth=self.config.horizon,
                        verify_actions=self.config.verify_actions,
                        action_durations=self.config.action_durations,
                    ),
                    event_logger=logger,
                )
                if self.bootstrap_fixture is None:
                    agent.reset()
                else:
                    initial_frame = apply_bootstrap_fixture(
                        env,
                        self.bootstrap_fixture,
                        self.inputs["rom"]["sha256"],
                    )
                    agent.reset(initial_frame=initial_frame)
                decisions = agent.run(self.config.evaluation_decisions)
                agent.clear_archive()
            after = model.checkpoint_digest
            if before != after:
                raise RuntimeError("frozen cycle evaluation changed model parameters")
            logger.log(
                "frozen_parameter_audit",
                status="pass",
                parameter_sha256_before=before,
                parameter_sha256_after=after,
            )
            logger.close("complete")
        except Exception as exc:
            if agent is not None:
                agent.clear_archive()
            logger.close("error", str(exc))
            build_run_summary(logger.run_dir)
            raise
        summary = build_run_summary(logger.run_dir)
        return {
            "run": str(logger.run_dir.relative_to(self.experiment_dir)),
            "decisions": len(decisions),
            "actions": summary["committed_actions"],
            "durations": summary["committed_durations"],
            "unique_frames": summary["unique_frames"],
            "unique_scenes": summary["unique_scenes"],
            "checkpoint_unchanged": True,
        }

    def run(self, cycles: int) -> Dict[str, Any]:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        device = choose_torch_device()
        torch.manual_seed(self.config.seed + int(self.state["completed_cycles"]))
        model = self._load_model(device)
        target_cycles = int(self.state["completed_cycles"]) + cycles
        collector_initialized = False
        collection_logger = RunLogger(
            self.experiment_dir / "collection_runs",
            metadata={
                "mode": "duration_aware_branched_collection",
                "experiment": self.experiment_dir.name,
                "inputs": self.inputs,
                "durations": self.config.action_durations,
                "bootstrap": bootstrap_metadata(self.bootstrap_fixture),
            },
            store_frames=self.store_collection_frames,
        )
        try:
            with NativeLibretroEnv(self.host, self.core, self.rom) as native:
                collection_env = LoggedEnvironment(native, collection_logger)
                while int(self.state["completed_cycles"]) < target_cycles:
                    phase = self.state["phase"]
                    cycle = (
                        int(self.state["current_cycle"])
                        if phase != "idle" and self.state.get("current_cycle") is not None
                        else int(self.state["completed_cycles"]) + 1
                    )
                    segment_id = f"cycle-{cycle:06d}"
                    if phase in ("idle", "collecting"):
                        self._save_state(phase="collecting", current_cycle=cycle, error=None)
                        if not self.dataset.has_segment(segment_id):
                            collection_logger.log("collection_cycle_started", cycle=cycle)
                            if self.bootstrap_fixture is not None:
                                apply_bootstrap_fixture(
                                    collection_env,
                                    self.bootstrap_fixture,
                                    self.inputs["rom"]["sha256"],
                                )
                                collector_initialized = True
                            sequences = collect_branched_sequences(
                                collection_env,
                                roots=self.config.roots_per_cycle,
                                branches_per_root=self.config.branches_per_root,
                                horizon=self.config.horizon,
                                action_frames=self.config.action_durations[0],
                                action_durations=self.config.action_durations,
                                seed=self.config.seed + cycle,
                                reset_env=not collector_initialized,
                                group_offset=int(self.state["next_group"]),
                                event_logger=collection_logger,
                                source_run_id=collection_logger.run_id,
                            )
                            collector_initialized = True
                            self.dataset.append_segment(segment_id, sequences)
                            self._save_state(
                                next_group=int(self.state["next_group"])
                                + self.config.roots_per_cycle
                            )
                            collection_logger.log(
                                "collection_cycle_finished",
                                cycle=cycle,
                                sequences=len(sequences),
                                dataset=self.dataset.statistics(),
                            )
                        else:
                            persisted = self.dataset.load()
                            recovered_next_group = max(
                                (sequence.group for sequence in persisted), default=-1
                            ) + 1
                            if recovered_next_group > int(self.state["next_group"]):
                                self._save_state(next_group=recovered_next_group)
                            collection_logger.log(
                                "collection_segment_recovered",
                                cycle=cycle,
                                segment=segment_id,
                            )
                        self._save_state(phase="collected")
                        phase = "collected"
                    if phase == "collected":
                        self._train_cycle(cycle, model, device)
                        phase = "trained"
                    if phase == "trained":
                        evaluation = self._evaluate_cycle(cycle, model, device)
                        metrics_path = self.experiment_dir / "metrics" / f"cycle-{cycle:06d}.json"
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        metrics["evaluation"] = evaluation
                        _atomic_json(metrics_path, metrics)
                        self._save_state(
                            phase="idle",
                            completed_cycles=cycle,
                            current_cycle=None,
                            last_evaluation=evaluation,
                        )
            collection_logger.close("complete")
            build_run_summary(collection_logger.run_dir)
        except BaseException as exc:
            self._save_state(error=f"{type(exc).__name__}: {exc}")
            collection_logger.close("interrupted" if isinstance(exc, KeyboardInterrupt) else "error", str(exc))
            build_run_summary(collection_logger.run_dir)
            raise
        return {
            "experiment": str(self.experiment_dir),
            "device": str(device),
            "completed_cycles": self.state["completed_cycles"],
            "checkpoint": self.state["checkpoint"],
            "dataset": self.dataset.statistics(),
            "last_evaluation": self.state.get("last_evaluation"),
        }


def _durations(value: str) -> Tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("durations must be comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a durable collect-train-evaluate experiment")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=1, help="additional cycles to complete")
    parser.add_argument("--roots", type=int, default=20)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--durations", type=_durations, default=(1, 2, 4, 8, 16))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-decisions", type=int, default=20)
    parser.add_argument("--verify-actions", type=int, default=6)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--action-size", type=int, default=32)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-collection-frame-images", action="store_true")
    parser.add_argument(
        "--bootstrap",
        choices=("none", *sorted(BOOTSTRAP_FIXTURES)),
        default="none",
        help="evaluator-owned initialization applied before collection and evaluation",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="optional compatible checkpoint used before the first training cycle",
    )
    args = parser.parse_args()
    config = ExperimentConfig(
        roots_per_cycle=args.roots,
        branches_per_root=args.branches,
        horizon=args.horizon,
        action_durations=args.durations,
        epochs_per_cycle=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_decisions=args.eval_decisions,
        verify_actions=args.verify_actions,
        validation_modulus=args.validation_modulus,
        latent_size=args.latent_size,
        action_size=args.action_size,
        ensemble_size=args.ensemble_size,
        seed=args.seed,
    )
    experiment = DurableExperiment(
        args.experiment_dir,
        args.host,
        args.core,
        args.rom,
        config,
        store_collection_frames=not args.no_collection_frame_images,
        bootstrap_fixture=(
            None if args.bootstrap == "none" else get_bootstrap_fixture(args.bootstrap)
        ),
        initial_checkpoint=args.initial_checkpoint,
    )
    print(json.dumps(experiment.run(args.cycles), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
