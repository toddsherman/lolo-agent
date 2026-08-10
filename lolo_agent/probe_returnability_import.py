from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .environment import Action
from .experience_import import classify_reward_track, decode_logged_png
from .pixels import Frame
from .run_logging import read_events, sha256_file
from .spatial_returnability import ReturnabilityExample


@dataclass(frozen=True)
class ProbeReturnabilityCorpus:
    training: Tuple[ReturnabilityExample, ...]
    validation: Tuple[ReturnabilityExample, ...]
    metadata: Dict[str, Any]


def _example_key(example: ReturnabilityExample) -> Tuple[str, str, str, int]:
    return (
        example.source.digest,
        example.target_digest,
        example.action.value,
        example.duration,
    )


def _labels(examples: Iterable[ReturnabilityExample]) -> Counter[int]:
    return Counter(example.label for example in examples)


def extract_probe_returnability(
    run_dir: Path,
    required_reward_track: str = "strict",
) -> Tuple[List[ReturnabilityExample], Dict[str, Any]]:
    """Import explicit matched-NOOP labels from one completed telemetry run."""

    if required_reward_track not in ("strict", "assisted"):
        raise ValueError("reward track must be 'strict' or 'assisted'")
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    events_path = run_dir / "events.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"probe telemetry run is not complete: {run_dir}")
    reward_track = classify_reward_track(manifest)
    if reward_track != required_reward_track:
        raise ValueError(
            f"cannot import {reward_track!r} probes into the "
            f"{required_reward_track!r} corpus: {run_dir.name}"
        )
    if manifest.get("frame_storage") != "content-addressed-png":
        raise ValueError(f"probe telemetry has no stored pixel frames: {run_dir}")

    run_id = str(manifest.get("run_id") or run_dir.name)
    starts: Dict[Tuple[int, str], Dict[str, Any]] = {}
    completed: Dict[Tuple[int, str], Dict[str, Any]] = {}
    verified: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for event in read_events(run_dir):
        kind = event.get("event")
        branch_id = event.get("branch_id")
        branch_key = (int(event.get("attempt", 0)), str(branch_id))
        if kind == "bidirectional_probe_started" and branch_id:
            if branch_key in starts:
                raise ValueError(f"duplicate probe start in {run_id}: {branch_key}")
            starts[branch_key] = event
        elif kind == "bidirectional_probe_completed" and branch_id:
            if branch_key in completed:
                raise ValueError(f"duplicate probe completion in {run_id}: {branch_id}")
            completed[branch_key] = event
        elif kind == "branch_verified" and branch_id:
            if branch_key in verified:
                raise ValueError(f"duplicate verified branch in {run_id}: {branch_id}")
            verified[branch_key] = event
    if not completed:
        raise ValueError(f"no completed bidirectional probes found in {run_dir}")
    if starts.keys() != completed.keys() or completed.keys() != verified.keys():
        raise ValueError(f"probe lifecycle is incomplete in {run_id}")

    planning = manifest.get("metadata", {}).get("planning_config", {})
    expected_configuration = {
        "maximum_depth": int(planning.get("returnability_probe_depth", 0)),
        "beam_width": int(planning.get("returnability_probe_beam_width", 0)),
        "pixel_l1_threshold": float(
            planning.get("returnability_probe_pixel_l1_threshold", -1.0)
        ),
        "actions": tuple(str(action) for action in planning.get("actions", [])),
    }
    if (
        expected_configuration["maximum_depth"] <= 0
        or expected_configuration["beam_width"] <= 0
        or expected_configuration["pixel_l1_threshold"] < 0.0
        or not expected_configuration["actions"]
    ):
        raise ValueError(f"manifest has no valid probe configuration: {run_id}")

    frames: Dict[str, Frame] = {}

    def frame(digest: str) -> Frame:
        if digest not in frames:
            loaded = decode_logged_png(run_dir / "frames" / f"{digest}.png")
            if loaded.digest != digest:
                raise ValueError(f"probe frame digest mismatch in {run_id}: {digest}")
            frames[digest] = loaded
        return frames[digest]

    examples_by_key: Dict[Tuple[str, str, str, int], ReturnabilityExample] = {}
    duplicate_records = 0
    labels = Counter()
    for branch_key in sorted(completed):
        start = starts[branch_key]
        event = completed[branch_key]
        branch = verified[branch_key]
        branch_id = branch_key[1]
        shared_fields = (
            "decision",
            "candidate_rank",
            "initial_action",
            "initial_action_frames",
            "source_frame",
            "endpoint_frame",
            "maximum_depth",
            "beam_width",
            "pixel_l1_threshold",
            "actions",
        )
        if any(start.get(field) != event.get(field) for field in shared_fields):
            raise ValueError(f"probe start/completion mismatch in {run_id}: {branch_id}")
        configuration = {
            "maximum_depth": int(event["maximum_depth"]),
            "beam_width": int(event["beam_width"]),
            "pixel_l1_threshold": float(event["pixel_l1_threshold"]),
            "actions": tuple(str(action) for action in event["actions"]),
        }
        if configuration != expected_configuration:
            raise ValueError(f"probe configuration drift in {run_id}: {branch_id}")
        action = Action(event["initial_action"])
        duration = int(event["initial_action_frames"])
        source_digest = str(event["source_frame"])
        target_digest = str(event["endpoint_frame"])
        if duration <= 0 or int(event.get("paths_evaluated", 0)) <= 0:
            raise ValueError(f"invalid probe coverage in {run_id}: {branch_id}")
        returned = bool(event.get("return_observed"))
        no_return = bool(event.get("no_return_within_probe_budget"))
        returning_paths = int(event.get("returning_paths", 0))
        shortest_depth = event.get("shortest_return_depth")
        if returned == no_return:
            raise ValueError(f"contradictory probe label in {run_id}: {branch_id}")
        if returned != (returning_paths > 0 and shortest_depth is not None):
            raise ValueError(f"inconsistent return evidence in {run_id}: {branch_id}")
        if (
            str(branch.get("action")) != action.value
            or int(branch.get("action_frames", 0)) != duration
            or str(branch.get("frame")) != target_digest
        ):
            raise ValueError(
                f"probe does not match verified branch in {run_id}: {branch_id}"
            )
        source = frame(source_digest)
        target = frame(target_digest)
        label = int(returned)
        example = ReturnabilityExample(
            source,
            target_digest,
            action,
            duration,
            run_id,
            label,
            target,
        )
        key = _example_key(example)
        existing = examples_by_key.get(key)
        if existing is not None:
            if existing.label != label:
                raise ValueError(f"conflicting probe labels in {run_id}: {key}")
            duplicate_records += 1
            continue
        examples_by_key[key] = example
        labels[label] += 1

    examples = list(examples_by_key.values())
    metadata = {
        "run": str(run_dir),
        "run_id": run_id,
        "reward_track": reward_track,
        "manifest_sha256": sha256_file(manifest_path),
        "events_sha256": sha256_file(events_path),
        "probe_configuration": {
            **expected_configuration,
            "actions": list(expected_configuration["actions"]),
        },
        "completed_probes": len(completed),
        "unique_examples": len(examples),
        "duplicate_records": duplicate_records,
        "positives": labels[1],
        "negatives": labels[0],
        "unique_source_frames": len({item.source.digest for item in examples}),
        "unique_target_frames": len({item.target_digest for item in examples}),
        "decoded_frames": len(frames),
    }
    return examples, metadata


def _deduplicate(
    examples: Iterable[ReturnabilityExample], partition: str
) -> Tuple[List[ReturnabilityExample], int]:
    by_key: Dict[Tuple[str, str, str, int], ReturnabilityExample] = {}
    duplicates = 0
    for example in examples:
        key = _example_key(example)
        existing = by_key.get(key)
        if existing is not None:
            if existing.label != example.label:
                raise ValueError(f"conflicting labels in {partition} partition: {key}")
            duplicates += 1
            continue
        by_key[key] = example
    return list(by_key.values()), duplicates


def load_probe_returnability_corpus(
    training_runs: Sequence[Path],
    validation_runs: Sequence[Path],
    required_reward_track: str = "strict",
) -> ProbeReturnabilityCorpus:
    if not training_runs or not validation_runs:
        raise ValueError("probe training and validation runs are both required")
    resolved_training = [
        Path(path).expanduser().resolve() for path in training_runs
    ]
    resolved_validation = [
        Path(path).expanduser().resolve() for path in validation_runs
    ]
    if set(resolved_training) & set(resolved_validation):
        raise ValueError("the same probe run cannot be used in both partitions")

    training_parts = [
        extract_probe_returnability(path, required_reward_track)
        for path in resolved_training
    ]
    validation_parts = [
        extract_probe_returnability(path, required_reward_track)
        for path in resolved_validation
    ]
    source_metadata = [metadata for _, metadata in training_parts + validation_parts]
    run_ids = [metadata["run_id"] for metadata in source_metadata]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("probe source run IDs must be unique")
    configurations = {
        json.dumps(metadata["probe_configuration"], sort_keys=True)
        for metadata in source_metadata
    }
    if len(configurations) != 1:
        raise ValueError("all probe sources must use one matched-NOOP configuration")

    training, training_duplicates = _deduplicate(
        (example for examples, _ in training_parts for example in examples),
        "training",
    )
    validation, validation_duplicates = _deduplicate(
        (example for examples, _ in validation_parts for example in examples),
        "validation",
    )
    training_keys = {_example_key(example) for example in training}
    transition_overlap = [
        example
        for example in validation
        if _example_key(example) in training_keys
    ]
    validation = [
        example for example in validation if _example_key(example) not in training_keys
    ]
    training_source_digests = {example.source.digest for example in training}
    source_overlap = [
        example
        for example in validation
        if example.source.digest in training_source_digests
    ]
    validation = [
        example
        for example in validation
        if example.source.digest not in training_source_digests
    ]
    training_labels = _labels(training)
    validation_labels = _labels(validation)
    if set(training_labels) != {0, 1} or set(validation_labels) != {0, 1}:
        raise ValueError(
            "probe training and validation partitions must each contain both labels"
        )
    configuration = source_metadata[0]["probe_configuration"]
    return ProbeReturnabilityCorpus(
        tuple(training),
        tuple(validation),
        {
            "version": 1,
            "target": "matched-NOOP budget-scoped visual return",
            "reward_track": required_reward_track,
            "probe_configuration": configuration,
            "training_sources": [metadata for _, metadata in training_parts],
            "validation_sources": [metadata for _, metadata in validation_parts],
            "training_examples": len(training),
            "training_positives": training_labels[1],
            "training_negatives": training_labels[0],
            "validation_examples": len(validation),
            "validation_positives": validation_labels[1],
            "validation_negatives": validation_labels[0],
            "training_duplicates_removed": training_duplicates,
            "validation_duplicates_removed": validation_duplicates,
            "validation_transition_overlap_removed": len(transition_overlap),
            "validation_source_overlap_removed": len(source_overlap),
            "validation_overlap_removed": len(transition_overlap)
            + len(source_overlap),
        },
    )


def balanced_probe_sample(
    examples: Sequence[ReturnabilityExample], maximum_examples: int, seed: int
) -> List[ReturnabilityExample]:
    if maximum_examples < 2:
        raise ValueError("maximum examples must be at least two")
    randomizer = random.Random(seed)
    positive = [example for example in examples if example.label == 1]
    negative = [example for example in examples if example.label == 0]
    count = min(len(positive), len(negative), maximum_examples // 2)
    if count == 0:
        raise ValueError("balanced probe sampling requires both labels")
    sample = randomizer.sample(positive, count) + randomizer.sample(negative, count)
    randomizer.shuffle(sample)
    return sample


def probe_validation_sample(
    examples: Sequence[ReturnabilityExample], maximum_examples: int, seed: int
) -> List[ReturnabilityExample]:
    """Preserve held-out prevalence; sample only when the configured cap requires it."""

    if maximum_examples < 2:
        raise ValueError("maximum examples must be at least two")
    labels = _labels(examples)
    if set(labels) != {0, 1}:
        raise ValueError("probe validation requires both labels")
    if len(examples) <= maximum_examples:
        return list(examples)
    randomizer = random.Random(seed)
    sample = randomizer.sample(list(examples), maximum_examples)
    sampled_labels = _labels(sample)
    for label in (0, 1):
        if sampled_labels[label] == 0:
            replacement = randomizer.choice(
                [example for example in examples if example.label == label]
            )
            replace_index = next(
                index
                for index, example in enumerate(sample)
                if sampled_labels[example.label] > 1
            )
            sampled_labels[sample[replace_index].label] -= 1
            sample[replace_index] = replacement
            sampled_labels[label] += 1
    return sample
