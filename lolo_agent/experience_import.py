from __future__ import annotations

import argparse
import json
import os
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .ensemble_world_model import VisualSequence
from .environment import Action
from .pixels import Frame
from .run_logging import read_events, sha256_file, utc_now
from .sequence_store import SequenceStore


@dataclass(frozen=True)
class ExperienceSource:
    run_dir: Path
    through_decision: Optional[int] = None


def classify_reward_track(manifest: Dict[str, Any]) -> str:
    """Classify policy provenance without exposing rewards to model training."""

    configured = manifest.get("metadata", {}).get("reward_track")
    if configured is None or configured == "strict":
        return "strict"
    if isinstance(configured, str) and configured.startswith("human_prior"):
        return "assisted"
    raise ValueError(f"unrecognized telemetry reward track: {configured!r}")


def decode_logged_png(path: Path) -> Frame:
    """Decode the dependency-free, filter-0 PNG format emitted by RunLogger."""

    payload = Path(path).read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a logged PNG: {path}")
    offset = 8
    header: Optional[Tuple[int, int, int]] = None
    compressed = bytearray()
    color_channels = {0: 1, 4: 2, 2: 3, 6: 4}
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError(f"truncated PNG chunk: {path}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        checksum = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])[0]
        if zlib.crc32(data, zlib.crc32(kind)) & 0xFFFFFFFF != checksum:
            raise ValueError(f"PNG checksum mismatch: {path}")
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            if depth != 8 or color_type not in color_channels:
                raise ValueError(f"unsupported logged PNG format: {path}")
            if compression or filtering or interlace:
                raise ValueError(f"unsupported logged PNG encoding: {path}")
            header = (width, height, color_channels[color_type])
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    if header is None:
        raise ValueError(f"logged PNG has no header: {path}")
    width, height, channels = header
    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"logged PNG byte count mismatch: {path}")
    rows = []
    for row in range(height):
        start = row * (stride + 1)
        if raw[start] != 0:
            raise ValueError(f"logged PNG uses an unexpected filter: {path}")
        rows.append(raw[start + 1 : start + stride + 1])
    return Frame(width, height, channels, b"".join(rows))


def _frame(run_dir: Path, digest: str, cache: Dict[str, Frame]) -> Frame:
    cached = cache.get(digest)
    if cached is not None:
        return cached
    frame = decode_logged_png(run_dir / "frames" / f"{digest}.png")
    if frame.digest != digest:
        raise ValueError(f"telemetry frame digest mismatch: {digest}")
    cache[digest] = frame
    return frame


def extract_experience(
    source: ExperienceSource,
    group_offset: int,
    committed_horizon: int = 3,
) -> Tuple[List[VisualSequence], Dict[str, Any]]:
    """Extract pixel/action facts while deliberately ignoring scores and labels."""

    run_dir = Path(source.run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_cache: Dict[str, Frame] = {}
    steps: Dict[int, Dict[str, Any]] = {}
    branches: Dict[str, Dict[str, Any]] = {}
    sequences: List[VisualSequence] = []
    committed_chain: List[Dict[str, Any]] = []
    decisions_to_group: Dict[int, int] = {}
    next_group = group_offset
    verified_count = 0
    committed_windows = 0

    def decision_group(decision: int) -> int:
        nonlocal next_group
        if decision not in decisions_to_group:
            decisions_to_group[decision] = next_group
            next_group += 1
        return decisions_to_group[decision]

    for event in read_events(run_dir):
        kind = event.get("event")
        if kind == "env_step" and event.get("phase", "agent") == "agent":
            steps[int(event["seq"])] = event
        elif kind == "branch_verified":
            decision = int(event["decision"])
            if source.through_decision is not None and decision > source.through_decision:
                continue
            step = steps.get(int(event["env_step_seq"]))
            if step is None:
                raise ValueError(
                    f"branch {event.get('branch_id')} references a missing environment step"
                )
            source_digest = step["source_frame"]
            target_digest = step["target_frame"]
            sequence = VisualSequence(
                decision_group(decision),
                (
                    _frame(run_dir, source_digest, frame_cache),
                    _frame(run_dir, target_digest, frame_cache),
                ),
                (Action(step["action"]),),
                (int(step["action_frames"]),),
            )
            sequences.append(sequence)
            state_id = event.get("state_id")
            if state_id:
                branches[state_id] = {"step": step, "decision": decision}
            verified_count += 1
        elif kind == "decision_committed":
            decision = int(event["decision"])
            if source.through_decision is not None and decision > source.through_decision:
                continue
            branch = branches.get(event.get("committed_state_id"))
            if event.get("restored_archive") or branch is None:
                committed_chain.clear()
                continue
            step = branch["step"]
            if committed_chain and committed_chain[-1]["target_frame"] != step["source_frame"]:
                committed_chain.clear()
            committed_chain.append(step)
            if len(committed_chain) > committed_horizon:
                committed_chain.pop(0)
            if len(committed_chain) == committed_horizon:
                frames = [_frame(run_dir, committed_chain[0]["source_frame"], frame_cache)]
                frames.extend(
                    _frame(run_dir, item["target_frame"], frame_cache)
                    for item in committed_chain
                )
                sequences.append(
                    VisualSequence(
                        decision_group(decision),
                        tuple(frames),
                        tuple(Action(item["action"]) for item in committed_chain),
                        tuple(int(item["action_frames"]) for item in committed_chain),
                    )
                )
                committed_windows += 1

    if not sequences:
        raise ValueError(f"no verified pixel/action experience found in {run_dir}")
    metadata = {
        "run": str(run_dir),
        "run_id": manifest.get("run_id", run_dir.name),
        "reward_track": classify_reward_track(manifest),
        "manifest_sha256": sha256_file(manifest_path),
        "events_sha256": sha256_file(run_dir / "events.jsonl"),
        "through_decision": source.through_decision,
        "verified_transitions": verified_count,
        "committed_windows": committed_windows,
        "sequences": len(sequences),
        "groups": len(decisions_to_group),
        "unique_frames": len(frame_cache),
        "first_group": group_offset,
        "next_group": next_group,
    }
    return sequences, metadata


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def import_experience_cycle(
    experiment_dir: Path,
    sources: Sequence[ExperienceSource],
    committed_horizon: int = 3,
    reward_track: str = "strict",
) -> Dict[str, Any]:
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    state_path = experiment_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("phase") != "idle" or state.get("current_cycle") is not None:
        raise ValueError("experience can only be imported while the experiment is idle")
    if not sources:
        raise ValueError("at least one experience source is required")
    if reward_track not in ("strict", "assisted"):
        raise ValueError("reward track must be 'strict' or 'assisted'")
    cycle = int(state["completed_cycles"]) + 1
    segment_id = f"cycle-{cycle:06d}"
    store = SequenceStore(experiment_dir / "dataset")
    if store.has_segment(segment_id):
        raise FileExistsError(store.segment_path(segment_id))
    group = int(state["next_group"])
    all_sequences: List[VisualSequence] = []
    source_metadata = []
    for source in sources:
        extracted, metadata = extract_experience(source, group, committed_horizon)
        if metadata["reward_track"] != reward_track:
            raise ValueError(
                f"cannot import {metadata['reward_track']!r} experience into the "
                f"{reward_track!r} dataset: {metadata['run_id']}"
            )
        all_sequences.extend(extracted)
        source_metadata.append(metadata)
        group = int(metadata["next_group"])
    store.bind_reward_track(reward_track)
    store.append_segment(segment_id, all_sequences)
    provenance = {
        "version": 2,
        "created_at": utc_now(),
        "segment": segment_id,
        "reward_track": reward_track,
        "persistent_inputs": ["pixels", "actions", "action_durations"],
        "excluded_inputs": ["evaluator_annotations", "planner_scores", "object_labels", "rewards"],
        "committed_horizon": committed_horizon,
        "sources": source_metadata,
        "sequences": len(all_sequences),
        "dataset": store.statistics(),
    }
    _atomic_json(experiment_dir / "imports" / f"{segment_id}.json", provenance)
    state.update(
        {
            "phase": "collected",
            "current_cycle": cycle,
            "next_group": group,
            "error": None,
            "updated_at": utc_now(),
        }
    )
    _atomic_json(state_path, state)
    return provenance


def _source(value: str) -> ExperienceSource:
    path, separator, decision = value.rpartition(":")
    if separator and decision.isdigit():
        return ExperienceSource(Path(path), int(decision))
    return ExperienceSource(Path(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import verified pixel/action telemetry as the next experiment cycle"
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=_source,
        action="append",
        required=True,
        help="run directory, optionally suffixed with :THROUGH_DECISION",
    )
    parser.add_argument("--committed-horizon", type=int, default=3)
    parser.add_argument(
        "--reward-track",
        choices=("strict", "assisted"),
        default="strict",
        help="bind the dataset to strict or explicitly assisted policy provenance",
    )
    args = parser.parse_args()
    if args.committed_horizon <= 0:
        parser.error("--committed-horizon must be positive")
    result = import_experience_cycle(
        args.experiment_dir,
        args.source,
        args.committed_horizon,
        reward_track=args.reward_track,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
