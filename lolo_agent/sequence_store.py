from __future__ import annotations

import json
import os
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .ensemble_world_model import VisualSequence
from .environment import Action
from .pixels import Frame


@dataclass(frozen=True)
class StoredTransition:
    """One pixel-state edge reconstructed from persistent sequence metadata."""

    source_digest: str
    target_digest: str
    action: Action
    duration: int
    source_run_id: str


class SequenceStore:
    """Crash-safe, segmented sequence dataset with deduplicated compressed frames."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.frames_dir = self.root / "frames"
        self.segments_dir = self.root / "segments"
        self.track_path = self.root / "reward-track.json"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)

    @property
    def reward_track(self) -> Optional[str]:
        if not self.track_path.is_file():
            return None
        payload = json.loads(self.track_path.read_text(encoding="utf-8"))
        track = payload.get("reward_track")
        if track not in ("strict", "assisted"):
            raise ValueError(f"unsupported dataset reward track: {track!r}")
        return str(track)

    def bind_reward_track(self, reward_track: str) -> None:
        """Permanently prevent strict and assisted experience from mixing."""

        if reward_track not in ("strict", "assisted"):
            raise ValueError("reward track must be 'strict' or 'assisted'")
        existing = self.reward_track
        if existing is not None:
            if existing != reward_track:
                raise ValueError(
                    f"dataset is bound to the {existing!r} reward track, not {reward_track!r}"
                )
            return
        temporary = self.root / f".{self.track_path.name}.tmp"
        temporary.write_text(
            json.dumps(
                {"version": 1, "reward_track": reward_track},
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.track_path)

    def segment_path(self, segment_id: str) -> Path:
        if not segment_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in segment_id):
            raise ValueError("segment IDs may contain lowercase letters, digits, dashes, and underscores")
        return self.segments_dir / f"{segment_id}.jsonl"

    def has_segment(self, segment_id: str) -> bool:
        return self.segment_path(segment_id).is_file()

    def append_segment(self, segment_id: str, sequences: Sequence[VisualSequence]) -> Path:
        if not sequences:
            raise ValueError("cannot persist an empty sequence segment")
        destination = self.segment_path(segment_id)
        if destination.exists():
            raise FileExistsError(destination)
        records = []
        for sequence in sequences:
            frame_records = []
            for frame in sequence.frames:
                frame_path = self.frames_dir / f"{frame.digest}.rgb.zlib"
                if not frame_path.exists():
                    temporary_frame = self.frames_dir / f".{frame.digest}.tmp"
                    temporary_frame.write_bytes(zlib.compress(frame.pixels, level=6))
                    os.replace(temporary_frame, frame_path)
                frame_records.append(
                    {
                        "digest": frame.digest,
                        "width": frame.width,
                        "height": frame.height,
                        "channels": frame.channels,
                    }
                )
            records.append(
                {
                    "version": 2,
                    "group": sequence.group,
                    "frames": frame_records,
                    "actions": [action.value for action in sequence.actions],
                    "durations": list(sequence.durations),
                    "source_run_id": sequence.source_run_id,
                }
            )
        temporary = self.segments_dir / f".{segment_id}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination

    def _records(self) -> Iterable[Dict[str, Any]]:
        for segment in sorted(self.segments_dir.glob("*.jsonl")):
            with segment.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    record = json.loads(line)
                    if record.get("version") not in (1, 2):
                        raise ValueError(
                            f"unsupported sequence record in {segment}:{line_number}"
                        )
                    record["_segment_id"] = segment.stem
                    yield record

    def _decode_records(self, records: Iterable[Dict[str, Any]]) -> List[VisualSequence]:
        frame_cache: Dict[str, Frame] = {}
        sequences = []
        for record in records:
            frames = []
            for item in record["frames"]:
                digest = item["digest"]
                frame = frame_cache.get(digest)
                if frame is None:
                    payload_path = self.frames_dir / f"{digest}.rgb.zlib"
                    pixels = zlib.decompress(payload_path.read_bytes())
                    frame = Frame(
                        int(item["width"]),
                        int(item["height"]),
                        int(item["channels"]),
                        pixels,
                    )
                    if frame.digest != digest:
                        raise ValueError(f"dataset frame digest mismatch: {digest}")
                    frame_cache[digest] = frame
                frames.append(frame)
            sequences.append(
                VisualSequence(
                    int(record["group"]),
                    tuple(frames),
                    tuple(Action(value) for value in record["actions"]),
                    tuple(int(value) for value in record.get("durations", [])),
                    self._record_source_run_id(record),
                )
            )
        return sequences

    @staticmethod
    def _record_source_run_id(record: Dict[str, Any]) -> str:
        source_run_id = (
            str(record.get("source_run_id", ""))
            if record.get("version") == 2
            else ""
        )
        return source_run_id or f"legacy-segment:{record['_segment_id']}"

    @classmethod
    def _record_group_key(cls, record: Dict[str, Any]) -> Tuple[str, int]:
        """Disambiguate group counters that may restart in another source run."""

        return cls._record_source_run_id(record), int(record["group"])

    def load(self) -> List[VisualSequence]:
        return self._decode_records(self._records())

    def transition_metadata(self) -> List[StoredTransition]:
        """Read graph edges without decoding the large RGB frame payloads."""

        transitions = []
        for record in self._records():
            frames = [str(item["digest"]) for item in record["frames"]]
            actions = [Action(value) for value in record["actions"]]
            durations = [int(value) for value in record.get("durations", [])]
            if not durations:
                durations = [4] * len(actions)
            run_id = self._record_source_run_id(record)
            transitions.extend(
                StoredTransition(source, target, action, duration, run_id)
                for source, target, action, duration in zip(
                    frames, frames[1:], actions, durations
                )
            )
        return transitions

    def load_frame_subset(self, digests: Iterable[str]) -> Dict[str, Frame]:
        """Decode only requested content-addressed frames from the store."""

        requested = set(digests)
        if not requested:
            return {}
        metadata: Dict[str, Tuple[int, int, int]] = {}
        for record in self._records():
            for item in record["frames"]:
                digest = str(item["digest"])
                if digest in requested and digest not in metadata:
                    metadata[digest] = (
                        int(item["width"]),
                        int(item["height"]),
                        int(item["channels"]),
                    )
            if len(metadata) == len(requested):
                break
        missing = requested - metadata.keys()
        if missing:
            raise KeyError(f"dataset does not contain {len(missing)} requested frames")
        frames = {}
        for digest, (width, height, channels) in metadata.items():
            pixels = zlib.decompress(
                (self.frames_dir / f"{digest}.rgb.zlib").read_bytes()
            )
            frame = Frame(width, height, channels, pixels)
            if frame.digest != digest:
                raise ValueError(f"dataset frame digest mismatch: {digest}")
            frames[digest] = frame
        return frames

    def load_sample(self, maximum_sequences: int, seed: int = 0) -> List[VisualSequence]:
        """Uniformly sample records before decoding their large RGB payloads."""

        if maximum_sequences <= 0:
            raise ValueError("maximum sequence count must be positive")
        randomizer = random.Random(seed)
        sample: List[Dict[str, Any]] = []
        for seen, record in enumerate(self._records(), 1):
            if len(sample) < maximum_sequences:
                sample.append(record)
                continue
            replacement = randomizer.randrange(seen)
            if replacement < maximum_sequences:
                sample[replacement] = record
        return self._decode_records(sample)

    def load_group_sample(
        self,
        maximum_groups: int,
        seed: int = 0,
        minimum_multistep_groups: int = 0,
    ) -> List[VisualSequence]:
        """Sample complete causal branch groups while decoding only selected RGB data."""

        if maximum_groups <= 0:
            raise ValueError("maximum group count must be positive")
        if minimum_multistep_groups < 0:
            raise ValueError("minimum multistep group count must be non-negative")
        if minimum_multistep_groups > maximum_groups:
            raise ValueError("minimum multistep groups cannot exceed maximum groups")
        randomizer = random.Random(seed)
        sampled_groups: List[Tuple[str, int]] = []
        seen_groups = set()
        if minimum_multistep_groups:
            multistep_groups = sorted(
                {
                    self._record_group_key(record)
                    for record in self._records()
                    if len(record["actions"]) > 1
                }
            )
            selected_multistep = randomizer.sample(
                multistep_groups,
                min(minimum_multistep_groups, len(multistep_groups)),
            )
            sampled_groups.extend(selected_multistep)
            seen_groups.update(selected_multistep)
        reserved_count = len(sampled_groups)
        replaceable_seen = 0
        for record in self._records():
            group = self._record_group_key(record)
            if group in seen_groups:
                continue
            seen_groups.add(group)
            replaceable_seen += 1
            if len(sampled_groups) < maximum_groups:
                sampled_groups.append(group)
                continue
            replaceable_groups = maximum_groups - reserved_count
            replacement = randomizer.randrange(replaceable_seen)
            if replacement < replaceable_groups:
                sampled_groups[reserved_count + replacement] = group
        selected = set(sampled_groups)
        return self._decode_records(
            record
            for record in self._records()
            if self._record_group_key(record) in selected
        )

    def statistics(self) -> Dict[str, int]:
        segments = sorted(self.segments_dir.glob("*.jsonl"))
        sequences = 0
        for segment in segments:
            with segment.open(encoding="utf-8") as handle:
                sequences += sum(1 for line in handle if line.strip())
        return {
            "segments": len(segments),
            "sequences": sequences,
            "unique_frames": len(list(self.frames_dir.glob("*.rgb.zlib"))),
            "compressed_frame_bytes": sum(
                path.stat().st_size for path in self.frames_dir.glob("*.rgb.zlib")
            ),
        }
