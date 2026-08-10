from __future__ import annotations

import json
import os
import random
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .ensemble_world_model import VisualSequence
from .environment import Action
from .pixels import Frame


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
                    "version": 1,
                    "group": sequence.group,
                    "frames": frame_records,
                    "actions": [action.value for action in sequence.actions],
                    "durations": list(sequence.durations),
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
                    if record.get("version") != 1:
                        raise ValueError(
                            f"unsupported sequence record in {segment}:{line_number}"
                        )
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
                )
            )
        return sequences

    def load(self) -> List[VisualSequence]:
        return self._decode_records(self._records())

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

    def load_group_sample(self, maximum_groups: int, seed: int = 0) -> List[VisualSequence]:
        """Sample complete causal branch groups while decoding only selected RGB data."""

        if maximum_groups <= 0:
            raise ValueError("maximum group count must be positive")
        randomizer = random.Random(seed)
        sampled_groups: List[int] = []
        seen_groups = set()
        for record in self._records():
            group = int(record["group"])
            if group in seen_groups:
                continue
            seen_groups.add(group)
            seen = len(seen_groups)
            if len(sampled_groups) < maximum_groups:
                sampled_groups.append(group)
                continue
            replacement = randomizer.randrange(seen)
            if replacement < maximum_groups:
                sampled_groups[replacement] = group
        selected = set(sampled_groups)
        return self._decode_records(
            record for record in self._records() if int(record["group"]) in selected
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
