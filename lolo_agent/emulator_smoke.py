from __future__ import annotations

import argparse
from pathlib import Path

from .environment import Action
from .libretro import LibretroEnv


def _write_ppm(frame_path: Path, width: int, height: int, pixels: bytes) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify raw libretro frames and save states")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--boot-frames", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with LibretroEnv(args.core, args.rom) as env:
        env.reset()
        for _ in range(args.boot_frames):
            frame = env.step(Action.NOOP)
        root = env.save_state()
        env.step(Action.START, 2)
        for _ in range(60):
            first = env.step(Action.NOOP)
        first_state = env.save_state()
        env.load_state(root)
        env.step(Action.START, 2)
        for _ in range(60):
            second = env.step(Action.NOOP)
        second_state = env.save_state()
        if first != second:
            raise SystemExit("save-state determinism check failed")
        # Serialized buffers may contain core-owned padding bytes, so bytewise
        # equality is not a valid semantic determinism test. Restore both and
        # require their future pixel trajectories to agree instead.
        env.load_state(first_state)
        for _ in range(30):
            first_continuation = env.step(Action.NOOP)
        env.load_state(second_state)
        for _ in range(30):
            second_continuation = env.step(Action.NOOP)
        if first_continuation != second_continuation:
            raise SystemExit("save-state continuation check failed")
        env.load_state(root)
        for _ in range(62):
            control = env.step(Action.NOOP)
        if first == control:
            raise SystemExit("controller branch did not diverge from NOOP branch")
        if args.output:
            _write_ppm(args.output, frame.width, frame.height, frame.pixels)
        print(f"core={env.core_name} version={env.core_version}")
        print(f"geometry={frame.width}x{frame.height} channels={frame.channels}")
        print(f"fps={env.fps:.6f}")
        print(f"frame_sha256={frame.digest}")
        print(f"state_bytes={len(root)}")
        print("save_state_replay=pass")
        print("save_state_continuation=pass")
        print("controller_branch=pass")


if __name__ == "__main__":
    main()
