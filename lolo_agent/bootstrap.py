from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .environment import Action
from .pixels import Frame, signature_key
from .run_logging import LoggedEnvironment


@dataclass(frozen=True)
class BootstrapStep:
    action: Action
    frames: int


@dataclass(frozen=True)
class BootstrapFixture:
    name: str
    steps: Tuple[BootstrapStep, ...]
    expected_rom_sha256: Optional[str] = None
    expected_frame_sha256: Optional[str] = None
    expected_scene_signature: Optional[str] = None

    @property
    def total_frames(self) -> int:
        return sum(step.frames for step in self.steps)


LOLO1_FIRST_ROOM = BootstrapFixture(
    name="lolo1-first-room",
    steps=(
        BootstrapStep(Action.NOOP, 254),
        BootstrapStep(Action.START, 4),
        BootstrapStep(Action.NOOP, 21),
        BootstrapStep(Action.START, 1),
        BootstrapStep(Action.NOOP, 944),
    ),
    expected_rom_sha256=(
        "914c676959612fc6738a297b6b799dff848e43de4e9bd3c9f3c6783efd059e01"
    ),
    expected_frame_sha256=(
        "cff8e18ba9c039fa4d955be39641ecedb92c54effb2f8177662eca12390e4a78"
    ),
    expected_scene_signature="040604030303040302",
)


BOOTSTRAP_FIXTURES: Dict[str, BootstrapFixture] = {
    LOLO1_FIRST_ROOM.name: LOLO1_FIRST_ROOM,
}


def get_bootstrap_fixture(name: str) -> BootstrapFixture:
    try:
        return BOOTSTRAP_FIXTURES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BOOTSTRAP_FIXTURES))
        raise ValueError(f"unknown bootstrap fixture {name!r}; choose from {choices}") from exc


def apply_bootstrap_fixture(
    env: LoggedEnvironment,
    fixture: BootstrapFixture,
    rom_sha256: Optional[str] = None,
) -> Frame:
    if (
        fixture.expected_rom_sha256 is not None
        and rom_sha256 != fixture.expected_rom_sha256
    ):
        raise ValueError(
            f"bootstrap fixture {fixture.name!r} does not match the supplied ROM"
        )
    if any(step.frames <= 0 for step in fixture.steps):
        raise ValueError("bootstrap action durations must be positive")

    logger = env.logger
    logger.log(
        "bootstrap_started",
        fixture=fixture.name,
        steps=len(fixture.steps),
        total_frames=fixture.total_frames,
        expected_frame=fixture.expected_frame_sha256,
        expected_scene_signature=fixture.expected_scene_signature,
    )
    frame = env.reset(start_attempt=False, phase="bootstrap")
    for index, step in enumerate(fixture.steps, 1):
        frame = env.step(step.action, step.frames)
        logger.log(
            "bootstrap_action_committed",
            fixture=fixture.name,
            index=index,
            action=step.action,
            action_frames=step.frames,
            env_step_seq=env.last_step_seq,
            **logger.frame_fields(frame),
        )

    scene = signature_key(frame.coarse_signature(columns=3, rows=3))
    if (
        fixture.expected_frame_sha256 is not None
        and frame.digest != fixture.expected_frame_sha256
    ):
        raise RuntimeError(
            f"bootstrap fixture {fixture.name!r} ended at unexpected pixels: "
            f"expected {fixture.expected_frame_sha256}, got {frame.digest}"
        )
    if (
        fixture.expected_scene_signature is not None
        and scene != fixture.expected_scene_signature
    ):
        raise RuntimeError(
            f"bootstrap fixture {fixture.name!r} ended in unexpected scene: "
            f"expected {fixture.expected_scene_signature}, got {scene}"
        )
    logger.log(
        "bootstrap_completed",
        fixture=fixture.name,
        steps=len(fixture.steps),
        total_frames=fixture.total_frames,
        **logger.frame_fields(frame),
    )
    return env.start_attempt_from_current(
        frame, reason=f"bootstrap:{fixture.name}"
    )
