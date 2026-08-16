# Experimental protocol

## Information boundary

The agent may receive only pixels, its previously issued controller actions,
and subsequent pixels. It may create, retain, and restore opaque emulator save
states. It must not inspect CPU/PPU memory, sprite tables, tile IDs, room IDs,
score counters decoded from RAM, or evaluator success signals.

Video preprocessing must be identical across train and evaluation. Cropping or
palette normalization may be fixed globally, but not selected per room.

## Splits

Split Lolo 1 by room before collecting experience. Publish the room identifiers
used by the evaluator, but do not expose the current identifier to the agent.
Use multiple seeds and report room solve rate, controller actions, emulator
frames, restored branches, and wall-clock compute.

For Lolo 2, start from the final Lolo 1 checkpoint. Freeze every persistent
learned parameter and persistent statistic. The agent may construct fresh
temporary search trees, novelty counts, and episodic memories for the current
attempt, then discard them.

Opaque native save-state handles and the temporary alternative-branch archive
are attempt memory. They must not be serialized into the learned checkpoint or
carried between evaluation rooms unless the published evaluation policy treats
those rooms as one continuous attempt.

## Pre-registered room partition

The room split is pre-registered in `configs/evaluation-partitions.json` and
loaded by `lolo_agent/partitions.py`. The manifest is immutable once
committed: the loader rejects any manifest that does not declare itself
immutable, and both the file digest and a canonical content signature are
recorded with every load so any later edit is detectable.

Rooms are allocated to exactly one of four categories: `training`,
`development`, `withheld_lolo1`, and `sequel`. Lolo 1 rooms 1 through 3 are
development rooms because save states and targeted engineering decisions
already used them; Room 3 in particular is not eligible as withheld and the
loader enforces this. The withheld Lolo 1 rooms are the final room of each
floor from floor five through floor ten, pre-registered before any
engineering contact. Every Lolo 2 room is sequel.

The withheld and sequel partitions are strict-track only and frozen. The
loader rejects writing training artifacts from withheld or sequel runs and
emits a loud `partition_update_rejected` event; training or assisted events
cannot be imported into a strict withheld corpus without an explicit
test-only override. Each evaluation records `evaluation_partition_loaded`
and a `persistent_artifact_digest_audited` audit covering every persistent
artifact class (neural, spatial, entity, and relational), and each run's
`manifest.json` carries explicit partition, reward-track, and
update-authority fields. The manifest also records the frozen baseline
digests used by `v318` through `v321`: neural checkpoint, behavior
checkpoint, native host, core, ROM, and the per-version planning
configurations.

## Freeze audit

Hash the serialized persistent model immediately before and after each frozen
evaluation. The hashes must match. Save-state blobs and temporary memory are
excluded from the checkpoint. Run evaluation in a fresh process where possible
and fail closed if any model update is attempted.

## Evaluation-side success

The evaluator may use emulator instrumentation or manually verified screen
recognition to determine whether a room was completed. This value is recorded
for metrics only and is not returned through `PixelSaveStateEnv`. Timeouts and
reset policy must be fixed before the test set is run.

Report both withheld-room and sequel results alongside ablations for planning,
save-state branching, episodic memory, intrinsic novelty, and model learning.
