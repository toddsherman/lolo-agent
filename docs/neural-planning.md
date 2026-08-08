# Neural rollout planning

## Current design

`EnsembleVisualDynamicsModel` shares a convolutional encoder, action embedding,
and pixel decoder across three independently initialized dynamics heads. It is
trained on short action sequences collected from alternative save-state
branches. Dynamics heads receive bootstrapped subsets of each batch.

For a candidate action sequence, `NeuralRolloutPlanner` propagates every head
through latent space. It scores latent change and head disagreement, while
preserving several distinct first actions in its beam. The planner does not
decode every imagined step during search.

`VerifiedNeuralAgent` treats these rollouts as proposals:

1. rank multi-step action sequences in latent space;
2. restore the current native save-state handle;
3. execute several distinct first actions in the real emulator;
4. score their actual pixels, prediction errors, and temporary novelty;
5. commit one branch and archive recent alternatives.

The neural model is frozen during this process. Novelty counts, controller
coverage, action streaks, delayed-return costs, and archived state handles are
temporary attempt memory and are discarded after the run.

All proposals, real verification branches, committed decisions, archive
operations, opaque state lifecycles, and pixel observations are recorded by the
event-sourced telemetry layer described in [telemetry.md](telemetry.md). Logging
is observational and does not return evaluator annotations to the planner.

New duration-conditioned checkpoints also embed the number of emulator frames
for which each controller button is held. Planning, verification, telemetry,
and replay therefore distinguish, for example, `RIGHT@1` from `RIGHT@16`.
Fixed-duration version-one checkpoints remain supported but cannot safely plan
over multiple press lengths.

During real verification, the agent prioritizes least-tested distinct buttons
within the current coarse visual scene before spending branch budget on extra
durations. Equal-duration `NOOP` and non-`NOOP` probes estimate temporary
controllability. If different buttons produce the same evolving pixels, the
agent selects the longest neutral wait and avoids archiving timestamps as if
they were meaningful alternatives. A short visual-dynamics grace window keeps
neutral waiting through temporary static pauses, then expires automatically if
motion does not resume.

Exact-frame novelty is moderated by coarse-scene novelty. Archive pruning is
also scene-diverse: branches from a highly populated scene are removed before
the last frontier from a minority scene. Both mechanisms are visual and
temporary; neither introduces menu, room, object, or success labels.

## Multi-step validation

Sequence groups, rather than individual frames, are assigned to training or
validation. This prevents sibling branches from the same root state appearing
on both sides of the split.

The August 2026 M5 smoke run used 96 training sequences and 24 held-out
sequences at horizon three. Held-out RGB L1 errors changed from:

```text
before: 0.402257, 0.411288, 0.411205
after:  0.392015, 0.400750, 0.400262
```

Ensemble disagreement increased with horizon:

```text
0.00016290, 0.00044415, 0.00084617
```

The uncertainty/error correlation was positive but weak (`0.218831`). These are
pipeline diagnostics from one small run, not benchmark results.

## Safety gate for imagined rollouts

Deep model rollouts remain advisory. The first action is verified with real
emulator branches because uncertainty is not yet calibrated strongly enough to
trust unverified execution.

Removing verification requires all of the following on held-out gameplay
sequences across multiple seeds:

- error remains bounded at the configured planning horizon;
- ensemble disagreement increases with horizon;
- uncertainty/error correlation is consistently positive;
- action ranking outperforms random and one-step ablations;
- the frozen checkpoint digest remains unchanged throughout evaluation.

## Episodic recovery

Verified but rejected branches are retained temporarily as opaque native state
handles. If the agent remains in the same coarse visual scene for several
consecutive decisions, it may restore a recent branch from a different scene.
The archive has a fixed capacity and age window. No screen or object is labeled
as a menu, password entry, death, room, or success state.

This mechanism was necessary because exact-frame novelty can remain high during
animations or cursor movement even when controller exploration is repetitive.

The agent also tracks delayed returns to informative visual signatures. If a
trajectory reaches the same sufficiently varied pixel signature again after
several decisions, it assigns temporary cost to the intervening
scene/action/duration choices. On the next decision it restores a distinct
archived state from inside that loop when one is available. Later planning in
the same attempt penalizes choices that previously participated in such a
return. Solid fades are excluded using visual variation alone; there are no
screen, object, menu, death, or progress labels.

## Known limitations

- The training sample is dominated by boot, menu, and castle animation frames.
- Controller duration is represented explicitly by duration-conditioned
  checkpoints, although the current discrete duration set remains configured
  rather than autonomously expanded.
- Action coverage currently provides a strong temporary exploration prior and
  can over-regularize behavior toward uniform controller use.
- The model has no explicit object slots, reachability head, or reversibility
  prediction yet.
- Room completion remains evaluator-only and is not available to the planner.
