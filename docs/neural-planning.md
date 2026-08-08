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
coverage, action streaks, and archived state handles are temporary attempt
memory and are discarded after the run.

All proposals, real verification branches, committed decisions, archive
operations, opaque state lifecycles, and pixel observations are recorded by the
event-sourced telemetry layer described in [telemetry.md](telemetry.md). Logging
is observational and does not return evaluator annotations to the planner.

New duration-conditioned checkpoints also embed the number of emulator frames
for which each controller button is held. Planning, verification, telemetry,
and replay therefore distinguish, for example, `RIGHT@1` from `RIGHT@16`.
Fixed-duration version-one checkpoints remain supported but cannot safely plan
over multiple press lengths.

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

## Known limitations

- The training sample is dominated by boot, menu, and castle animation frames.
- Controller duration is fixed at four emulator frames. An eight-frame audit
  changed behavior substantially, so duration should become a learned action
  variable rather than a hand-selected permanent macro.
- Action coverage currently provides a strong temporary exploration prior and
  can over-regularize behavior toward uniform controller use.
- The model has no explicit object slots, reachability head, or reversibility
  prediction yet.
- Room completion remains evaluator-only and is not available to the planner.
