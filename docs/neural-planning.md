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
handles. If the agent stops producing new exact visual signatures for several
consecutive decisions, it may restore a recent branch from a different scene.
The archive has a fixed capacity and age window. No screen or object is labeled
as a menu, password entry, death, room, or success state.

This mechanism was necessary because exact-frame novelty can remain high during
animations or cursor movement even when controller exploration is repetitive.
Coarse-scene dwell remains telemetry but no longer triggers recovery by itself;
an animation that keeps producing new frames is not treated as stagnant.

The agent also tracks delayed returns to informative visual signatures. If a
trajectory reaches the same sufficiently varied pixel signature again after
several decisions, it assigns temporary cost to the intervening
scene/action/duration choices. On the next decision it restores a distinct
archived state from inside that loop when one is available. Later planning in
the same attempt penalizes choices that previously participated in such a
return. Solid fades are excluded using visual variation alone; there are no
screen, object, menu, death, or progress labels.

## Persistent-frontier value

Every recently visited visual state carries a temporary successor-novelty
trace. First visits to exact-frame and coarse-scene signatures are credited
backward through those traces with a discount; repeated pixels contribute zero.
A state therefore gains value only when its
descendants continue producing unfamiliar visual regions over several real
decisions. After the configured horizon, the observed discounted return is
folded into that signature's running value estimate.

If the trajectory returns to a prior informative signature, provisional gains
inside the loop are discarded and the affected signatures receive a negative
return sample. Save-state alternatives inherit value from the visual state
where they were created. Candidate actions, archive restoration, and archive
pruning use these learned temporary estimates in addition to immediate novelty.
Traces restart after an archive jump so evidence is never propagated across a
teleport that the controller did not cause.

The same return is learned for the originating
visual-signature/action/duration choice. Inherited state value is used only
while an alternative is untested; once that exact choice has produced a
trajectory, its own successor return overrides the inherited optimism. This
lets two controller choices from the same pixels acquire different values
without assigning a semantic meaning to either button.

### Learned visual abstraction

Frontier keys are online clusters in the frozen encoder's latent space rather
than exact pixel signatures. A frame is compared only with clusters sharing its
coarse visual scene, then joins the nearest cluster below a latent-RMSE
threshold or starts a new one. Cluster centroids are temporary running means;
updating them does not alter neural parameters. Exact signatures remain in use
for novelty and conservative delayed-loop detection.

This lets neighboring animation frames share action-duration return evidence
when the learned encoder represents them similarly. The coarse-scene partition
guards against merging visually distinct regions that happen to be close in an
imperfect latent space. Both cluster membership and value memory reset for each
evaluation attempt.

### Interaction-derived behavioral refinement

Visual clusters are candidates for sharing, not sufficient evidence. Whenever
a state becomes the current planning root, the verifier reserves a neutral
anchor and an actively selected controller probe at a common duration. Before
multiple behavioral hypotheses exist, the active probe rotates toward the
least-observed control in that visual cluster. When hypotheses compete, the
agent selects the previously observed control whose successor centroids have
the largest latent separation. This reuses the existing verification budget.

For each probe the agent records the frozen-encoder displacement from source
pixels to observed successor pixels. A state joins an existing behavioral
cluster only when its visual cluster matches and the mean
successor-displacement RMSE over shared probes is below the configured
threshold. A sole hypothesis may conservatively acquire a previously unseen
probe using the neutral anchor match; once multiple hypotheses exist, full
shared-probe evidence is required. Probe centroids accumulate across visits.

Unprobed successor states receive unique provisional frontier signatures. They
do not inherit another state's value merely because their pixels look alike.
When a provisional state is later probed, its temporary values, active traces,
and archived origin references migrate into the matching behavioral cluster.
The cluster centroids and migrations are temporary evaluation memory; model
weights remain frozen.

### Interaction-derived temporal options

The verifier also links a controller choice to delayed consequences across an
action-independent visual sequence. When matched real-action probes show that
the current pixels evolve independently of the selected controller input, the
agent starts a temporary option trace. It credits the immediately preceding
behavioral-signature/action/duration choice only when a real, same-duration
save-state branch with another controller action produced distinguishable
pixels. Otherwise the sequence is retained as uncredited passive motion; this
prevents a coincidental button press from receiving value for an uncaused
timer transition. A credited trace follows neutral waits through short static
grace pauses and ends at the next controllable planning root.

The endpoint sample uses only interaction-derived quantities: whether the
endpoint behavioral signature is new, how many coarse visual scenes the trace
crossed, its duration, and whether it returned to its initiating signature.
Novel endpoints and broader, longer transitions receive positive value;
returns to the source receive a penalty. Later candidate scoring and archive
selection can reuse the running value for that exact initiating choice.

Boot-time passive motion has no initiating choice and is logged but does not
produce a learned sample. An archive restore discards any active trace before
arming the restored branch as a new initiating choice, preventing credit from
crossing an uncaused save-state jump. Option traces and values migrate with
provisional behavioral signatures, reset between attempts, and never update
the frozen neural model.

The value is learned online from pixels and transition topology and is erased
on reset. It is not part of the neural checkpoint, so frozen evaluation still
does not update persistent parameters.

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
