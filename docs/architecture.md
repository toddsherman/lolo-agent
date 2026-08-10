# Architecture

The core deliberately separates the agent's observations from evaluator-only
ground truth.

```text
                 evaluator-only success / room accounting
                                  ^
                                  |
Nestopia <-> native host <-> PixelSaveStateEnv <-> branching planner
             RGB/actions      opaque handles       |       |
                                                   |       +-> temporary memory
                                                   +----------> world model
```

`PixelSaveStateEnv` returns raw images. Save-state blobs are opaque: the agent
may store and restore them but cannot decode them. Controller button names are
hardware affordances, not game semantics.

`EmpiricalWorldModel` is the dependency-free planning baseline. It predicts distributions
over coarse, object-agnostic visual signatures for each action. It combines a
state-conditioned table with an action-conditioned fallback, enabling limited
transfer while remaining dependency-free. It is intentionally replaceable by
a convolutional encoder plus recurrent latent dynamics model.

`VisualDynamicsModel` is the first neural training baseline. It learns a
convolutional image encoder, action-conditioned latent transition, and pixel
decoder on branched emulator transitions. It currently validates the neural
data and freeze pipeline; planner integration follows after multi-step model
validation.

`BranchingAgent` performs receding-horizon beam search. It restores a state,
tries actions, scores visual novelty, prediction surprise, uncertainty, and
screen change, and commits only the first action of the best path. During
training all observed branches can update the model. In evaluation, model
updates raise an error and only temporary per-run memory changes.

The mock puzzle is not imported by the agent package. Its symbolic state and
success predicate are evaluator-side test fixtures; the agent receives only
the rendered byte array.

## Spatial causal successor

The first persistent spatial successor is implemented in
`spatial_world_model.py`. It preserves a learned 2-D token map, predicts
action-localized pixel effects with an uncertainty ensemble, and uses sparse
rendering so unchanged pixels are copied rather than regenerated. Strict and
assisted telemetry are now bound to separate dataset tracks.

Its local flow/residual renderer passes a trajectory-balanced run-held-out
offline gate by a narrow margin. Planner integration remains gated because a
native zero-weight shadow run beat persistence on only 3/30 verified branches.
See
`spatial-causal-model-2026-08-10.md` for the reproducible MPS result.

The remaining successor milestones are:

1. changed-region rendering that reliably beats persistence on native branches;
2. multi-fold run- and room-held-out validation without semantic inputs;
3. object-centric slots or sparse entity tokens discovered without labels;
4. terminal/reversibility estimates learned from long-horizon reachability,
   not hand-authored death or object rules.
