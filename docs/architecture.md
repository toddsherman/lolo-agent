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
offline and native prediction gate. Its counterfactual usefulness score remains
disabled by default because paired planning ablations were mixed.
See
`spatial-causal-model-2026-08-10.md` for the reproducible MPS result.

`spatial_returnability.py` adds a separate ensemble relation head over frozen
spatial tokens. Its targets come only from the observed pixel-transition graph:
positive transitions have an observed return path, well-probed non-returns are
negative, and uncertain cases are censored. The sidecar passes one run-held-out
development fold but fails native branch calibration, so it is attached only
to telemetry and cannot affect the planner.

`bidirectional_probe.py` collects stronger relation targets directly from
opaque save-state branches. It performs a bounded pixel-only search from each
endpoint toward a duration-matched NOOP reference from the original root. The
collector has a separate logging phase and no reference to planner scores,
objects, rewards, or evaluator labels. Its output is intended to replace the
policy-dependent graph negatives used by the first returnability sidecar.

`probe_returnability_import.py` converts those explicit outcomes into a
provenance-audited relation corpus. Complete training and validation runs are
named separately; strict and assisted telemetry cannot mix, exact transition
and source-image overlap are removed from validation, and both partitions must
retain return and budget-scoped non-return examples. The relation head is
trained from frozen spatial encodings of the observed source and verified
endpoint. During live planning it can inspect only predicted endpoints; after
branch execution its telemetry uses the observed endpoint. Neither path
currently affects control.

`entity_behavior.py` adds the first persistent object-centric sidecar. It
clusters pooled RGB patches into anonymous appearance types and learns
conditional empirical outcome distributions from matched controls and passive
intervals. Outcomes are position-relative pixel-effect hashes; types have no
sprite names or supplied mechanics. Context-specific distributions can differ
from the cross-room fallback, so one appearance may be stationary in one
anonymous visual context and mobile in another. Exact save-state evidence is
deduplicated, contradictory observations reduce confidence, and a frozen mode
supports parameter-immutable evaluation. The sidecar currently has selection
weight zero while native held-out calibration is collected.

The remaining successor milestones are:

1. changed-region rendering that reliably beats persistence on native branches;
2. multi-fold run- and room-held-out validation without semantic inputs;
3. promote calibrated anonymous type-conditioned predictions from observation
   to planning, then replace the assisted action-ray locator with a learned
   controlled-entity tracker;
4. a native-generalizing reachability/reversibility representation, followed
   by terminal-risk estimates learned from trajectories rather than hand-authored
   death or object rules.
