# Run telemetry

Each `lolo-neural-run` writes a new directory below `runs/` (ignored by Git):

```text
runs/run-<UTC timestamp>/
├── manifest.json
├── events.jsonl
├── frames/<frame SHA-256>.png
├── decisions.csv
├── transitions.json
└── summary.json
```

The event log is the source of truth. Derived CSV, graph, and summary files can
be rebuilt at any time:

```bash
lolo-log summarize --run runs/run-20260808T120000.000000Z
```

## Captured events

Telemetry includes:

- run, emulator, and frozen-parameter audit lifecycle events;
- environment resets and attempt numbers;
- every real controller action, duration, source frame, target frame, and pixel
  change, including actions explored on rejected save-state branches;
- anonymous save, load, and release events for every opaque state capability;
- all final latent planner candidates and their model scores and uncertainty;
- each real verified branch with a stable branch ID, candidate rank, action,
  plan, novelty, prediction error, visual change, penalty, total score, and the
  exact `env_step` event that produced it, plus matched-neutral action-effect
  contrast, learned estimate, sample count, and planning bonus;
- archive insertions, restorations, pruning, and restoration reasons;
- delayed visual-return detections, loop length, prior visit, every credited
  decision and scene/action/duration choice, plus unavailable recoveries;
- autonomous-motion detections and neutral grace waits across short pauses;
- rejected autonomous-motion hypotheses when the behavioral state already has
  a learned controlling action;
- learned-hazard filtering at commit and archive restoration, plus rejected
  archive insertions and the exact temporal-option evidence responsible;
- temporal-option starts, every passive continuation, controllable endpoints,
  credited initiating state/action/duration choices, same-duration
  counterfactual counts and pixel contrast, novelty/scene-span/duration/return
  score components, returns to the source or another previously known
  behavioral endpoint, exact-choice and action-level learned values, whether a
  negative sample met the global action-hazard criteria, and traces
  discarded at save-state jumps or run boundaries;
- delayed temporal-counterfactual reservation, every matched neutral step with
  source and target state aliases, factual-versus-counterfactual pixel
  contrast, endpoint contrast, and explicit state-release reasons;
- persistent-frontier successor-novelty updates, completed samples, loop
  penalties, provisional traces discarded at save-state jumps, and the value
  used to rank each archived branch, including state/action/duration samples;
- frozen-encoder abstraction assignments, latent distance, cluster creation,
  running cluster size, and the abstract signature used by each decision;
- interaction-derived behavioral-cluster assignments, matched controller
  probes, per-probe successor-latent distances, provisional-state deferrals,
  frontier-signature migrations, active-probe selection reasons, and prior
  probe-observation counts;
- committed decisions, temporary action, duration, and action-duration pair
  counts, scene streaks, archive size, exact-visual stagnation streaks, and
restored-branch status.

`summary.json` aggregates matched action-effect observations by controller
action, branches with a known effect estimate, hazard-filter events and choices,
archive hazard rejections, and negative samples that met the global hazard
criteria. The same fields remain available at branch/decision granularity in
`events.jsonl`; the most useful action-effect and coverage fields are also
materialized in `decisions.csv`.

Frame pixels are stored once under their digest, even if thousands of events
refer to the same screen. Short-lived planning save-state tokens are never
serialized; their lifecycle records use run-local aliases such as
`state-00000042`. After each committed decision, the evaluator also writes one
opaque, content-addressed core snapshot under `states/` and records it with
`decision_snapshot_stored`. The agent cannot read or parse those bytes. They
exist solely to make cross-session restoration constant-time, which preserves
the pixel-only observation boundary while avoiding replay of every rejected
planning branch.

## Evaluator bootstrap boundary

`lolo-neural-run --bootstrap lolo1-first-room` runs a minimal deterministic
controller macro before attempt 1. The fixture is accepted only for its known
ROM SHA-256 and only when its endpoint exactly matches the expected first-room
frame and coarse visual signature. `--bootstrap none` is the default, preserving
strict power-on evaluation.

The event stream records `bootstrap_started`, every `env_step` with
`phase=bootstrap`, `bootstrap_action_committed`, and `bootstrap_completed` under
attempt 0. `attempt_started` and `env_attached` then mark the pixel-exact handoff
to the agent as attempt 1. Summary fields report the fixture, action counts,
durations, and total emulator frames separately; bootstrap actions are excluded
from `investigated_actions`, `investigated_durations`, and per-attempt agent
statistics. The transition graph retains them with their phase-bearing source
events so a visualization can show or filter the complete session.

Both replay modes include the power-on-to-room bootstrap. The committed player
then follows only choices made by the agent, while the full player additionally
shows its rejected save-state branches.

`decisions.csv` is the convenient per-decision view. `transitions.json` contains
nodes for visual states and counted directed edges for all investigated
controller transitions. `summary.json` contains total and per-attempt counts,
committed versus investigated action distributions, archive restores, delayed
returns and recoveries, branches, states, unique frames, and unique coarse
scenes. The decision CSV exposes both the restore reason and whether a delayed
return recovery was pending after each committed emulator action. It also
records the successor-novelty reward and learned persistent-frontier value.
It also exposes the learned temporal-option value used for each committed
choice, whether that estimate came from the exact choice, an action-level
prior, or no evidence, and whether a passive option trace remained active after
the decision. Summary counts distinguish started, completed, discarded, and
credited option samples.

Matched-neutral exploration adds the following raw events and decision fields:

- `matched_neutral_verified` identifies evaluator-generated `NOOP` endpoints
  paired by root state and duration;
- `causal_observation_wait` records the neutral observation committed before a
  new intervention;
- `causal_spatial_signature`, `causal_changed_pixels`,
  `causal_change_centroid`, `causal_spatial_novelty`, and
  `causal_spatial_bonus` describe action-dependent screen changes;
- `causal_cell_coverage`, `causal_cell_unvisited`, `causal_cell_count`, and
  `causal_cell_coverage_bonus` describe global attempt-level coverage of the
  coarse cells changed relative to matched `NOOP`; the bonus is disabled by
  default and enabled with `--causal-cell-coverage-weight`;
- `causal_spatial_archive_bonus` records the live rarity value used when an
  archived branch is selected;
- `causal_cell_coverage_archive_bonus` records the corresponding live global
  coverage value used to rank an archived branch;
- `persistent_change_evidence_updated` records coarse cells whose visual value
  stayed different from their learned pre-change modal value for the configured
  number of committed observations, or later ceased to do so; baselines adapt
  before activation so moving sprites do not permanently imprint the first
  frame; `--persistent-change-minimum-value-drop` can restrict evidence to
  persistent disappearance of visually salient content without naming it;
- `persistent_change_archives_filtered` records temporary preference for
  archived states that preserve all active persistent changes, while
  `persistent_change_preservation_unavailable` makes the fallback to older
  alternatives explicit; this rule-free mechanism is disabled by default and
  enabled with `--persistent-change-stability-decisions`;
- `archive_branch_rejected` distinguishes covered causal frontiers, exhausted
  causal outcomes, non-causal alternatives, and learned hazards;
- `archive_causal_outcome_added` records a retained persistent visual
  transition, while `causal_outcome_exhausted` is the rejection reason for an
  already-restored coarse-pixel and directional-pose key.

`summary.json` aggregates matched-neutral verifications, total and unique
causal-spatial observations, unique committed causal signatures, first-visited
causal cells, and global coverage score/bonus totals.
`decisions.csv` includes the corresponding per-commit fields so a visualization
can map where action-dependent changes occurred without adding semantic object
labels.

On the explicitly labelled assisted reward track, semantic archive search adds:

- `human_prior_graph_source_signature` and
  `human_prior_graph_target_signature`, stable keys containing the visible
  goal state, detected player tile, HUD life glyph, and reversible world
  context;
- `human_prior_chest_obtained`, which persists the directly observed treasure
  contact milestone across later commits, archive restores, and episodic
  resumes;
- `human_prior_world_source_context`,
  `human_prior_world_target_context`, and
  `human_prior_world_effect_signature` on verified, archived, restored, and
  committed transitions. Ordinary single-step directional movement is
  represented by the graph position and never also toggles adjacent coarse
  sprite-spill cells into the world context, including blocked presses whose
  detected anchor stays on the same tile but whose facing pixels change.
  Exact option search still receives the unfiltered multi-action observation
  for its stricter persistence, phase, player-mask, and action-control audits;
- `human_prior_world_effect_confirmation`, which records the candidate coarse
  effect, immediate-control acceptance, future outcome spread, confirmation
  observations, action, duration, and endpoint frame;
- `human_prior_semantic_frontier_override`, which explains when a new assisted
  graph state is retained despite coarse causal-frontier deduplication; and
- `human_prior_best_first_archives_filtered`,
  `human_prior_best_first_frontier_exhausted`, and
  `human_prior_graph_stagnation_detected`, which make every semantic
  backtracking decision reconstructable;
- `human_prior_option_search_started`,
  `human_prior_option_neutral_verified`,
  `human_prior_option_local_neutral_verified`,
  `human_prior_option_branch_verified`, and
  `human_prior_option_search_completed`, which preserve every exact
  save-state sequence rollout, its duration-matched all-`NOOP` reference,
  action/duration path, parent and endpoint state IDs, player-masked
  non-player effect signature, conservative nonlocal effect cells,
  pixel-derived goal analysis, novelty counts, score, and selection. With
  `--human-prior-option-search-long-direction-frames`, each directional action
  additionally receives one long-press edge while buttons and neutral waits
  retain the base option-search duration. `action_duration_edges` records the
  exact expansion set, and each neutral event records `elapsed_frames` plus
  the heterogeneous duration tuple used for its matched reference.
  `parent_graph_signature` records the observed pixel-state parent of every
  endpoint. On an episodic resume, the parent field—or the ordered verified
  prefixes in older telemetry—reconstructs each emulator-observed
  prefix-to-prefix graph edge, its local controller-edge coverage, and
  temporary option coverage. Thus a multi-action macro has measurable
  progress at its verified intermediate states, while states whose controls
  were tested inside a longer option are not mislabeled as fresh control
  frontiers. This evidence remains available even when the option was not
  selected or committed, and the same exact experiment is not treated as
  unseen after a process restart. Every branch is also compared with both an
  all-`NOOP` rollout from the search root and a duration-matched `NOOP` from
  its immediate parent. `human_prior_option_action_dependent_endpoint`,
  `human_prior_option_local_action_dependent`, their visual-difference fields,
  the two neutral player slots, `human_prior_option_player_matches_neutral`,
  and `human_prior_option_causal_goal_reward` expose the result. Movement,
  novelty, shaping reward, graph progress, and graph edges that the matched
  neutral rollout reproduces are not credited to the controller. Local action
  coverage is still recorded, so an ineffective input is known to have been
  tested. Resume reconstruction applies the same rule to new telemetry and
  infers it from matching neutral frame digests where older logs contain the
  required counterfactual rollouts;
- `human_prior_episodic_graph_plan_selected`, emitted by the opt-in
  `--human-prior-episodic-graph-guidance` policy. It records the current
  pixel-state signature, reachable waypoint, plan kind, pixel gap, and verified
  action cost remaining to the waypoint. `milestone_route` and
  `milestone_bridge` target a component that previously preceded a positive
  visual outcome; `known_route` distinguishes an already connected route from
  investigating one missing bridge. When that room phase has no known positive
  outcome, `control_frontier` instead targets a different reachable state with
  locally untested controller actions. `frontier_actions` records those
  actions, but does not predict which one is useful. A state that is itself a
  historical milestone precursor never produces a zero-length milestone
  route: direct reward verification remains responsible for that outcome,
  while graph guidance may target an unfinished control frontier. Exhausted bounded-search
  states and stationary animation-only signature changes are ineligible. The
  policy reconstructs only emulator-verified source/target signatures and
  action-coverage counts from temporary telemetry; it does not store an object
  label, object rule, or supplied action sequence. Every exact branch, archived
  endpoint, restore, and commit carries
  `human_prior_episodic_graph_plan_kind`,
  `human_prior_episodic_graph_progress`,
  `human_prior_episodic_graph_bridge_reached`, and
  `human_prior_episodic_graph_remaining_cost`, so route reuse and the first
  newly connected component can be visualized decision by decision;
- `human_prior_episodic_milestone_sources_filtered` records when a historical
  positive-outcome source is excluded because every explicitly positive,
  observed heart-removal transition from that pixel state matches an
  empirically exhausted ordering. Ordinary graph edges that happen to cross a
  noisy goal-state signature cannot qualify a milestone source.
  The source becomes eligible again if that ordering hypothesis is later
  disproved. This prevents graph reuse from silently bypassing the same
  learned ordering constraint applied to direct reward and archive choices;
- `human_prior_archive_episodic_graph_revalidated` recomputes every archived
  endpoint against the graph plan active at restore time. Stored progress from
  the plan that originally created an archive is audit-only and cannot later
  make that archive look useful under an unrelated plan. Restore events expose
  both the live fields above and the corresponding
  `human_prior_episodic_graph_stored_*` values;
- `human_prior_option_search_depth_completed`, which makes beam loss and
  tracker failure directly auditable at every depth. It records raw
  candidates, globally deduplicated and novel candidates, detected- and
  missing-player counts, eligible tracker gaps, gap-streak rejections, retained
  detected and missing nodes, and the cumulative option-state count. Search
  state deduplication spans all previous depths. Missing-player endpoints have
  no novelty bonus, occupy at most
  `--human-prior-option-search-missing-player-reserve` slots after detected
  endpoints, and may not exceed
  `--human-prior-option-search-missing-player-max-streak` consecutive edges;
  `--human-prior-option-search-position-reserve` can reserve part of the beam
  for one representative of geometrically distinct detected player positions.
  Representatives favor larger displacement and lower visit counts, allowing
  a necessary temporary move away from the shaped goal to survive alongside
  high-reward candidates. The depth event records the candidate count,
  retained count, and exact reserved positions;
- `human_prior_option_world_effect_stability`, which replays a bounded sample
  of distinct option effects beside duration-matched all-`NOOP` controls at
  future horizons and records the intersected coarse cells, conservative
  nonlocal subset beyond a two-cell Manhattan guard around every detected
  player anchor, persistence ratio, safety checks, and every factual/control
  observation. The guard covers sprite spill plus a one-tile disagreement in
  the snapped detector; the local/nonlocal split prevents those pose pixels
  from being promoted as a stable world change. A maximum stable-cell
  footprint rejects layout-wide animation-phase differences. This audit is
  telemetry-only and does not affect endpoint selection;
- `human_prior_option_world_effect_phase_alignment`, which compares a
  localized candidate patch with nearby future all-`NOOP` frames. A match below
  the configured patch-L1 threshold classifies the observation as an animation
  phase shift and prevents it from reaching action-ablation controls or the
  assisted effect frontier;
- `human_prior_option_world_effect_action_control`, which is emitted only for
  a safe, localized persistent candidate and replays the sequence while
  replacing each action in turn with an equal-duration `NOOP`. It records
  which intervention positions remain causally necessary for the compact
  nonlocal effect. When the effect frontier is enabled, each confirmed control
  also emits `human_prior_option_effect_controllability_probe`: the factual and
  action-ablated endpoints must have both the same detected player location
  and the same nonempty pixel-level player footprint, then every directional
  action sequence through
  `--human-prior-option-effect-controllability-depth` is branched from both.
  The footprint gate prevents sub-tile movement or facing differences from
  masquerading as a changed world affordance. The event records the coarse
  endpoint match, footprint match, each footprint size and symmetric
  difference, configured depth, every exact action path, both reachable-position
  sets, newly reachable positions, and factual/control pixel-outcome spread.
  An effect enters the frontier only when the matched-footprint factual state
  adds at least one reachable player position;
- `--human-prior-option-causal-effect-frontier` is a separate opt-in assisted
  track for delayed consequences. It applies the same stability, phase,
  localization, safety, and leave-one-action-out requirements, but permits a
  confirmed effect when the bounded reachability probe has not yet exposed a
  new player position. It does not weaken or replace the immediate effect
  frontier's reachability requirement;
- with `--human-prior-option-effect-local-controls`, compact persistent effects
  rejected only by the conservative player-neighborhood mask also receive
  telemetry-only action ablations. Nearby cells are admitted only at horizons
  where factual and control player endpoints match exactly. A union of the
  factual/control player-palette footprints (plus a small outline halo) is
  removed before comparison, and each remaining coarse cell must contain at
  least `--human-prior-option-effect-local-minimum-cell-pixels` changed pixels.
  The event records both the ignored-pixel count and support threshold. These
  observations never enter the effect frontier;
- `human_prior_option_effect_frontier_eligible`, emitted when the opt-in
  assisted effect frontier is enabled, records the confirmed action indices
  and the reversible learned world-context transition. Its `frontier_reason`
  distinguishes `immediate_reachability_gain` from `delayed_causal_effect`,
  while `action_control_confirmed_indices` and
  `reachability_confirmed_action_indices` expose the two gates independently.
  Only safe, localized, persistent, phase-distinct,
  action-ablation-confirmed effects can enter either frontier;
- with `--human-prior-option-entity-frontier`, the same controlled rollouts
  also maintain temporary room-local appearance prototypes. For each
  intervention, telemetry records the pre-action facing ray, intersected
  effect cells, pooled factual/control appearance distance, and anonymous
  before/factual/control prototype IDs. A candidate emits
  `human_prior_option_entity_frontier_eligible` only when a persistent,
  phase-checked, leave-one-action-out-confirmed appearance change lies on that
  interaction ray. The resulting anonymous appearance hash becomes a new
  reversible graph context even when bounded player reachability has not yet
  changed. Remote display changes do not intersect the ray and cannot enter.
  Option-beam and effect-candidate deduplication preserve the action-derived
  facing direction, so visually similar same-tile states remain distinct long
  enough to test direction-dependent interactions. Anonymous absolute
  appearance hashes at action-changed cells also remain in the temporary
  option key, and `NOOP` becomes an allowed exact-option action on this
  opt-in track. This lets an interaction or display change survive long enough
  to test a later interaction without treating the intermediate change as
  reward. When one search verifies multiple distinct anonymous entity states,
  it retains one bounded save-state representative for each distinct learned
  world context instead of discarding all but the highest-scoring endpoint.
  The same representative retention applies to confirmed immediate and delayed
  causal-effect contexts, so the ordinary primary and a learned transformation
  can coexist in the bounded archive.
  `human_prior_option_archive_added.selected_primary` distinguishes the
  ordinary primary choice from additional causal alternatives;
  `human_prior_option_effect_frontier_reason` records why an effect alternative
  qualified. `human_prior_option_search_completed.distinct_effect_contexts_archived`
  and
  `human_prior_option_search_completed.distinct_entity_contexts_archived`
  record the numbers preserved by that search. Entity branches store the
  post-settling frame and emulator state used by the persistence/control
  audit, rather than the immediate possibly in-flight action frame.
  `human_prior_option_archive_added` records
  `human_prior_option_settling_steps`,
  `human_prior_option_settling_frames`, and
  `human_prior_option_immediate_frame`; the eligible event carries the same
  horizon and immediate-frame link, while its normal `frame` is the settled
  endpoint. Anonymous entity appearance features exclude the union of the
  detected factual/control player footprints before hashing, so facing pixels
  that overlap an affected coarse cell do not create false object states.
  `human_prior_option_world_effect_action_control.entity_player_masked_pixels`
  records the final mask size, and each horizon observation records its own
  `entity_player_masked_pixels` count;
- with nonzero `--human-prior-option-entity-curiosity-weight` or
  `--human-prior-option-entity-curiosity-reserve`, every exact branch records
  the anonymous target appearance fingerprint, learned type when matched,
  relative context, prior sample count, posterior novelty, within-frame spatial
  rarity, and combined curiosity. The reserve retains distinct under-tested
  appearance/action tuples in the search beam. The bounded matched-control
  probe set ranks one uncertain interaction per spatial locus before spending
  a second probe on another action at the same locus, so one visually novel
  patch cannot consume the whole audit budget. Each probe reports
  `distinct_interaction_groups_available` and
  `interaction_group_reserved`.
  `human_prior_option_entity_curiosity_probe` links the reserved tuple to its
  leave-one-action-out result and behavior evidence, including static/no-effect
  outcomes. Prefix replay carries the same prior-position reference used by
  exact search. `interaction_cell` and `audited_interaction_cell` expose both
  sides of that attribution, while `interaction_cell_matched=false` makes the
  row ineligible for learning or entity-frontier promotion. Summary telemetry
  reports curiosity branches, retained beam slots, probes, known predictions,
  and accepted learning evidence. Both settings are part of
  `search_budget_sha256`, so enabling curiosity reopens a source that was
  exhausted under the policy-neutral budget;
- with nonzero `--human-prior-option-entity-inert-penalty-weight`, exact
  branches additionally record semantic sample coverage, learned inert and
  measured-effect probabilities, evidence confidence, and the exact score
  subtraction. `anonymous_entity_behavior_observed` stores the full
  pixel-derived outcome descriptor and before/after semantic posteriors.
  Summary telemetry counts semantic/inert observations and penalized branches.
  `anonymous_entity_predicted_inert_penalty` preserves the learned prior while
  `anonymous_entity_inert_penalty`, `_eligible`, and `_suppressed` record the
  applied decision after current verified effects take precedence.
  Legacy checkpoints remain readable but opaque outcomes contribute no penalty
  until a matching descriptor is observed. The weight is included in
  `search_budget_sha256`;
- assisted best-first recovery prefers a confirmed
  `immediate_reachability_gain` effect over an ordinary physical frontier, but
  does not give the same precedence to a `delayed_causal_effect` hypothesis.
  `human_prior_best_first_archives_filtered` records
  `immediate_option_effect_frontier_preferred`, the number of
  `immediate_reachability_option_effects`, and the selected archive's effect
  reason so the distinction is auditable;
- long-lived recovery and archive entries own independent emulator handles.
  In particular, `archive_affordance_checkpoint_added.state_id` identifies a
  cloned source state, while `parent_state_id` records the original decision
  root. This prevents a life-hazard checkpoint release from invalidating an
  affordance archive that describes the same visual source;
- `decision_committed.target_pose_action` records the action-derived facing
  associated with the committed emulator state. Episodic resume restores this
  temporary pose directly when present; older logs are supported by replaying
  the executed action for live decisions and the full path for restored
  archive decisions. `episodic_human_prior_memory_seeded.pose_action` exposes
  the reconstructed value used by the resumed interaction planner. A source
  archive restore that reached an unfinished local-control frontier also
  preserves its bounded navigation-recovery grace across the process
  boundary; `navigation_recovery_grace_restored` and
  `navigation_recovery_grace_elapsed_decisions` expose that continuity;
- `human_prior_option_search_deferred` and
  `human_prior_option_search_skipped`, which distinguish a cheaper unseen
  local archive endpoint from an already exhausted sequence-search source.
  Exhaustion is qualified by the actual search budget: depth, beam width,
  missing-player reserves, action/duration edges, effect-probe settings, and
  entity-frontier settings. `search_budget_sha256` links start, completion,
  and skip events without treating the opaque hash as policy input.
  `human_prior_option_search_reopened` records when the same semantic pixel
  source is searched again under a different budget; a bounded depth-4 result
  therefore cannot permanently suppress a later depth-6 experiment. Summary
  telemetry counts these budget reopens. Endpoint eligibility treats an exact
  unvisited graph state as novel even when its raw player coordinate was seen
  in another heart set or world context. The branch event exposes both
  `target_graph_state_visits` and `target_player_position_visits`, making that
  distinction auditable. A completed search with no unexpanded endpoint also
  records a room-local bounded topological observation. A search that merely
  finds only globally visited endpoints does not make this stronger claim.
  Subsequent exact
  searches emit `human_prior_option_exhausted_frontiers_filtered` when they
  decline to archive a route back into that state, and one-step selection
  emits `human_prior_exhausted_option_frontier_filter_evaluated`. Milestones
  and changed world contexts remain eligible; one-step selection fails open
  if no alternative graph egress remains. When every exact endpoint from a
  source is already exhausted, the bounded observation propagates backward to
  that source; this computes cul-de-sacs from interaction rather than object
  labels. Each observation records the maximum exact-search depth that
  supported it. A deeper search reopens shallower claims instead of filtering
  them, while equal-or-deeper evidence remains active. The observation is
  replayed across save-state
  resumes, cleared at room boundaries, and withdrawn when a later search from
  the same state finds a retainable endpoint. Telemetry explicitly reports
  `policy_effect=bounded_frontier_avoidance` and `hazard_evidence=false`;
  after a bounded option search returns no usable endpoint,
  `human_prior_option_exhaustion_egress_filter_evaluated` prefers verified
  actions that actually change the current pixel-derived graph state over
  non-moving actions. It reports every suppressed branch and fails open when
  no graph-changing action exists. This is a decision-local escape policy,
  not a persistent hazard label or a supplied route;
- `human_prior_option_recovery_armed` and
  `human_prior_option_recovery_deferred`, which record whether a branch added
  by the current exact search may be restored in the same decision. Immediate
  recovery requires positive verified root-relative assisted goal progress or,
  when episodic graph guidance is explicitly enabled, lower verified action
  cost to its selected waypoint. Ordinary neutral or regressive endpoints stay
  in the archive for later best-first exploration but cannot preempt the normal
  planner. Option archive
  `goal_progress_reward` is signed total assisted progress rather than an
  unsigned milestone-only bonus;
- semantic recovery does not discard a confirmed physical, world-effect,
  entity, or control frontier merely because its coarse scene signature equals
  the current frame. Such entries still pass through the normal downstream
  frontier-exhaustion and ranking filters, making localized reversible changes
  recoverable without treating all same-scene archives as novel;
- with `--human-prior-goal-exhaustion-rollback`,
  `goal_milestone_checkpoint_created.checkpoint_source` distinguishes an exact
  retained `archive_parent` from a `matching_current_parent`. An old archive
  whose parent was not retained emits
  `goal_milestone_checkpoint_unavailable` and cannot become a rollback target;
  `goal_milestone_exhaustion_learned` and
  `goal_milestone_exhaustion_state_restored` record the opt-in assisted
  preparation loop. Rollback requires a repeated semantic state, no
  recoverable archive frontier, an exact option search that adds no endpoint,
  and (by default) at least 16 committed post-milestone decisions. Before that
  threshold, `goal_milestone_exhaustion_deferred` records the bounded search
  result and the remaining evidence budget without changing policy. The
  threshold is configurable with
  `--human-prior-goal-exhaustion-minimum-steps`. A verified new graph state,
  player position, or world context emits
  `goal_milestone_exhaustion_progress_reset` and restarts this consecutive
  no-progress clock. The exhaustion event preserves the exact milestone
  choice, graph/position coverage, exhausted option sources, pre-milestone
  state ID, descendant invalidations, and restored pixel goal state. Rollback
  also requires an explicitly known, changed source/target heart set; legacy
  checkpoints without that metadata remain valid for observed life-loss
  recovery but cannot assert bounded exhaustion. On episodic import,
  `goal_target_heart_slots_source=legacy_decision_telemetry` means the loader
  recovered the target only after matching the opaque checkpoint's source
  frame, behavioral source, controller edge, and source heart set to an exact
  ancestral `decision_committed` event. A partial match is never accepted. The
  event records the
  observed heart-set transition as a soft preparation-ordering hint; it does
  not create a negative temporal-option hazard sample. Those samples remain
  reserved for observed loss or causal recoverability evidence. Legacy
  unqualified goal-exhaustion values are counted by
  `episodic_human_prior_memory_seeded.unqualified_exhaustion_hazards_ignored`
  and are not restored into policy. The compatibility event name remains
  `goal_milestone_exhaustion_learned`, while
  `hazard_evidence=false`, `policy_effect=milestone_priority_only`, and
  `preparation_transition_learned` make the semantics explicit;
- `goal_milestone_checkpoint_snapshot_stored` persists that opaque
  pre-milestone capability without exposing its bytes or metadata to policy.
  `episodic_goal_milestone_checkpoint_state_imported` records restoration in a
  child evaluator, and `checkpoint_source=episodic_resume` distinguishes it
  from a parent retained in live memory. With a positive
  `--human-prior-goal-exhaustion-frontier-budget`,
  `goal_milestone_frontier_budget_exhausted` bounds post-milestone exploration
  even when an expanding semantic archive would otherwise postpone rollback,
  but it cannot bypass the minimum exploration requirement. The preparation
  hint is scoped to the visible heart-set transition—including the valid empty
  set after the last heart—so later planning deprioritizes the same collection
  order without hard-vetoing the controller action.
  `human_prior_exhausted_milestone_filter_evaluated` records each contextual
  policy evaluation, the exact source/target heart sets, candidate controller
  edges, delayed precursor endpoints, available preparation alternatives,
  filtered count, and fail-open status. A delayed precursor retains the source
  heart set but places the pixel-detected player on a goal slot whose later
  disappearance produced the exhausted transition. Filtering occurs only when
  another verified non-loss branch changes semantic player/world state or
  reaches a different milestone; otherwise the hint fails open.
  `archive_branch_rejected.reason=exhausted_milestone_ordering` prevents direct
  verification from storing the same exact or precursor endpoint before commit
  selection. `human_prior_option_ordering_endpoint_rejected` applies the same
  rule to depth-search endpoints, and
  `human_prior_exhausted_milestone_archives_filtered` removes compatible
  seeded or older in-memory archives before restore.
  `human_prior_preparation_archives_preserved` records neutral archive branches
  allowed to retain the current pre-milestone heart set even when the
  historical best remaining-heart count is lower; a player endpoint on the
  failed first-goal slot is not eligible for that exemption. These policies do
  not weaken life-loss or causal entity-hazard filters. Summary fields
  aggregate filter evaluations, exact and precursor branches, fail-opens,
  option rejections, filtered archives, and preserved preparation archives;
- `human_prior_option_milestone_settled` links the immediate action endpoint
  to the stable frame and state used for milestone analysis and archival.
  `human_prior_option_milestone_candidates_collapsed` reports the reduction in
  equivalent candidates before that replay cost is paid.
  `human_prior_option_milestone_settlement_rejected` records a transient goal
  signal that did not survive settling, while
  `human_prior_option_milestone_duplicate_rejected` records a later path to an
  already committed semantic outcome. `human_prior_milestone_outcome_recorded`
  identifies the first committed transition or archive restore that established
  that outcome for the current room. These outcome records are included in
  resume-chain memory; `episodic_human_prior_memory_seeded.milestone_outcomes`
  reports how many were reconstructed. Reconstruction pairs each outcome with
  `(run_id, decision)` so local decision numbers cannot collide across chained
  runs. Direct one-step verification exposes the same memory through
  `branch_verified.human_prior_milestone_outcome_known`,
  `human_prior_milestone_reward_suppressed`, and
  `human_prior_effective_goal_reward`. The repeated milestone loses only its
  already-observed assisted bonus; navigation and life-loss terms keep their
  signs. `human_prior_known_milestone_frontier_choice` records selection of an
  unvisited semantic player endpoint before repetition, while
  `human_prior_known_milestone_fallback` records deliberate repetition when no
  such endpoint remains. This makes necessary repeated milestone actions
  possible without letting blocked animation changes masquerade as semantic
  progress;
- exact option-search depth telemetry reports
  `repeated_milestone_candidates` and
  `repeated_milestone_parents_retained`. Previously observed or exhausted
  heart-set transitions remain auditable endpoints but are not expanded as
  beam parents, preserving capacity for alternative collection orders;
- navigation shaping is endpoint-novelty gated. `branch_verified` records
  `human_prior_effective_navigation_reward` and
  `human_prior_navigation_reward_suppressed`, plus target graph-state visits,
  phase-position visits, and unexpanded controller actions. A repeated
  position therefore cannot earn the same geometric distance reward forever.
  When bounded post-milestone exploration has learned that one collection
  ordering is exhausted, `human_prior_navigation_retargeted` records a
  temporary, room-local retarget to the other still-visible pixel slots.
  `human_prior_navigation_failed_targets`,
  `human_prior_navigation_active_targets`, the ordering-specific source and
  target distances, and `human_prior_navigation_ordering_reward` expose the
  complete inference for replay and visualization. This memory neither names
  an object nor supplies an action sequence; after an alternate slot is
  collected, its changed visible-goal set stops matching the failed ordering
  and ordinary navigation resumes automatically. `decisions.csv` contains the
  same fields, while `summary.json` counts retargeted evaluations, exact-search
  branches, commits, and the total committed ordering reward. A verified
  option with positive ordering-adjusted progress remains archive-eligible at
  a previously visited graph state; visibility novelty is not allowed to erase
  a useful detour learned from the room's transition topology.
  `human_prior_ordering_progress_recorded` starts a bounded alternate-order
  trial. If a later exact search from that progress frontier yields no
  retainable endpoint, `human_prior_ordering_hypothesis_disproved` preserves
  the original exhaustion observation but removes its policy authority,
  discards archive scores derived from it, and permits the formerly excluded
  milestone to be reconsidered. During that bounded reconsideration,
  `human_prior_navigation_reconsidered*` fields allow positive progress toward
  only the reopened slot to survive the global visited-state filter; unrelated
  visited states remain ineligible, preserving the anti-loop gate. A later
  independently exhausted trial can reactivate the hypothesis. These events
  explicitly report search budget,
  failed and alternate slots, discarded archives, `hazard_evidence=false`, and
  policy effect; `summary.json` counts progress trials, disproofs, discarded
  archives, and reactivations. A disproof also records the exact search-budget
  digest and its depth, beam width, and position reserve. Resume invalidates
  that negative conclusion when any of those exploration capacities becomes
  stronger, retains the factual exhaustion observation, and retries the
  alternate ordering; `budget_invalidated_ordering_disproofs` makes that
  memory revision explicit.
  Every verified source/action edge is also reconstructed from the resume
  chain, whether or not that branch was committed. Thus "unexpanded" means
  that a controller action has not yet been tested from the semantic source,
  rather than merely that it has not won selection.
  `human_prior_semantic_frontier_choice` records when a repeated graph state
  selects a new player endpoint or a least-visited endpoint whose outgoing
  actions still need expansion. `human_prior_graph_recovery_suppressed` records
  the configured local-expansion grace after navigation, and archive restores
  report whether the detected player position changed and armed that grace.
  `human_prior_navigation_grace_armed`,
  `human_prior_target_position_visits_before`, and
  `human_prior_target_unexpanded_actions` distinguish a useful local frontier
  from a position change back to an already verified endpoint.
  `human_prior_best_first_archives_filtered` separately reports physical,
  unseen-world-state, and target-control frontiers. A stable action-dependent
  world change remains eligible even when sprite overlap prevents the assisted
  player tracker from reporting movement. Deep option search logs
  `global_semantic_archive_frontier_available` when it defers to such a saved
  branch. Once an endpoint's position, world state, and outgoing controls are
  all covered, `human_prior_semantic_archives_exhausted` records removal and
  release of that now-terminal save-state capability;
- each exact search can retain multiple globally novel semantic endpoint
  representatives, bounded by
  `--human-prior-option-archive-representatives` and the global archive
  capacity. `semantic_state_representatives_available` and
  `semantic_state_representatives_archived` report the result. When the
  position-diversity reserve is enabled, the same representative budget first
  preserves a terminal endpoint at a spatially distinct player position;
  `position_representatives_available`,
  `position_representatives_archived`, and
  `human_prior_option_archive_position_representative` expose that choice.
  The representative maximizes distance from the selected endpoint before
  considering distance from the source, and its selected-player anchor and
  Manhattan divergence are logged for replay and visualization.
  Confirmed
  replay-stable causal-effect frontiers remain first-class and are preferred
  over ordinary movement frontiers during best-first recovery;
- `human_prior_option_archive_added`, plus
  `human_prior_verified_option`, `human_prior_option_depth`, and
  `human_prior_option_path_visits_before` on restore/commit events, which make
  whole-sequence selection and replay explicit.

These fields are experimental assisted-track telemetry. They do not expose ROM
memory and are not present in the strict policy state. `summary.json` rolls up
confirmation/acceptance counts, unique effect signatures, committed world
contexts, graph states, player positions, semantic overrides, best-first
filter/exhaustion events, graph-stagnation events, sequence searches,
deferrals, cached skips, verified option branches, archived endpoints, and
committed options. `decisions.csv` carries the graph/world-context keys and
selected-option fields on every commit.

`pixel_novel_room_started` marks the first planning boundary after the generic
pixel-only dark-transition detector resolves to a visually novel bright scene.
It records the frame, newly discovered assisted heart slots, reset graph
signature, and new known-scene checkpoint. Screen-coordinate coverage,
persistent-change evidence, and assisted per-room graph counters are reset at
this boundary; frozen model parameters and reusable learned dynamics are not.
`summary.json` reports `pixel_novel_rooms_started`.

## Unlabelled entity audit

`lolo-entity-audit` derives a telemetry-only patch representation from stored
committed frames:

```bash
lolo-entity-audit runs/<run-id> --output entity-audit.json
```

It partitions each 256×240 screen into a 16×15 grid, pools each cell into a
quantized 4×4 RGB feature, and incrementally clusters similar patches. The
output preserves the prototype grid per decision, prototype occupancy and
spatial rarity, persistent rare-patch state, state transitions, directional
action-target patches, and repeated interaction-edge counts. Prototype IDs
carry no object names or rules and never enter the policy unless a later
experiment explicitly enables that use.

## Frozen spatial-model telemetry

Supplying `--spatial-shadow-checkpoint` adds predictions from the spatial model
without changing the existing planner by default. Every `planner_candidates`
row includes the counterfactual-usefulness score, raw predicted-activity score,
predicted action-versus-NOOP pixel and effect differences, predicted pixel
change, predicted effect, uncertainty, mode, selection weight, and weighted
priority bonus. Each real save-state branch also emits
`spatial_shadow_branch_evaluated`, linked by decision, branch ID, candidate
rank, action, and duration. It records:

- pixel and effect-weighted prediction error;
- the corresponding frame-persistence baselines;
- whether the prediction beat persistence;
- spatial-effect L1 and F1;
- predicted versus observed effect mass; and
- predicted pixel-change magnitude;
- predicted action-versus-duration-matched-NOOP pixel and effect differences;
- the raw activity and counterfactual-usefulness scores;
- actual action-effect contrast against the matched verified NOOP branch; and
- ensemble uncertainty.

`spatial_shadow.csv` provides one flat row per verified branch for plotting.
`summary.json` reports evaluation count, persistence wins, mean metrics, and
whether the separate spatial parameter-hash audit passed. It also records
whether selection was enabled, the configured weight, and the mean selection
bonus. The run manifest stores checkpoint file and parameter hashes plus the
mode and weight.

`--spatial-selection-weight N` enables a controlled frozen-model ablation. The
counterfactual-usefulness score is multiplied by `N` to prioritize which
save-state candidates are verified first. It is not added to the final
verified-branch commit score: once real outcomes are available, the existing
outcome-based objective decides which branch is committed. No model parameters
are updated, and verified-branch error is never fed back into the current
decision. Candidate, branch, and committed-decision telemetry retain the raw
score, weight, bonus, mode, and `spatial_selection_applied_to_commit` flag.
Omitting the option, or setting it to zero, preserves observational mode
exactly.

Supplying `--spatial-returnability-checkpoint` with the exact spatial
checkpoint it was trained against adds two observational fields to candidate
and verified-branch rows:

- `spatial_shadow_predicted_returnability` is the ensemble mean probability of
  an observed visual-state return path within the training horizon;
- `spatial_shadow_returnability_uncertainty` is the variance across relation
  heads.

The sidecar is parameter-hash bound to its frozen spatial encoder and cannot be
loaded with another encoder. The manifest records both checkpoint hashes and
the pixel-graph target configuration. `summary.json` reports mean probability,
mean uncertainty, and whether `spatial_returnability_parameter_audit` proved
that evaluation left the sidecar unchanged. Returnability has no selection or
commit weight; the current checkpoint is research telemetry, not a reward.

## Bidirectional save-state probes

`--returnability-probe-depth N` enables explicit, telemetry-only endpoint
experiments. For every verified branch, the collector restores its save state,
tests all configured controller actions at the branch duration, and retains the
closest `--returnability-probe-beam-width` endpoints for the next depth. A
candidate is compared with the frame produced by restoring the original root
and applying NOOP for the same total emulator time. This matched control keeps
ordinary animation from masquerading as irreversibility.

The append-only event stream includes:

- `bidirectional_probe_started` with the actions, depth, beam, threshold, state
  aliases, and source pixels;
- `bidirectional_probe_reference` for every duration-matched NOOP frame;
- `bidirectional_probe_step` for every tested path, including state aliases,
  emulator event links, pixel distance, exact-match status, and return result;
- `bidirectional_probe_completed` with path coverage, shortest observed return,
  best distance, and `no_return_within_probe_budget`.

All generated emulator steps use phase `returnability_probe`. The standard
experience importer excludes them, and branch scores, attempt memory, and
persistent parameters never see their outcomes. `returnability_probes.csv`
provides one row per path. `summary.json` reports probed branches and paths,
return counts, budget-scoped non-returns, mean best distance, and the artifact
location. “No return” always means within the logged action/depth/beam budget;
it is not asserted as a universal property.

When a probe run resumes from assisted-policy telemetry, its manifest inherits
assisted provenance as `human_prior_resume_observational`. Such a run can be
used for evaluator validation but cannot enter the strict dataset.

### Explicit-probe import

`lolo-spatial-probe-returnability-train` takes separate repeated
`--training-run` and `--validation-run` arguments. Before decoding examples it
requires a complete manifest, content-addressed pixel frames, the requested
reward track, matching probe settings, one start/completion/verified-branch
lifecycle per label, internally consistent return evidence, and valid source
and endpoint digests. Conflicting labels abort the import.

Transitions are deduplicated within each partition. Exact source, endpoint,
action, and duration overlap is removed from validation first; any remaining
example whose source pixels occur in training is removed as well. Both counts
are reported in the metrics provenance. Both remaining partitions must contain
positive and budget-scoped negative labels. Training is balanced by label;
validation keeps the natural prevalence so ROC AUC, Brier error, majority
accuracy, and calibration are not distorted by a tiny balanced sample.
Manifests and event logs are hashed into the resulting metrics file.

## Attempts and level labels

An attempt begins whenever the environment is reset. Because room number,
completion, death, and object identity are deliberately excluded from the agent
interface, the logger does not invent semantic room boundaries. An evaluator
can add labels after the run without exposing them to the agent:

```bash
lolo-log annotate-level \
  --run runs/run-20260808T120000.000000Z \
  --label lolo1-withheld-room-07 \
  --start-seq 500 \
  --end-seq 1900 \
  --attempt 1

lolo-log summarize --run runs/run-20260808T120000.000000Z
```

These labels live in `evaluator_annotations.jsonl`, separate from the immutable
agent event stream. Rebuilding the summary adds the matching label to each
decision row. This separation lets later visualizations group attempts and
states by a human or evaluator-known room while preserving the pixel-only
research boundary during play.

Use `--no-frame-images` for digest-only profiling runs. The default keeps PNGs
because they make state timelines and transition-graph inspection much easier.

## Anonymous entity behaviors

When the anonymous behavior sidecar is enabled, each rare passive patch and
each controlled interaction produces an
`anonymous_entity_behavior_observed` event. The record contains an anonymous
type ID and appearance fingerprint, context signature, action and duration,
predicted and observed outcome hashes, evidence count, probability, entropy,
confidence, surprise, hazard probability, relative effect cells, and the
deduplicated evidence ID. It never contains a supplied sprite or mechanic name.

`anonymous_entity_passive_scan_completed` records how many rare patches were
tracked through each matched passive interval and how many controlled-sprite
cells were excluded. Each additional configured duration-only save-state
branch emits `anonymous_entity_passive_horizon_verified`, including the root
state, frame duration, resulting frame, and linked environment-step sequence.
`anonymous_entity_causal_horizon_verified` records a matched pair consisting
of one verified intervention endpoint and its equal-duration neutral control,
both advanced by the same wait horizon. The associated
`anonymous_entity_causal_contrast_completed` event reports relation-changing
candidates, newly localized candidates, terminal and life-loss contrasts, and
the number that received local attribution.
`anonymous_entity_behavior_checkpoint_updated` binds a clean learning run to
before/after parameter digests. Frozen evaluation instead emits
`anonymous_entity_behavior_parameter_audit` and fails if the digest changes.

`anonymous_entity_behavior_shadow_prediction` records one rare patch at one
future horizon for a verified endpoint. It includes empirical, unconditional,
and causally attributed hazard probabilities, causal support, relative context,
and the simulated veto verdict. `anonymous_entity_behavior_shadow_branch_evaluated`
aggregates these rows per endpoint. Summaries flatten them to
`entity_behavior_shadow.csv` and `entity_behavior_shadow_branches.csv` and join
supported predictions to matched causal contrasts for TP/TN/FP/FN counts.

When policy authority is explicitly enabled,
`anonymous_entity_hazard_veto_evaluated` records every detected endpoint,
supporting anonymous identity/relation/horizon, how many branches were actually
filtered, alternatives remaining, and whether the all-hazard fail-open fired.
The committed decision repeats the veto status and its own predicted hazard.

`lolo-log summarize` writes all behavior observations to
`entity_behaviors.csv` and adds counts for accepted evidence, known predictions,
prediction matches, known hazard predictions, hazard-classification matches,
observed hazards, terminal observations, passive-horizon branches, anonymous
types, and distinct outcome signatures to `summary.json`. Causal summaries add
matched horizons, contrasts, hazard contrasts, localized candidates, behavior
attributions, and hazard attributions. Causally attributed rows in
`entity_behaviors.csv` include the intervention action and duration, branch
role, other-branch hazard state, and first localization horizon. See
`anonymous-entity-behavior.md` for the learning and freeze protocol.
Terminal rows from the ordinary passive scanner carry
`evidence_eligible=false`; the summary counts them as
`anonymous_entity_behavior_terminal_evidence_withheld`.

## Episodic resume

Use a state the agent previously reached without replaying a hand-authored
controller sequence:

```bash
lolo-neural-run \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --checkpoint checkpoints/cycle-000010.pt \
  --resume-run runs/<parent-run> \
  --resume-decision 879 \
  --decisions 500
```

The child run records `episodic_resume_completed` and stores the parent run ID,
decision, source location, and source-event SHA-256 in its manifest. Replay
validates that hash and imports the decision's content-hashed opaque snapshot;
legacy runs without snapshots retain the exact full-event reconstruction
fallback. Gameplay-only resumes exclude `START` and `SELECT` from the agent
action set.

If a legacy run was recorded by a protocol-compatible older native host,
`--allow-compatible-resume-host` permits a one-time migration despite the host
file digest changing. It does not relax ROM or core validation, and the legacy
fallback still verifies every replayed frame before the first new snapshot is
written. Normal resumes keep strict host-digest validation.

`episodic_human_prior_memory_seeded` records reconstruction of the assisted
track's temporary graph-state visits, player-position visits, graph-edge and
verified-option coverage, learned world context, exact milestone-exhaustion
values, and pixel-derived goal memory across the recursive resume chain.
Logged `pixel_novel_room_started` boundaries discard earlier rooms' temporary
graph, position, edge, and option counters during reconstruction.
Current hearts, player, and life are anchored to the resumed pixels whenever
the latest source decision is strict; stale semantic fields from an older
assisted ancestor cannot overwrite the current save state. These are
evaluator-legal episodic counters; the frozen neural and spatial parameters
remain unchanged.

Promoted option alternatives also have evaluator-owned persistence. Only the
bounded alternatives admitted to the live archive are exported; ordinary beam
nodes are not. `option_archive_snapshot_stored` links a content-hashed opaque
checkpoint to its anonymous live state ID and is marked `agent_visible=false`.
When resuming, archive add/restore/remove events are replayed only through the
chosen `decision_snapshot_stored` boundary. Still-active alternatives are
imported without exposing their bytes to the policy, reseeded into the bounded
archive, and copied into the child run so a later descendant remains
self-contained. `episodic_option_archive_state_imported`,
`episodic_option_archives_seeded`, and
`episodic_option_archive_skipped` make this lifecycle auditable. Alternatives
that contain an uncommitted goal milestone are conservatively skipped until
their independent rollback-parent checkpoint can also be persisted. Live
pixels are verified before and after every export/import, and ordinary state
save/release balance remains explicit. An imported `state_saved` event carries
`imported_option_archive`, `option_archive_state_file`, and its SHA-256 so both
legacy decision reconstruction and full high-speed replay bind the state alias
to the exact persisted alternative while leaving the live frame unchanged.

## Deterministic high-speed playback

The event stream contains enough information to reconstruct the entire native
emulator session, including anonymous save, load, release, and controller-step
operations. Render both replay views with:

```bash
lolo-replay \
  --run runs/<run-id> \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --mode both \
  --speed 120
```

The renderer verifies the ROM, core, and host binaries against the run manifest,
recreates state handles in event order, and expands every multi-frame controller
press into individual NES frames. It checks every reconstructed endpoint
against the original telemetry and fails on the first divergence.
For an episodically resumed run it first verifies and replays the content-hashed
parent log, so the same integrity check covers the provenance boundary.

Two standalone local players are generated:

- `replays/committed/index.html` follows only actions ultimately chosen by the
  agent, with explicit frames for resets and archive-restoration jumps;
- `replays/full/index.html` shows every planning branch, state restoration,
  action frame, and committed-decision marker in chronological order.

Both players are scrub-capable and offer 5, 15, 30, 60, 120, and 240 fps. Once
rendered, playback requires only the HTML and its content-addressed PNG folder;
the emulator and ROM are not read by the browser. `replay_manifest.json` records
the verification result and exact input hashes.
