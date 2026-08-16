# Strict Room 3 collection recon — provenance verdict (2026-08-16)

Status: recon complete; collection BLOCKED on a claim-boundary decision
Context: closes the tracker coverage gap quantified in
`docs/tracker-ood-eval-2026-08-16.md` / learnings §4.32.

## Verdict

Resuming any run from the v318/v324 Room 3 states records reward track
`human_prior_resume_observational` (assisted ancestry folded into the
track at `neural_run.py:2248-2256`), and the strict import path refuses
it (`experience_import.py:25-33`, `:237-241`). The partition manifest is
not the gate; the reward-track machinery is. A strict power-on run cannot
reach Room 3 (only a Room 1 bootstrap fixture exists; strict play has
cleared Room 1 and explored but not solved Room 2). The target Room 3
configurations exist only in the assisted lineage — by definition.

**Therefore: Room 3 tracker-training data requires ratifying a policy —
may strict-policy collection branched from an assisted-era save state
enter the strict store (with ancestry disclosed)?** Mechanically ~2 lines
(a `strict_from_assisted_state` track value on the strict side of
`classify_reward_track`, plus `strict_lineage` allowlisting), but it is a
claim-boundary concession: §4.31 reserved assisted-era *telemetry* for
evaluation; this extends assisted-era *states* to collection roots.

Options:
- (a) Ratify `strict_from_assisted_state` (recommended): the policy is
  strict, frames/actions/controls are detector-free, ancestry stays
  visible in `episodic_resume` and the claim text discloses "collection
  roots include states reached by assisted play during development."
- (b) Refuse: strict store stays maximally pure; Room 3 palette coverage
  waits until strict play reaches Room 3 itself; only the
  duration-diversity axis is closable now (legal today from strict
  floor2 roots).
- A laundering loophole exists (`--resume-state-run`'s track is never
  consulted) and MUST NOT be used; fix task filed.

## Verified design (executable once the decision lands)

- Duration-pinned sub-runs are mandatory: the NOOP control anchor sits
  only at max(duration_choices), which is why every existing labeled arm
  is duration 16. Pin `--action-durations` per sub-run (1,2,4,8,16).
- `--verify-actions 9` mandatory or all control probes silently drop
  (`neural_planner.py:19402`).
- Strict runs never enter the assisted option search; label density comes
  from the probe machinery at default planning depth — depth/beam
  irrelevant.
- Two roots × 5 durations; pilot 12 decisions/sub-run (~30–40 min total,
  ~1% corpus growth) then a 200-decision tier (~2 h, ~12k arms, ~13% of
  corpus — needed because sampling has no oversampling knob).
- Import: all sub-runs → one new segment in `lolo1-medium` (merged store
  + full label regeneration is the only no-code path; import advances the
  experiment cycle state machine — plan the completion/reset step).
- Regen: `counterfactual_labels --maximum-roots 1000000` to a new
  destination; old roots regenerate byte-identically.
- Retrain: v2-uncapped settings → `controllable-tracker-v3.pt`; then
  rerun `tracker_ood_eval` (no longer clean OOD for the two trained
  states — disclosed) and the §4.31(c) mask-sensitive gate.

Full command lines and citations are preserved in the session recon
transcript; this note is the durable summary.
