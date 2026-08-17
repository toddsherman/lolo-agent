# Session handoff — 2026-08-16 (WP0/WP1 build + direction review)

Status: live working record; update as results land
Operator: Claude (Fable 5) session driven by Todd
Companion docs: `docs/roadmap.md` (plan of record), `docs/learnings.md`
(negative-results log)

## 1. Purpose

Todd's standing goal: run the research loop — build, bounded experiment,
record learnings, improve — until a model beats *Adventures of Lolo* under the
strict interaction-only claim. This file records what this session did,
what is in flight, and what an incoming agent (or a later session with no
context) needs to continue without repeating work. Credits may run out
mid-stream; treat this file as the recovery point.

## 2. Completed this session

1. **Full repo + roadmap review.** Findings (all verified against code):
   - Full unit suite passes: 402 tests, 4 skipped, ~11s
     (`.venv/bin/python -m unittest discover -s tests`).
   - Roadmap ↔ code consistency confirmed: every existing "primary file"
     exists; every planned file (`object_tracks.py`, `accessibility.py`,
     `controllable_tracker.py`, `phase_model.py`, `relational_planner.py`)
     correctly absent before the build below.
   - `neural_planner.py` is 27,650 lines — larger than the rest of the
     package combined; WP1 extraction is the highest structural priority.
   - Review recommendations recorded: golden/characterization tests before
     the WP1 extraction; a mechanical `make check`/CI gate for invariant 12;
     make the WP1 `observe_frame(frame, player_mask, phase)` player-mask
     input explicitly swappable so WP5's learned tracker can replace the
     assisted mask without reworking correspondence; WP6's
     `AccessibilityDelta` is a static cell graph — plan for phase/time-
     conditioned reachability for rooms with moving hazards.
2. **`docs/learnings.md` reviewed** (added by another agent this morning).
   Its §6 rejected directions and §8 do-not-repeat checklist are treated as
   binding filters on new proposals.

## 3. In flight (two background multi-agent workflows)

### 3.1 Research-direction panel

Question: does the plan of record point in the right direction, and where
should it change? Structure: 3 grounding readers (docs corpus, planner core,
learned-model stack) → 5 independent direction proposers with distinct
lenses (learning-first, objective-design, search-and-oracle,
science-and-claims, pragmatic-delivery) → 1 adversarial critic per proposal
checking against already-falsified evidence, strict-track constraints, and
overlap with the existing roadmap.

**COMPLETE.** All five critic verdicts: adopt_modified. Synthesis written to
`docs/direction-review-2026-08-16.md` — plan of record survives; adopted
amendments A–E (early accessibility probe on the confirmed push; WP5
mechanized in parallel on the spatial-v10 backbone; strict-lineage linter +
preregistration addendum; WP9a offline milestone spike; WP7 off Gate 4's
critical path + WP2-lite descope). Rejected elements and verified code
findings are recorded there, including: assisted player mask is load-bearing
inside tracked-state signatures; strict track DID clear Floor 1
(medium-experiment-2026-08-08.md) — correct any contrary narrative;
lolo1-medium dataset is strict-bound; v313 archived zero save states.

### 3.2 WP0 + WP1 build — COMPLETE, COMMITTED

Landed as commits `236ea65` (WP1: object_tracks.py extraction, zero behavior
change, telemetry verified byte-identical, 18 new tests) and `b95b68f`
(WP0: evaluation-partitions manifest, partitions.py loader/audit,
research-cycle wiring, 32 new tests). Full suite: 452 tests, OK. Task A's
original builder died on an API error; the verifier caught the gap and a
fix round delivered WP0 in full — second verification round green.
Direction review + offline diff docs committed as `8d8eb45`.

Withheld allocation shipped as default (rooms 25/30/35/40/45/50 — final
room of floors 5–10): **pending Todd's ratification** before broad room
training.

Original build plan (for audit):

Implements roadmap §16 steps 1–2 / backlog Tasks A and B. Structure:
2 recon agents (read-only extraction map of `neural_planner.py`; WP0
manifest design) → 2 builders in parallel with disjoint file ownership →
adversarial verifier (full suite + whole-diff behavior-change review) with
up to 2 fix rounds.

File ownership (also the conflict rule if other agents join mid-build):

- **Task A (WP0):** `configs/` (new partition manifest),
  new `lolo_agent/partitions.py`, `lolo_agent/research_cycle.py`,
  new `tests/test_partitions.py`, `docs/protocol.md` addition.
- **Task B (WP1):** new `lolo_agent/object_tracks.py`,
  `lolo_agent/neural_planner.py` (delegation edits only),
  `lolo_agent/unlabeled_entities.py` (only if required),
  new `tests/test_object_tracks.py`.

Acceptance (from roadmap WP0/WP1): immutable manifest + loader rejection of
training writes from withheld/sequel partitions + digest audit over every
persistent artifact class + partition telemetry events; zero planner
behavior change; v318/v321 archive-metadata compatibility via
self-contained fixtures (tests must NOT read gitignored `experiments/`
paths); pure conversion functions; full suite green.

**Deliverable:** verified diff committed to `main` with a capability-style
commit message. If this session dies mid-build: run
`git -C /Users/toddsherman/Projects/lolo status` and the full suite; the
verifier's checklist above defines "done"; incomplete work should be
finished against the same acceptance list, not restarted.

### 3.3 Offline accessibility diff (Amendment A step 0) — COMPLETE

Preregistered null result: pushed-configuration and pre-push coverage
envelopes are identical beyond the object footprint; v319's exhausted d9
search bounds the pushed frontier inside the pre-push envelope. Recorded in
`docs/offline-accessibility-diff-2026-08-16.md` and `docs/learnings.md`
§4.26. The native paired probe (arms: v318 `state-00000117` vs the
`33addc6c` rollback checkpoint; three directed targets) is now the decisive
next experiment; its full design is in the experiment note.

## 4. Decisions pending for Todd

1. **Room-partition allocation (WP0).** The build ships a default proposal
   (rooms 1–3 development since they influenced engineering; a pre-registered
   subset of later Lolo 1 rooms withheld; all Lolo 2 sequel). This binds all
   future evaluation and needs Todd's explicit ratification before broad
   room training begins (roadmap WP0 stopping rule).
2. **Push to remote.** Local commits are part of the roadmap cycle
   discipline; pushing to `origin` awaits Todd's confirmation.
3. **Direction-review adoption.** Any roadmap changes proposed by
   `docs/direction-review-2026-08-16.md` are recommendations until Todd
   ratifies them; WP0/WP1 are safe under all of them.

## 5. Session artifact locations (outside the repo)

Ephemeral — copy anything durable into `docs/` before relying on it:

- Direction panel transcripts:
  `~/.claude/projects/-Users-toddsherman-Projects-lolo/354d515c-2be6-4001-be73-90cf19acae16/subagents/workflows/wf_f106dc0b-fdd/`
- Build workflow transcripts:
  `.../subagents/workflows/wf_556ac385-2d9/`
- Persistent cross-session memory (Claude sessions only):
  `~/.claude/projects/-Users-toddsherman-Projects-lolo/memory/`

### 3.4 Native probe cycle — COMPLETE (see learnings §4.27–4.28)

Three native runs (v322/v323/v324) + instrument fix (`ddae223`). Certified
conclusion: the v318 push is accessibility-neutral; object displacement is
necessary for column-8 band entry; Gate 4 vehicle redirected to the
westward displacement. Next experiment preregistration goes in
`docs/paired-accessibility-probe-2026-08-16.md`'s successor note before
execution: certified paired probe of westward-displaced vs pre-push,
scoring certified-hold band entry (candidate roots: the westward-push
commits near d7–d8 of v323/v324, which have content-addressed decision
snapshots).

## 6. Next-cycle queue (after WP0/WP1 land)

Per roadmap §16 and `learnings.md` §10, unchanged unless the direction
review says otherwise:

1. WP2 multi-track correspondence (Task C), then Task D planner/archive
   integration (high-conflict file — one agent at a time in
   `neural_planner.py`).
2. WP3 displacement/transformation descriptors (Task E) + track telemetry
   (Task F).
3. Bounded two-manipulation native gate (Task I) — first new *learnings*
   moment; record outcome in `learnings.md` §4 with run IDs.
4. WP6 accessibility prototype in mock environments (Task G) in parallel.

Do not: raise beam/depth after failure, retrain returnability on tiny
negatives, add reward weights, or start broad room training before the
partition manifest is committed and ratified.

## 7. End-of-day state (2026-08-16 evening)

Suite: 729 tests OK. The loop (Todd's standing directive) is active with
persistent memory; see the /loop queue in the latest ScheduleWakeup.

- Gate 3 CLOSED (assisted track). WP9 step 1 FALSIFIED as written
  (learnings 4.33). WP5 arc complete through the mask-sensitive gate:
  localization closed across Room 3 (tracker v4), promotion failed on
  mask RESOLUTION (4.34) - next spike is pixel-mask reconstruction, then
  rerun the same gate.
- Landed modules: partitions, object_tracks, counterfactual_labels,
  strict_lineage, milestone_discovery(+run), controllable_tracker(+train),
  tracker_substitution_replay, tracker_ood_eval, mask_sensitive_gate,
  accessibility, accessibility_preference, object_correspondence,
  strict_from_assisted_state track.
- Pending: WP8-lite ablation (planner seam patch prepared in
  docs/wp8-lite-ablation-design-2026-08-16.md; planner file owned by
  Todd's still-running fix session); WP9a redesign (4.33 requirements);
  pixel-mask reconstruction spike; Todd's two external fix sessions
  unreported at day end.
