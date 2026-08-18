# Findings index — campaign of 2026-08-16/17

Status: navigational index (derived, not authoritative)  
Covers: `docs/learnings.md` §4.26–§4.52  
Companion documents: `docs/learnings.md`, `docs/roadmap.md` §17–§20

## 1. Purpose and scope

`docs/learnings.md` is the source of truth for what this project has
established, falsified, and retracted. It now runs past fifty numbered
entries and is read sequentially, which makes it hard to answer questions
of the form "what do we know about X" or "which run produced Y".

This file is a finding aid for the two-day campaign of 2026-08-16/17
(entries §4.26 onward). It restates each entry in one line, classifies it,
and points at the runs and artifacts that evidence it; it then groups the
entries by theme. It adds no findings of its own and softens none.

**Where this file and `learnings.md` disagree, `learnings.md` wins.** Every
row here is a pointer, not a substitute for the entry. Entries carry
qualifications — budget scope, censoring, retractions — that a one-line
summary cannot hold; read the entry before citing it.

When new §4.x entries land, extend §3 and the relevant grouping in §4, or
retire this file rather than let it drift.

## 2. How to read the classifications

`learnings.md` §2 defines four evidence levels: falsified at the measured
gate, negative result, engineering defect, and not yet demonstrated. The
campaign's entries also carry recorded classifications outside that set —
"milestone result", "measurement result", "instrument validation",
"measurement-validity finding", "process finding", "failed promotion
gate". The `Class` column below buckets each entry into one of five
labels; the `Recorded as` column preserves the entry's own wording, which
is the wording to cite.

- **falsified** — the stated hypothesis failed under its documented state
  and budget.
- **negative** — the mechanism ran correctly but did not improve the
  target outcome, including failed promotion gates.
- **defect** — the experiment exposed an implementation, telemetry, or
  inference error; conclusions drawn before the correction are invalid.
- **milestone** — a preregistered gate passed.
- **process** — a finding about instruments, gates, discriminators, or
  operating discipline rather than about the agent's capability.

Two cautions that apply to every row:

- **A finite search failure is never global unsolvability.** Results are
  scoped to their exact state, model, search budget, and controller-edge
  set (`learnings.md` §2).
- **Every v3xx run in this campaign is assisted-lineage.** No row below
  is strict-track evidence. See §6.

## 3. Entry index (§4.26–§4.52)

| § | One-line finding | Class | Recorded as | Evidence |
|---|---|---|---|---|
| 4.26 | Offline coverage diff: pushed and pre-push eras have identical beyond-footprint coverage envelopes (8 cells each); apparent differences were budget/reserve artifacts | negative | "Negative result", scoped to offline telemetry at explored budgets (depth ≤9, beam ≤128) | `docs/offline-accessibility-diff-2026-08-16.md`; `docs/direction-review-2026-08-16.md`; v319/v320/v321 (3,498 branches) vs v313/v314/v316/v317/v318 (6,899) |
| 4.27 | First causally paired accessibility fact — the object blocks the tile it occupies — but the configuration-hold instrument was blind to displacement, so the preregistered delta is censored | defect | "Engineering defect (hold-certification instrument) plus censored evidence" | `docs/paired-accessibility-probe-2026-08-16.md`; runs `entity-v322-…-arm-a-pushed-d12`, `entity-v323-…-arm-b-prepush-d12` |
| 4.28 | Certified rerun: with the object undisturbed, pushed and pre-push configurations reach exactly the same 7 cells — the v318 eastward push is accessibility-neutral | negative | "Negative result (certified, budget-scoped)" | `docs/paired-accessibility-probe-2026-08-16.md` §7–§8; run `entity-v324-…-arm-b-rerun-certified-d12` |
| 4.29 | The "westward push" inference of §4.27/§4.28 is falsified — the band-opener is *removal*; and `anonymous_object_track_cells` is accumulated history, not endpoint configuration | defect | "Engineering defect … plus a validated decomposition method" | `docs/object-removed-probe-2026-08-16.md` (premise correction); v324 telemetry |
| 4.30 | Removing the `(7,6)` entity yields 135 certified configuration-held branches into the column-8 band: 24 certified cells vs 7 in both baselines, including milestone cell `(12,11)` | milestone | "Milestone result" — the project's first verified accessibility-improving manipulation | `docs/object-removed-probe-2026-08-16.md`; run `entity-v325-room3-object-removed-probe-d12` (9,691 branches, 24 min) |
| 4.31 | WP5 substitution replay letter-passes on mask-irrelevant bits; the informative channel showed tracker v2 does not localize on Room 3 frames | negative | "Failed promotion in substance (formal letter-pass recorded and disqualified); plus an instrument-design lesson" | `docs/wp5-tracker-training-2026-08-16.md`; `experiments/lolo1-wp5/substitution-replay-report.json` (digest `6061b45e…`) |
| 4.32 | The tracker OOD gap is state-dependent, not uniform: held-in AUC 0.9997 / IoU 0.775 vs Room 3 pooled hit 0.478 / AUC 0.679 / IoU 0.022; object-present states fail totally | process | "Measurement result" completing the §4.31 diagnosis — the gap is corpus coverage, not architecture | `docs/tracker-ood-eval-2026-08-16.md`; `experiments/lolo1-wp5/tracker-ood-report.json` (digest `ecc5336b…`) |
| 4.33 | WP9 step 1 falsified as written: 7/47 heart collections scored positive against the fixed 0.80 recall gate; reversion-based negative valence failed 0/14 | falsified | "Falsified at the measured gate" for WP9 step 1 as written | `docs/milestone-scoring-2026-08-16.md`; `experiments/lolo1-wp5/milestone-scoring-report.json` (digest `424bb775…`) |
| 4.34 | Mask-sensitive gate FAILS at signature agreement 0.000 despite 100%/100%/98.8% mask overlap — the gap is mask resolution/extent, not localization | negative | "Failed promotion gate" with a completed diagnosis chain | `docs/wp5-tracker-training-2026-08-16.md`; `experiments/lolo1-wp5/mask-sensitive-gate-report.json` (digest `7bb95c5e…`) |
| 4.35 | Pixel reconstruction moves every measurable axis (IoU 0.33→0.40) but cannot pass a byte-equality signature bit — replication gates conflate "masks correctly" with "reproduces the incumbent including its defects" | negative | "Failed promotion gate plus a gate-design finding" | `docs/wp5-tracker-training-2026-08-16.md`; `experiments/lolo1-wp5/mask-sensitive-gate-v2-report.json` (digest `1052c9ea…`) |
| 4.36 | WP9a v2: life-loss negative valence 14/14 PASS via delayed divergence, but heart separation fails again at 15/47 = 0.319 through reset bleed-through | falsified | "Falsified at the measured gate, second time" — demoted from redesign to RETHINK | `docs/milestone-scoring-v2-2026-08-16.md`; `experiments/lolo1-wp5/milestone-scoring-v2-report.json` (digest `898676b5…`) |
| 4.37 | Functional gate: detection 0.30–0.45 vs gate 0.95 through symmetric erasure — but learned beats incumbent on preservation (0.97–0.98 vs 0.72–0.77), the first axis where it exceeds the incumbent | negative | "Failed promotion gate with the correct instrument at last" | `docs/wp5-tracker-training-2026-08-16.md`; `experiments/lolo1-wp5/functional-gate-report.json` (digest `414c6576…`) |
| 4.38 | Occupied/vacated disambiguation roughly doubles detection (0.30–0.45 → 0.68–0.78) and kills displacement erasure; in-place erasure remains, exhausting the label-semantics program | negative | "Failed promotion gate", second mechanism isolated | `docs/wp5-tracker-training-2026-08-16.md`; `experiments/lolo1-wp5/functional-gate-v2-report.json` (digest `7d1e5703…`) |
| 4.39 | Two-channel detection closes bit (a) at 1.000 on all three corpora and v322 passes all four bits — NO-PROMOTE solely on the bit-(b) placement-flip tail (0.936 / 0.943 vs 0.95) | negative | "Failed promotion gate by tail margin", every detection mechanism closed | `docs/wp5-tracker-training-2026-08-16.md`; `experiments/lolo1-wp5/functional-gate-v3-report.json` (digest `01a9b128…`) |
| 4.40 | Pose-diverse strict collection moved the stability tail slightly *against* the data lever (0.936→0.929, 0.943→0.933); the stopping rule fires and the campaign pauses | negative | "Negative result for the data lever" on the stability tail | `docs/wp5-tracker-training-2026-08-16.md`; `experiments/lolo1-wp5/functional-gate-v4-report.json` (digest `99285632…`) |
| 4.41 | WP9a v3: heart recall improves a third time (0.149 → 0.319 → 0.574, gate 0.80) while the negative gate regresses to 2/14 — both defects live inside the event representation | falsified | "Falsified at the measured gate, third time — the unit is wrong." No fourth rescore of this unit | `docs/milestone-scoring-v3-2026-08-16.md`; `experiments/lolo1-wp5/milestone-scoring-v3-report.json` (digest `e2c3434c…`) |
| 4.42 | Ensemble-variance anchor passes the functional gate on every axis and corpus: stability 1.000 / 0.9995 / 0.9728, detection 1.000, preservation 0.994–0.999 | milestone | "Promotion gate PASSED" — PROMOTE-to-shadow | `docs/wp5-tracker-training-2026-08-16.md`; `experiments/lolo1-wp5/functional-gate-v5-report.json` (digest `ac4bd00f…`) |
| 4.43 | WP8-lite ablation: the restore-preference mechanism fires correctly (bonus 25.0, full attribution) but the arms' committed trajectories are identical — the control's plain frontier score already restored the same removal-class branch | negative | "FAIL (mixed) per the preregistration"; mechanism validated, *discrimination* failed | `docs/wp8-lite-ablation-design-2026-08-16.md` §7; `experiments/lolo1-wp5/wp8lite-ablation-report.json` (digest `19f4092f…`); runs `entity-v327-…-control-w0-d12`, `entity-v328-…-treatment-w1-d12` |
| 4.44 | Relational shadow run: establish → hold → exploit chain forms with correct linkage and pre-execution logging; an option is re-instantiated at d8; non-interference confirmed | process | "Instrument validation" for the chain machinery; the capability claim remains untested | run `entity-v329-room3-relational-shadow-d12` (79,493 events); `docs/wp8-relational-planner-design-2026-08-17.md` |
| 4.45 | E1 FAIL: the exploit held authority across decisions 3–7, which contained **zero option searches**, so its reserve seam never fired — hypotheses can rank what search offers but cannot cause search | falsified | "Falsified at the measured gate", with an architectural cause: the relational planner is a *passenger* | `docs/wp8-relational-planner-design-2026-08-17.md` §12; `experiments/lolo1-wp5/e1-gate4-report.json` (digest `6b6708db…`); runs `entity-v330-…-control-off-d12`, `entity-v331-…-treatment-selection-d12` |
| 4.46 | Across v327–v331 the planner ran **zero** of its own option searches (the one search per run is the decision-0 resume audit); the stagnation deferral gate defers whenever any archive branch carries a frontier flag | defect | "Engineering/behavioral defect of the incumbent planner" | `docs/wp8-search-scheduling-design-2026-08-17.md` §Q1; runs v327–v331, with v325 (two planner-initiated searches) as contrast. **Contains one retracted bullet — see §4.49** |
| 4.47 | E3-pre at 16 decisions: the control does not collect `(12,11)`, so the discriminator is technically alive, but the trajectory closes to Chebyshev distance 2 and is moving at the target | process | "Measurement-validity finding" — a treatment win would evidence a 2–3 decision speedup, not a capability difference | run `entity-v332-room3-e3-pre-control-off-d16`; `docs/wp8-search-scheduling-design-2026-08-17.md` §Q4 |
| 4.48 | At 24 decisions the control reached `(12,10)` — one cell from the milestone — then a stagnation restore valued 60.35 pulled it to distance 4 and it never returned; standing adjacent to a certified milestone is worth zero to the incumbent scorer | process | "Discriminator VALIDATED, and the capability gap named" — the gap is *finishing*, not finding | run `entity-v333-room3-e3-pre-control-off-d24` (d17 commit seq 82868, d18 restore seq 83289) |
| 4.49 | E3 precondition S0 was already landed by `6a8488a`; measured commit-time archive supply is 52 of 56 (4 → 56), not 4; the carried configuration signature is frozen, not recomputed | process | "Process finding … plus a measured supply result and a disclosed latent risk" | `lolo_agent/neural_planner.py:13750` `_root_object_track_branch_fields`; offline lineage reconstruction over run v333 |
| 4.50 | E3 FAIL: the navigation mechanism fired (7 of 20 instants `differs: true`) and made the outcome worse — treatment min distance 3 vs control 1 — because the excursions steering removed are what deposit the archive ladder | falsified | "Falsified at the measured gate", refuting §4.48's proposed repair while preserving its diagnosis | `docs/wp8-search-scheduling-design-2026-08-17.md` §11–§12; `experiments/lolo1-wp5/e3-gate4-report.json` (digest `26a3cc22…`); runs `entity-v334-…-off-d24`, `entity-v335-…-selection-d24` |
| 4.51 | E5: supply preserved (archives span columns 6–12 in both arms), the closing intervention refused §4.48's exact failure move, treatment reached distance 1 at d16 — but `(12,10)` was never deposited as an archive, so the restore key had no candidate for the current position | falsified | "Falsified at the measured gate", diagnosis narrowed to "the planner cannot retain the valuable position it is currently standing on" | `docs/wp8-search-scheduling-design-2026-08-17.md` §13–§14; `experiments/lolo1-wp5/e5-gate4-report.json` (digest `3c6b1e18…`); runs `entity-v336-…-off-d24`, `entity-v337-…-restore-only-d24` |
| 4.52 | A healthy E5 treatment arm was reported dead from two orchestrator instrument artifacts: a `pgrep` pattern that could never match the module-form process, and a zero-commit count that is on-profile at 6.4k events | process | "Measurement error by the orchestrator", caught by the agent it was addressed to; no harm | mid-E5 operational monitoring of run `entity-v337-…-restore-only-d24`; no experiment artifact |

## 4. Thematic groupings

### 4.1 Accessibility and Gate 3 — §4.26 → §4.30

The one chain in this campaign that ends in a pass. It runs from an
offline diff through two instrument corrections to a certified positive
result, and it inverts its own premise twice on the way.

1. **§4.26 — offline diff.** A read-only telemetry comparison of the
   pushed and pre-push eras finds identical beyond-footprint coverage
   envelopes. Scoped to explored budgets (depth ≤9, beam ≤128); probe
   telemetry was absent in the pre-push arm, so the interaction-frontier
   question was censored, not answered.
2. **§4.27 — paired native probe (v322/v323).** Establishes the first
   causally paired accessibility fact — `(8,6)` probed blocked with the
   object present and committed-walkable with it absent — and exposes the
   hold-certification instrument as blind to object displacement. The
   preregistered delta comes back censored, with directional evidence
   recorded as directional only.
3. **§4.28 — certified rerun (v324).** With the instrument fixed, the bit
   is 0: the pushed and pre-push configurations reach exactly the same
   seven cells. **The v318 push is certified accessibility-neutral.** The
   §4.27 directional evidence resolves as configuration-*departure*, not
   configuration difference.
4. **§4.29 — track decomposition.** Corrects the record three times: the
   "westward push" reading of §4.27/§4.28 is falsified (the mechanism is
   *removal*); accumulated track cells are history, not endpoint state
   (five of six were stale); causal-archive restore events carry no track
   fields.
5. **§4.30 — object-removed probe (v325).** Bit = YES. 135 certified
   configuration-held branches enter the column-8 band; certified coverage
   goes 7 → 24 cells and exposes the milestone cell `(12,11)`. Zero life
   losses.

**Gate 3 status: CLOSED on the assisted track.** Formal closure required
one repetition from a fresh restore; run
`entity-v326-room3-object-removed-repetition-d12` reproduced the 24-cell
envelope at Jaccard 1.0 (`docs/object-removed-probe-2026-08-16.md`
"Repetition results", roadmap §17 item 4). Strict-track re-measurement
remains gated on WP5. There is no dedicated §4.x entry for v326.

The thesis this chain demonstrates natively: **manipulation value is a
measurable property of configurations, not of manipulations.** The same
machinery certified one confirmed, persistent, causally verified
manipulation as neutral (§4.28) and another as enabling (§4.30).

### 4.2 The four Gate 4 levers and why each failed

Gate 4 — deliberate preparation — **remains OPEN**. Four levers have been
measured under preregistration; each failed for a distinct, named
mechanism, and none of the failures is a tuning problem.

| Lever | Entry | Runs | Why it failed |
|---|---|---|---|
| Restore preference (accessibility term in restore selection) | §4.43 | v327 / v328 | **Redundant.** The mechanism fired correctly, but the control's plain frontier score already restored the same removal-class branch. No score conflict existed at that root, so consequence bits could not discriminate deliberate from incidental choice. |
| Search-time hypothesis reserve | §4.45, cause in §4.46 | v330 / v331 | **No searches to ride.** The reserve seam fires only inside an option search; the exploit held authority across decisions 3–7, which contained zero. §4.46 then found zero planner-initiated searches across v327–v331 — the planner's own archive growth trips a stagnation deferral gate. |
| Commit steering (navigation target) | §4.50 | v334 / v335 | **Starves the exploration that deposits archives.** The lever worked (7 of 20 instants changed behavior) and the outcome went backwards: min distance 3 vs the control's 1. Archive geography collapsed from columns 6–12 to 6–8 and every subsequent restore ratcheted westward. |
| Target-aware restore key alone | §4.51 | v336 / v337 | **The target was never a candidate.** Supply was preserved and the closing intervention refused §4.48's exact failure move, but at d16 the agent stood at `(12,10)`, which was never deposited as an archive in either arm. The key re-ranks archives that exist; the cell needing to be held was the current position. |

Supporting entries for this line:

- **§4.44** — the chain machinery itself (establish → hold → exploit,
  chain-parent linkage, option storage and reuse, pre-execution logging)
  works on real native state under telemetry authority, with
  non-interference confirmed. The exploit's `budget_exhausted`
  termination there is the expected null in telemetry mode and is
  explicitly *not* evidence about budget sizing.
- **§4.47 / §4.48** — validation of the `(12,11)` discriminator. §4.47
  found it alive at 16 decisions but visibly closing; §4.48's 24-decision
  extension refuted the convergence worry and named the capability gap:
  the incumbent can wander into the neighborhood but has no mechanism to
  close, because proximity to a certified milestone is worth zero to its
  scorer.
- **§4.49** — E3's preconditions verified against code at HEAD, measured
  archive supply (52 of 56), and a disclosed frozen-signature risk.

The residue after four levers is small and specific. §4.51's plan change
is **E6: deposit the certified-adjacent position as an archive** so the
restore key that already works has something to reach, with a veto on
stagnation-restoring away from such a position as the declared fallback.

Two claims that should *not* be carried forward from this line:

- Not "the planner cannot find valuable configurations" — §4.43 and
  roadmap §18 item 1 correct that: existing novelty machinery chose the
  removal twice with no accessibility term.
- Not "steering needs better weighting" — §4.50 records the failure as
  structural, and notes that the tie-break-not-reward care taken in the
  mechanism did not save it.

### 4.3 The WP5 perception chain, labels to promotion — §4.31 → §4.42

Six gate iterations, each isolating exactly one mechanism, each fix
verified by the next gate rerun unchanged. Every report digest is recorded
byte-identical on rerun.

| Step | Entry | Mechanism isolated | Outcome |
|---|---|---|---|
| Substitution replay | §4.31 | Mask-irrelevance — the gated bits could pass without the mask mattering | Letter-pass disqualified; tracker does not localize on Room 3 |
| OOD evaluation | §4.32 | Corpus coverage, state-dependent (object-present states fail totally) | Gap quantified; diagnosis is data, not architecture |
| Mask-sensitive gate | §4.34 | Resolution/extent — cell-block erasure vs pixel silhouette plus halo | FAIL; localization closed |
| Pixel reconstruction spike | §4.35 | Replication criterion — byte-equality cannot be met by a mask more stable than the incumbent | FAIL; replication gates retired for perception |
| Functional gate | §4.37 | Symmetric erasure — occupied/vacated blur covers the effect in both endpoints | FAIL; learned first exceeds incumbent on preservation |
| Occupied/vacated labels | §4.38 | In-place erasure — nothing vacates, so no silhouette refinement can expose it | FAIL; label-semantics program exhausted |
| Two-channel detection | §4.39 | Detection quantity — masked-region differential alongside outside-mask signature | Bit (a) 1.000; v322 passes all four bits; NO-PROMOTE on the stability tail |
| Data lever | §4.40 | — (lever tested, not a mechanism) | Tail moved slightly against the lever; stopping rule fires, campaign pauses |
| Ensemble-variance anchor | §4.42 | Anchor marginality — cells straddling 0.5 across adjacent poses | **PASS / PROMOTE-to-shadow** |

**Status: WP5 campaign COMPLETE.** The learned masking convention (tracker
v4 + pixel head v3 occupied-v2 + reconstruction v3 + detection quantity
v2) is detector-free, lineage-clean, and passes the functional gate on
every axis and corpus, exceeding the assisted incumbent on stability,
preservation, and in-place detection (§4.42).

Scope limits to preserve when citing this:

- **Promoted to shadow, not wired.** The next step is shadow wiring in the
  planner as telemetry alongside the assisted convention, which queues
  behind the planner-file release (§4.42 plan change, roadmap §20 status
  note). There is no native evidence from the learned convention yet.
- WP5's acceptance clause — strict tracking without `PixelHeartGoalPrior`
  imports — becomes *reachable* once shadow telemetry accumulates; it is
  not met.
- §4.42 records watch-items for shadow telemetry rather than re-tuning:
  v325's residual 2.7% stability tail, empty-mask rates, and the
  detection-channel shift toward the differential.
- §4.37 also established, report-only, that the assisted incumbent misses
  6–12% of ground-truth manipulations — "the incumbent is not a gold
  standard; it is merely incumbent."
- Assisted-era telemetry was used here as *evaluation* ground truth only
  (§4.31), which does not contaminate strict corpora.

### 4.4 The WP9 falsification series — §4.33 → §4.36 → §4.41

Three preregistered rescores, three falsifications, one representation
verdict. Heart recall improved monotonically and never approached the
gate.

| Version | Entry | Heart recall (gate 0.80) | Negative valence (14) | Falsified mechanism |
|---|---|---|---|---|
| v1 | §4.33 | 0.149 (7/47) | 0/14 | Event-level dependence-censoring, return-censoring, rarity non-separation; reversion valence fails because matched NOOP controls also die |
| v2 | §4.36 | 0.319 (15/47) | 14/14 PASS | Reset bleed-through; class-scoped valence over merged component classes flips whole collection classes negative |
| v3 | §4.41 | 0.574 | 2/14 | Occurrence-scoped valence starves the terminal class — fatal commits' own windows are empty |

**Verdict: the event unit itself is wrong** (§4.41). Class-scoped valence
bleeds; occurrence-scoped valence starves; both defects live inside the
matched-endpoint-pair + successor-window representation. No fourth rescore
of this unit is authorized.

What survives: delayed factual-vs-control divergence is validated as the
terminal-valence signal, and it generalized out of the design sample (10
of 14 v2 passes came from runs never inspected at design time, §4.36).

What replaces it: milestone events represented at the **object level** —
collection as a tracked appearance ceasing at a heart cell, death as
control-loss divergence on the tracked controllable region — with
replay/reset-stable identity. WP9 step 1 is paused at the representation
level and rescheduled after WP2/WP3 integration provides object-level
event streams (§4.41, roadmap §20 status note).

The meta-observation worth carrying: **monotone metric improvement across
redesigns can coexist with a structurally unsatisfiable representation.**
The paired gates exposed it where either gate alone would have invited
more tuning.

### 4.5 Instrument and gate-design lessons

Roadmap §18 item 4 makes gate design a first-class deliverable. This
grouping collects the entries that earned it. Several of the campaign's
most expensive corrections were measurement defects, not capability
failures.

**Instruments that were blind:**

- **§4.27** — the configuration-hold signature could not see object
  displacement, censoring the delta the whole experiment existed to
  measure. A paired design is only as strong as its hold instrument.
- **§4.29** — accumulated track state cannot distinguish "changed at some
  point" from "still changed"; every archive class that can reseed a
  search root must carry the track block.
- **§4.31** — preregistered bits must be sensitive to the mechanism they
  claim to test. Bits 1–2 were mask-irrelevant for those archive shapes,
  so the pass demonstrated nothing.
- **§4.46** — search frequency was an implicit assumption, not a metric;
  it is now a first-class planner-health number to report per run.

**Gates that measured the wrong thing:**

- **§4.35** — replication criteria conflate "masks correctly" with
  "reproduces the incumbent including its defects". A learned mask more
  stable than the assisted one can never pass a byte-equality bit.
  Substitution gates must be functional.
- **§4.37** — promotion gates refereed by neutral ground truth measure
  *both* conventions; the incumbent's own 6–12% misses were structurally
  invisible to replication gates.
- **§4.39** — a class-mix-sensitive false-positive bound was falsified
  from training data *at design time*, before the gate run was spent.
- **§4.43** — an ablation root must exhibit score conflict. Root selection
  is part of experimental power, not just provenance.
- **§4.47** — discriminator validity is a trend property, not a binary at
  one budget. "Control never does X in N steps" must be checked against
  whether the control is converging on X.
- **§4.44** — a telemetry-mode null must not be read as a parameter
  finding. Budget adequacy is only measurable under selection authority.

**Unit and precondition errors:**

- **§4.45** — `realization_branch_budget` is a per-depth parent-reserve
  slot count, not a verified-branch count; sizing it from traverse cost
  would have been a unit error.
- **§4.49** — verify preconditions against code at HEAD, not design-doc
  anchors. A `**`-splat defeats keyword-name scanning, and the earlier
  "confirmed missing" was itself a measurement error.

**Operational monitoring — §4.52.** The orchestrator reported a healthy
run dead from two of its own artifacts: a `pgrep -f "lolo-neural-run"`
pattern that could never match `python -m lolo_agent.neural_run`, and a
zero-committed-decisions count that is exactly on-profile at 6,409 events
(the first `decision_committed` lands at seq 75,742 in every arm of this
family). Nothing was relaunched. Three lessons are recorded:

1. Instrument-quality discipline applies to operational monitoring, not
   just experiments.
2. **Agreement between instruments that share an author is not
   corroboration.**
3. The guard that worked was procedural — the relaunch instruction stated
   in advance that a crashed arm is VOID, not FAIL, so a bad premise could
   not have become a bad result.

## 5. Run and artifact quick reference

| Runs | Entry | Role |
|---|---|---|
| v313 / v314 / v316 / v317 / v318 | §4.26 | Pre-push 4-heart-era Room 3 baselines (6,899 branches) |
| v319 / v320 / v321 | §4.26 | Pushed-configuration searches (3,498 branches) |
| v322 (arm A, pushed) | §4.27 | Paired accessibility probe, pushed arm |
| v323 (arm B, pre-push) | §4.27 | Paired accessibility probe, pre-push arm — hold certification failed |
| v324 (arm B rerun, certified) | §4.28, §4.29 | Certified rerun; source of the track decomposition |
| v325 (object-removed probe) | §4.30 | 24-cell certified envelope; also §4.46's two-search contrast |
| v326 (object-removed repetition) | roadmap §17 item 4 | Gate 3 closure, Jaccard 1.0 — no dedicated §4.x entry |
| v327 / v328 | §4.43 | WP8-lite ablation, control w0 / treatment w1 |
| v329 | §4.44 | Relational planner shadow run (telemetry-only) |
| v330 / v331 | §4.45 | E1, authority off / selection |
| v332 | §4.47 | E3-pre control, 16 decisions |
| v333 | §4.48, §4.49 | E3-pre control, 24 decisions — discriminator validated |
| v334 / v335 | §4.50 | E3, control / selection at 24 decisions |
| v336 / v337 | §4.51, §4.52 | E5, control / restore-only at 24 decisions |

Report artifacts live under `experiments/lolo1-wp5/`; native runs under
`experiments/lolo1-entity-v10/evaluations/`. Every report digest cited in
§3 is recorded in its entry as byte-identical on rerun.

## 6. What a reader should conclude

**On scope, before anything else.** Every native run in this campaign
(v322–v337) is assisted-lineage. The accessibility results, the Gate 3
closure, the certified envelopes, and all four Gate 4 lever measurements
are assisted-track evidence. None of them supports a strict-track claim.
Roadmap §17 item 7 and §17 item 4 both state this: WP5 remains required
for the strict headline claim, and strict-track re-measurement of Gate 3
is gated on it.

**What was established.**

- One accessibility-improving manipulation is verified end-to-end:
  removing the `(7,6)` entity more than triples certified reachable space
  (7 → 24 cells) and exposes a milestone-bearing cell, reproduced at
  Jaccard 1.0 (§4.30, Gate 3 closed on the assisted track). The full
  causal chain — detect manipulation, preserve configuration, measure
  accessibility consequence — runs on real emulator state with anonymous
  instruments only.
- The complementary negative is equally established: a confirmed,
  persistent, causally verified manipulation can be strategically neutral
  (§4.28). Preservation priority must be coupled to measured consequence.
- The learned masking convention passes a functional gate on every axis
  and exceeds the assisted incumbent on stability, preservation, and
  in-place detection (§4.42). It is promoted to shadow and **not yet
  wired**; there is no native evidence from it.

**What remains open.**

- **Gate 4.** Four levers, four distinct named failures (§4.43, §4.45 with
  §4.46, §4.50, §4.51). The diagnosis has narrowed considerably — from
  "the planner cannot prepare deliberately" to "the planner cannot retain
  the valuable position it is currently standing on" — but no lever has
  closed it, and the next one (E6) is designed, not run.
- **WP9 step 1.** Paused at the representation level after three
  falsifications (§4.33, §4.36, §4.41), awaiting object-level event
  streams from WP2/WP3.
- Room 3 completion additionally requires a *second* manipulation: the
  remaining hearts `(8,4)` and `(9,12)` lie outside the 24-cell envelope
  (roadmap §18 item 3).

**What not to conclude.**

- Not that novelty-driven exploration is the problem. §4.43 and §4.50 both
  cut the other way: novelty found the valuable configuration twice
  unaided, and the excursions that look wasteful are what deposit the
  archive ladder later progress climbs (roadmap §20 item 1).
- Not that any failed lever needs a bigger weight, a longer budget, or a
  wider search. §4.45 proved its budget inert; §4.50 records its failure
  as structural; `learnings.md` §8 forbids raising beam or depth on
  failure alone.
- Not that the bounded failures prove impossibility. Each is scoped to its
  state, model, budget, and controller-edge set.

**The shape of the campaign.** Of the 27 entries indexed here, two are
passes. Most of the rest are failures with a mechanism attached — which is
what makes them cheap to build on. The recurring meta-lesson, recorded
independently at least eight times (§4.27, §4.31, §4.35, §4.37, §4.39,
§4.43, §4.47, §4.52) and adopted as roadmap §18 item 4, is that gate and
instrument design is as load-bearing as capability work: budget design
effort for a gate equal to the capability it gates, and require of every
gate an instrument that can contradict it.
