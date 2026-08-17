# WP5 controllable-tracker full training — preregistration (2026-08-16)

Status: PREREGISTERED before execution; results to be appended
Basis: spike commit `4ce4086`; label corpus manifest digest `352b37f1…`
(16,602 roots, 77,730 arms, 64,509 non-empty); direction-review
Amendment B.

## Run

`lolo-controllable-tracker` training over the full strict-bound label
corpus: run-held-out split (hash-stable), lr 1e-3 (head-appropriate rate
per the spike finding — the recorded 1e-5 rule applies to fine-tuning
pretrained weights, not fresh heads), epochs 40, wall-clock ceiling
5,400 s (external watchdog), MPS. Frozen spatial-v10 backbone digest
pinned.

## Gates (fixed now)

1. Untrained same-architecture baseline beaten on all four
   baseline-relative requirements (per the training CLI's built-in gate).
2. Held-out cell ROC AUC ≥ 0.95 and IoU@0.5 reported (spike smoke: AUC
   0.979, IoU 0.269 at 400 arms — full data must not fall below smoke
   AUC materially; IoU ceiling from backbone resolution is expected and
   reported, not gated).
3. Uncertainty–error correlation positive on held-out runs.
4. Checkpoint lints clean under `strict_lineage` (assisted=False, zero
   violations) with pinned provenance digests.

Promotion beyond telemetry (the Amendment B substitution replay against
v320/v321 archives, and the held-out comparison vs the assisted
detector) are SEPARATE gates, not granted by this run.

## Results (appended 2026-08-16)

**All four gates PASS — with one preregistration deviation disclosed.**

Deviation: the training CLI's default caps limited the run to 4,000
training / 1,000 validation arms (hash-stable run-held-out split), not the
full 64,509-arm corpus as preregistered. Gates were scored on the capped
run; an uncapped confirmation run follows (same gates, explicit caps
lifted).

Capped-run numbers (40 epochs, lr 1e-3, MPS):

- Gate 1 (beat untrained baseline): PASS on every axis — AUC 0.6166 →
  **0.9980**, IoU 0.0089 → **0.4324**, loss 0.7027 → 0.0248, Brier
  0.2548 → 0.0076.
- Gate 2 (AUC ≥ 0.95): PASS at 0.9980; IoU 0.4324 reported (up from the
  spike's 0.269; backbone-resolution ceiling still expected).
- Gate 3 (uncertainty–error correlation > 0): PASS at +0.4242.
- Gate 4 (strict lineage): PASS — `assisted=False`, zero violations,
  label-manifest digest `352b37f1…` and backbone digest pinned.
- Bonus: Brier 0.0076 now beats the constant-prevalence baseline 0.0088
  (the spike's noted weakness, resolved by scale).

Precision 0.4401 / recall 0.9607 at 0.9% prevalence: the head finds
nearly all controllable cells and over-covers by ~2x — consistent with
the documented one-displacement-step label blur.

### Uncapped confirmation (appended 2026-08-16)

Full corpus (70,000/12,000 arm caps, i.e. uncapped in practice; 40
epochs): **all gates pass with large margins over the capped run** —
held-out AUC **0.9997**, IoU **0.7464** (the capped run's 0.43 was a
data-scale artifact, not a backbone ceiling), precision 0.7587 / recall
0.9789, Brier 0.0022 vs constant 0.0090, uncertainty–error correlation
+0.4734, lineage `assisted=False` with zero violations. Checkpoint:
`experiments/lolo1-wp5/controllable-tracker-v2-uncapped.pt`.

### Substitution-replay preregistration (promotion gate; added before execution)

Per direction-review Amendment B: an offline replay recomputes the
v318/v320/v321 tracked-state reconstruction with the LEARNED mask
substituted for the assisted player mask. The `object_tracks` pure
functions take `player_pixel_mask` as a parameter, so the replay calls
them with tracker-v2 masks over the recorded frames — no planner edits.
Scored bits, fixed now: (1) the confirmed manipulation identity
(source `(7,6)`, destination `(8,6)`, direction, effect signature)
reconstructs equivalently from v318/v321 archive metadata under the
learned mask; (2) the appearance fingerprint recovered at the destination
matches the recorded one within the established L1 threshold; (3) mask
divergence between learned and assisted masks over the replayed frames is
reported per frame (divergence telemetry per the salvaged Amendment B).
Pass = bits 1 and 2 hold on all replayed archives. Fail = recorded
conclusion per Amendment B: assisted masking remains load-bearing at
current data scale, and the strict claim must disclose it.

### Substitution-replay outcome (appended 2026-08-16)

Formal letter-PASS, substantive NO-PROMOTE — full analysis in
`docs/learnings.md` §4.31. Bits 1–2 proved mask-irrelevant for these
archive shapes; divergence telemetry shows the tracker does not localize
on Room 3 frames (out-of-distribution for the training corpus). Tracker
stays telemetry-only. Redesigned mask-sensitive gate required before any
promotion; OOD evaluation against counterfactual ground truth from
v322–v326 telemetry queued.

### Tracker v3 — coverage-closure results (appended 2026-08-16)

Tier-2 corpus (labels v3: 18,358 roots / 84,546 arms; +6,384 Room 3
arms). Training gates all pass (held-out AUC 0.9994, IoU 0.7393,
uncertainty correlation +0.51); held-in evaluation unchanged (AUC 0.9987,
IoU 0.768 — no forgetting). Room 3 re-evaluation (v322/v323/v325 ground
truth; no longer clean OOD for the two collected states — disclosed):

| state | v2 hit/AUC | v3 hit/AUC/IoU |
|---|---|---|
| object-present (v322) | 0.000 / 0.440 | **1.000 / 1.000 / 0.928** |
| pre-push (v323) | 0.001 / 0.446 | **0.946 / 0.996 / 0.799** |
| object-removed (v325) | 0.833 / 0.865 | 0.162 / 0.937 / 0.133 |

Reading: closure is state-local and immediate where collection covered the
palette; v325's ground truth spans the opened east region (columns 8–12)
which the d7-rooted strict explorer never visited, so its frames remain
uncovered. (v325's v2 "hit 0.833" was against blob-like masks — v3's
sharper masks expose the true gap; AUC rose 0.865→0.937 regardless.)
Next data increment: east-region collection from v325 decision snapshots
(d2 in-band pre-heart; d4 post-heart east), then labels v4 → tracker v4.
Report: `experiments/lolo1-wp5/tracker-v3-room3-report.json`.

### Tracker v4 — full Room 3 closure (appended 2026-08-16)

Labels v4 (20,002 roots / 90,678 arms; tier-3 east-region collection from
v325 d2/d4 snapshots). Training gates pass (AUC 0.9993, IoU 0.7559,
corr +0.53). Room 3 re-evaluation:

| state | v2 | v3 | v4 |
|---|---|---|---|
| object-present | 0.000/0.440 | 1.000/1.000 | 1.000/1.000/0.929 |
| pre-push | 0.001/0.446 | 0.946/0.996 | 0.952/0.995/0.822 |
| object-removed | 0.833*/0.865 | 0.162/0.937 | **1.000/1.000/0.866** |

(held-in 1.000/0.9991/0.785 throughout — no forgetting; * = blob artifact)

Three collection cycles, each closing exactly what it covered:
state-local, immediate, reproducible. The perception recipe —
counterfactual labels + targeted strict collection — is validated as an
iterative pipeline. Remaining before promotion: the §4.31(c)
mask-sensitive gate (agreement with the assisted mask specifically on
frames where masking demonstrably matters), noting the WP5 comparison
lands at parity with the color heuristic on these corpora (both at
ceiling against counterfactual ground truth), which per Amendment B's
salvaged form supports shadow-promotion with divergence telemetry rather
than a superiority claim.

### Mask-sensitive promotion gate — preregistration (§4.31(c); added before execution)

Instrument lesson being fixed: the original substitution replay's gated
bits were mask-IRRELEVANT for the replayed archive shapes (§4.31), so a
letter-pass demonstrated nothing about the learned mask. This gate scores
agreement ONLY where masking demonstrably matters. All definitions,
thresholds, and pass criteria below are fixed now, before the scorer runs
on any real corpus; the implementation is `lolo_agent/mask_sensitive_gate.py`
(unit-tested on synthetic fixtures only at preregistration time).

Corpora: the three probe corpora under
`experiments/lolo1-entity-v10/evaluations/` — v322 (object-present), v323
(pre-push), v325 (object-removed). Population per corpus: every
content-addressed frame in `frames/` (each unique frame once, sorted by
digest, digest-verified on decode).

Mattering-frame detector (assisted mask vs no mask; the learned mask does
not participate): per frame, run the assisted goal-prior detection
(`PixelHeartGoalPrior.detect_player`, `player_pixel_mask`, fresh per
frame, as in the replay's divergence sweep). A frame is MASK-MATTERING iff
`world_effect_cells_state_signature` over the coarse cells the assisted
mask touches (16×15 `UnlabeledEntityMemory` grid) differs between the
assisted-masked and the fully unmasked computation. Per-cell
`masked_cell_fingerprint` values are recorded alongside (same features,
so the two views agree modulo hash collision). Frames without an assisted
detection cannot be mattering and are reported by reason.

Gate quantities on mattering frames: recompute the same quantities with
the LEARNED mask substituted — tracker v4
(`experiments/lolo1-wp5/controllable-tracker-v4.pt`, parameter sha256
`b2fdd8ba…`, frozen spatial-v10 backbone `642d66ed…`) thresholded at the
pinned 0.5 validation operating point and pixel-expanded with the
`tracker_substitution_replay` helpers (`learned_mask_cells`,
`learned_pixel_mask`, `learned_reference_slot` — reused by import). The
comparison cell set is every coarse cell touched by EITHER mask, so
learned over-coverage that erases neighbouring anonymous appearance is a
scored disagreement, not an invisible one. Two per-frame agreement bits:

1. `signature_equal`: `world_effect_cells_state_signature` over the
   comparison cells identical under learned and assisted masks (exact
   equality is the planner's own world-state comparison — the
   silently-replaceable criterion).
2. `l1_within`: every comparison cell's `feature_at` vector within the
   established normalized-L1 appearance threshold 0.08
   (`UnlabeledEntityMemory.feature_distance`).

Preregistered gate bits (per corpus, over mattering frames): both
agreement rates ≥ 0.95, AND ≥ 50 mattering frames (instrument validity —
perfect agreement over fewer frames is vacuous, not a pass). The gate
passes only if all three corpora pass. Note `signature_equal` implies
`l1_within` per frame (the signature hashes the same features), so the
signature rate is the binding bit and the L1 rate is the graded
instrument reporting HOW close the learned reconstruction is.

Reported, not gated: agreement rates on non-mattering frames (expected
trivially high; disagreements there are still listed honestly), frame
counts by reason, and per-frame mask IoU
(`mask_divergence`) on every mattering frame.

One run, then results appended here. Deterministic content-digested
report to `experiments/lolo1-wp5/mask-sensitive-gate-report.json`
(digest prefix `wp5-mask-sensitive-gate:v1:`). Honest outcome either
way: PASS supports shadow-promotion with divergence telemetry (parity
claim only, per Amendment B's salvaged form); FAIL means the learned
mask changes downstream tracking quantities on exactly the frames where
masking matters and cannot silently replace the assisted mask yet.

### Mask-sensitive gate — results (appended 2026-08-16, after one preregistered run)

**FAIL on all three corpora — NO-PROMOTE.** Report
`experiments/lolo1-wp5/mask-sensitive-gate-report.json`, content digest
`7bb95c5e3e08640716585cceaf838c870ccc122b9600a3dde85966bad191c030`,
byte-identical on rerun. No preregistration deviations.

The instrument worked: every frame in every corpus is mask-mattering
(assisted detection on 6,474/6,474 frames; erasing the assisted mask
changes quantized features in 4–9 cells per frame), so unlike the §4.31
bits, these bits could not letter-pass vacuously — and they fail
decisively:

| corpus | mattering | sig-agree | L1-agree | mask IoU mean/med | bit |
|---|---|---|---|---|---|
| v322 object-present | 1116/1116 | **0.000** | 0.068 | 0.338 / 0.275 | FAIL |
| v323 pre-push | 3910/3910 | **0.000** | 0.039 | 0.342 / 0.280 | FAIL |
| v325 object-removed | 1448/1448 | **0.000** | 0.025 | 0.314 / 0.289 | FAIL |

(Non-mattering agreement rates are vacuous — there are no non-mattering
frames. Preregistration expected that bucket to be large; it is empty
because the probe corpora always show the player on screen.)

Reading — the failure is mask RESOLUTION and EXTENT, not localization:

- Localization is fine, as §4.32/v4 closure predicted: the learned mask
  overlaps the assisted mask on 100% of v322/v325 frames and 98.8% of
  v323 (pixel IoU mean ≈ 0.33 vs the v2 replay's 0.0002–0.0009 — three
  orders of magnitude — with 47 v323 frames of empty learned mask as the
  only residual coverage gap).
- But the learned mask erases 1–3 whole 16×16 cell blocks, while the
  assisted mask erases the sprite silhouette plus a 3-pixel halo spread
  partially across 5–9 cells. Fully-masked pools encode zeros; partially
  masked pools average the surviving background pixels — so the
  downstream features differ at the player cell BY CONSTRUCTION, and the
  halo's fringe erasure in neighbouring cells is not reproduced at all
  (max per-cell L1 median ≈ 0.4, five times the 0.08 bound; exact
  signature equality never occurs).

Consequence: tracker v4 knows WHERE the player is at cell resolution,
but a cell-resolution mask is not a drop-in replacement for the
pixel-resolution assisted mask in `object_tracks` quantities. Per the
preregistered criterion the tracker stays telemetry-only; the honest
§4.31 letter-pass/substantive-fail pattern is now inverted — a
substantive instrument, a clean fail. Promotion paths this measurement
licenses considering next (not attempted here): reconstruct a
pixel-resolution mask anchored at the learned cells (e.g. the connected
sprite component inside the predicted region), or move the downstream
convention itself to cell-resolution masking on both sides — a
convention change that must be gated on its own, not smuggled in as
substitution.

### Pixel-mask reconstruction spike — preregistration (§4.34 plan-change; added before execution)

Basis: learnings §4.34 — the mask-sensitive gate failed on RESOLUTION
(the learned mask erases whole 16×16 cell blocks; the assisted convention
erases a pixel sprite silhouette plus a ~3px halo), with localization
closed by tracker v4.  This spike is the first promotion path §4.34
licenses: reconstruct a pixel-resolution mask anchored at the learned
cells, then rerun the same gate unchanged.  Everything below is fixed
now, before the training run and before any gate execution; the
implementation is `lolo_agent/pixel_mask_head.py` +
`lolo_agent/pixel_mask_train.py` (unit-tested on synthetic fixtures only
at preregistration time; 31 tests).

Key insight being exploited: pixel-resolution supervision already exists
detector-free — the factual-vs-duration-matched-NOOP endpoint difference
is per-PIXEL before `counterfactual_labels` pools it to cells.

**Supervision (strict, detector-free).** Pixel silhouette targets per
labeled arm of the pinned v4 corpus: the byte-exact per-pixel
factual-vs-control endpoint difference, reduced to the 4-connected
components that survive leave-one-action-out corroboration at pixel
granularity (intersection of corroborating sibling arms' changed-pixel
sets; arms sharing an action never corroborate; empty-change siblings
abstain).  Censoring is inherited unchanged from the v4 cell records;
the pixel path cross-checks the record's changed cells and
corroborating-arm counts and fails loudly on mismatch.  Changed-but-
uncorroborated pixels are weighted hard negatives (residual).  Arms whose
pixel silhouette is empty are excluded exactly as the cell trainer
excludes empty cell masks.  Disclosed caveat inherited from the cell
path: the target is the union of vacated and occupied silhouettes (one
displacement step of blur); the appearance-conditioned head is expected
to resolve it because vacated pixels look like background.  No goal
prior, player detector, or any assisted symbol appears anywhere in the
label or training path (`strict_lineage` lint pinned by unit test).

**Head.** `PixelMaskHead`: 4 input channels (native 256×240 RGB + the
FROZEN tracker v4 per-cell probability map upsampled to its 16×16 pixel
blocks), three 3×3 convolutions (hidden 32, SiLU; 7×7 receptive field —
enough to see a 3px halo's context) and a 1×1 output logit per pixel;
19,713 parameters, single head (no ensemble — the cell trainer's
uncertainty gate axis is therefore N/A and dropped; disclosed deviation).
Tracker v4 (`controllable-tracker-v4.pt`, parameter sha256 `b2fdd8ba…`)
and the spatial-v10 backbone (`642d66ed…`) are frozen throughout; only
the new head trains.  Digest checks fail the run if either frozen
artifact's parameters change.

**Training run (fixed now).** v4 corpus: labels
`experiments/lolo1-wp5/wp5-labels-full-v4.jsonl` (file sha256
`a9f15805…`, manifest digest `ee8d4f8e…`, digest-verified per record) +
`experiments/lolo1-medium/dataset` (strict-bound).  Run-held-out
hash-stable split via the cell trainer's own `split_sequence_runs` and
`sample_arm_examples` (modulus 5, seeds 17/18), caps 6,000 training /
1,500 validation arms (a spike-scale corpus, disclosed as capped), 20
epochs, batch 16, lr 1e-3 (fresh-head rate per the recorded finding),
positive weight 8.0 / residual weight 4.0, MPS, internal wall-clock
ceiling 2,100 s so the external watchdog never kills a run mid-epoch.
Checkpoint to `experiments/lolo1-wp5/pixel-mask-head-v1.pt` pinning the
label manifest digest, the pixel-target corpus digest, tracker v4's
parameter sha256, and the backbone parameter sha256; metrics sidecar
alongside.

**Training gates (fixed now).** Untrained same-architecture baseline
beaten on all of: (1) held-out per-pixel loss, (2) pixel ROC AUC,
(3) mean probability on silhouette pixels above background, (4) above
residual pixels when residual pixels exist.  Reported, not gated:
precision/recall/IoU at the pinned 0.5 operating point, Brier vs
constant-prevalence.  Checkpoint must lint clean under the
`strict_lineage` checkpoint audit.

**Reconstruction convention (fixed now, before any gate run).** The
reconstructed pixel mask of a frame is: tracker v4 cells at probability
≥ 0.5 (the pinned operating point), dilated by 1 cell (anchor region);
head pixels at probability ≥ 0.5 INSIDE the anchor; Chebyshev-dilated by
3 pixels (the documented halo radius parameter of the recorded assisted
masking convention — a convention-matching constant, mirrored by value;
no assisted code participates).  An empty v4 cell mask yields an empty
pixel mask — the 47 v323 frames with empty v4 masks therefore stay
empty and will disagree; accepted, not patched.  Constants pinned by
unit tests; no post-hoc tuning of any threshold, dilation, or halo after
seeing real-corpus results.

**Gate rerun (preregistered).** Rerun `lolo_agent.mask_sensitive_gate`
UNCHANGED — same mattering-frame detector, same two agreement bits
(`signature_equal`, `l1_within`), same thresholds (agreement ≥ 0.95,
appearance L1 ≤ 0.08, ≥ 50 mattering frames per corpus), same three
probe corpora (v322/v323/v325) — except the learned mask source is the
reconstructed pixel mask.  Exact substitution mechanism: the gate's
`score_corpus` already takes the mask source as a predictor object; the
substituted predictor (`PixelSilhouettePredictor`) returns a
pixel-resolution prediction whose grid is one unit per pixel
(columns=width, rows=height) and whose probability map is the
reconstructed mask as a 1.0/0.0 indicator, so the gate's unchanged
helpers (`learned_mask_cells` / `learned_pixel_mask` /
`learned_reference_slot`, at the pinned 0.5 threshold) recover exactly
the reconstructed pixel mask — verified by a unit test that round-trips
a mask through the unchanged helpers and through `score_frame`.  The
driver is `python -m lolo_agent.pixel_mask_train gate`, which only loads
the pinned artifacts (verifying the head checkpoint's pinned tracker and
backbone digests against the loaded ones), composes the predictor, and
calls the gate module's own `score_corpus`/`build_report`; no gate code
is modified.  The driver module is assisted-coupled through the
evaluation-instrument imports exactly as `mask_sensitive_gate` itself
is; the lint test pins that neither new module references an assisted
symbol directly.

One preregistered scoring run, CPU, deterministic report to
`experiments/lolo1-wp5/mask-sensitive-gate-v2-report.json` (the
unchanged gate's own digest scheme), plus a byte-identical determinism
rerun to a scratch path (reported).  Honest outcome either way: PASS →
recommend shadow-promotion with divergence telemetry (parity claim only,
per Amendment B's salvaged form); FAIL → report which quantity diverges
and by how much, per corpus.

### Pixel-mask head v1 — training results (appended 2026-08-16)

**All four preregistered training gates PASS.**  Run exactly as
preregistered (6,000/1,500 arms, 20 epochs completed, MPS, 684 s
wall-clock — ceiling not hit).  Checkpoint
`experiments/lolo1-wp5/pixel-mask-head-v1.pt`, parameter sha256
`85e977f244d545b8047795f15d41d9cbc47c2ae5ed589298b8c572221870c167`,
pinning labels manifest `ee8d4f8e…`, pixel-target corpus digest
`4a8a6af0…`, tracker v4 `b2fdd8ba…`, backbone `642d66ed…`; frozen-digest
checks passed (tracker and backbone parameters unchanged);
`strict_lineage` checkpoint audit clean; head module lints clean
(assisted=False), and the lint test pins that neither new module
references an assisted symbol directly.

Pixel-target derivation: every one of the 7,500 selected arms produced a
non-empty corroborated pixel silhouette (0 empty-mask exclusions;
prevalence 0.42% of pixels), and the pixel/cell cross-checks (changed
cells, corroborating-arm counts) held on all of them.

| quantity (held-out, 92.2M pixels) | untrained baseline | trained |
|---|---|---|
| per-pixel loss | 0.7296 | **0.00859** |
| pixel ROC AUC | 0.5788 | **0.99843** |
| Brier (constant 0.00422) | 0.2682 | **0.00292** |
| precision / recall @0.5 | 0.0042 / 1.0 | 0.4997 / 0.9319 |
| IoU @0.5 | 0.0042 | 0.4821 |
| mean p: silhouette / residual / background | ~0.518 each | 0.828 / 0.738 / 0.0036 |

Reading: the head separates silhouette pixels from background by ~230x
in mean probability; the 2x over-coverage at 0.5 (precision ≈ 0.5 at
recall ≈ 0.93) is consistent with the disclosed vacated/occupied target
blur — and the high residual-pixel probability (0.738, separated from
the silhouette but well above 0.5) shows the head also fires on
changed-but-uncorroborated pixels, which the gate will price honestly.
Metrics sidecar: `experiments/lolo1-wp5/pixel-mask-head-v1.metrics.json`.

### Mask-sensitive gate v2 (reconstructed pixel mask) — results (appended 2026-08-16, after one preregistered run)

**FAIL on all three corpora — NO-PROMOTE.**  Report
`experiments/lolo1-wp5/mask-sensitive-gate-v2-report.json`, content
digest
`1052c9eae73918220d430b9ac0a53fe9fbe97f3b6d35106cfb1a7466959616a7`,
byte-identical on the preregistered determinism rerun.  No
preregistration deviations: the gate module ran unchanged (same
mattering detector — counts identical to v1 at 1116/3910/1448 — same
bits, same thresholds), with only the learned mask source substituted
via the preregistered predictor mechanism.

| corpus | mattering | sig-agree v1→v2 | L1-agree v1→v2 | mask IoU mean v1→v2 | bit |
|---|---|---|---|---|---|
| v322 object-present | 1116/1116 | 0.000 → **0.000** | 0.068 → 0.094 | 0.338 → 0.406 | FAIL |
| v323 pre-push | 3910/3910 | 0.000 → **0.000** | 0.039 → 0.056 | 0.342 → 0.404 | FAIL |
| v325 object-removed | 1448/1448 | 0.000 → **0.000** | 0.025 → 0.038 | 0.314 → 0.396 | FAIL |

Which quantity diverges, and by how much:

- The binding signature bit is 0.000 everywhere (required ≥ 0.95):
  exact quantized-feature equality over the comparison cells never
  occurs, on any of the 6,474 mattering frames.
- The graded L1 bit reaches 0.038–0.094 (required ≥ 0.95); the median
  per-frame max-cell L1 is 0.39–0.42, ~5x the 0.08 bound — essentially
  unchanged from v1 in the median, but no longer in the same failure
  class everywhere (below).
- Pixel mask IoU improved from ~0.33 to ~0.40 mean on every corpus, with
  per-frame maxima now 0.90–0.95 (v1 ceiling: 0.73); overlap fractions
  match v1 (100% / 98.7% / 100%).  v323 has 50 empty reconstructed
  masks: the 47 empty-v4-anchor frames from v1 (empty anchor ⟹ empty
  reconstruction by construction) plus 3 where no head pixel reached
  threshold inside a non-empty anchor.

Reading — the failure decomposes into two distinct residual gaps, and
the resolution-class gap of §4.34 is closed:

1. **Silhouette agreement is bimodal.**  On frames where the
   reconstructed mask nearly coincides with the assisted mask
   (IoU ≥ 0.8: 105/246/52 frames, ~4–9% per corpus), the L1 bound holds
   almost always (105/105, 215/246, 52/52; median max-cell L1 ≈ 0.026,
   a third of the bound) — the pixel reconstruction is a faithful
   drop-in there, something no v1 frame achieved.  On the ~85–88% of
   frames with IoU < 0.5 the bound always fails (median ≈ 0.42).  The
   learned silhouette is nearly constant-size (median ≈ 760 px,
   min 338, max ~1,100–1,670) while the assisted mask itself is
   strongly multi-modal on the same corpora (sampled 340 / ~1,100 /
   ~1,640 px modes within one corpus: the blue-anchored connected
   component shrinks when the sprite is partially occluded and leaks
   into adjacent white regions when touching them, halo included).  The
   remaining disagreement is therefore silhouette shape/extent on the
   probe distribution — partly head error (the disclosed
   vacated/occupied target blur inflates its positives; residual-pixel
   mean probability 0.738 confirms it fires on uncorroborated change),
   and partly the assisted convention's own frame-to-frame variance,
   which counterfactual supervision has no reason to reproduce.
2. **Exact signature equality is an equality-class criterion.**  Even
   the near-perfect frames (max-cell L1 ≈ 0.008–0.026, IoU 0.9+) never
   produce an equal signature: any handful of boundary pixels that
   differ inside one 4×4 pool shifts a pooled mean across a
   quantization step somewhere among the ~11 comparison cells.  The
   binding bit effectively demands pixel-identical masks in every pool
   of every touched cell — byte replication of the assisted mask, not
   appearance equivalence within the planner's own 0.08 tolerance.

Consequence: per the preregistered criterion the tracker + pixel head
stay telemetry-only; no shadow-promotion.  The spike's positive result
is instrumental: cell-block resolution is no longer the failure mode —
a learned pixel mask can match the assisted convention to within the
planner's appearance tolerance on the frames where it localizes the
same silhouette, and the measured residuals name the two remaining
obstacles precisely.  Paths this measurement licenses considering next
(not attempted here, each requiring its own preregistration): (a) close
the silhouette-extent gap with probe-distribution-targeted strict
collection (the three-cycle recipe, now at pixel granularity) and an
occupied/vacated disambiguation in the label path (e.g. intersecting
the changed-pixel set with sibling-arm evidence about the factual
endpoint); (b) accept that byte-exact signature reproduction is
unattainable for any learned mask and gate a convention change instead
— either cell-resolution masking on both sides (§4.34's alternative) or
an appearance-tolerance state-identity criterion — as an explicitly
gated downstream change with its own regression evidence, never as
silent substitution.

### WP5-final functional promotion gate — preregistration (§4.35 plan-change; added before execution)

Basis: learnings §4.35 — the replication gates are RETIRED for perception
promotion (a byte-equality criterion conflates "masks correctly" with
"reproduces the assisted mask including its defects"; even near-perfect
frames of the pixel spike could never letter-pass it).  This gate is the
§4.35-chosen path: an explicitly gated convention change, scored on
FUNCTION — does tracking built on the learned masking convention produce
correct outcomes — judged against detector-free counterfactual ground
truth, never against the assisted mask's bytes.  Everything below is
fixed now, before the scorer runs on any real corpus; the implementation
is `lolo_agent/functional_mask_gate.py` (unit-tested on synthetic
fixtures only at preregistration time; 26 tests, each gate bit with a
fixture the scorer must catch: a missed ground-truth change, a
self-consistency violation, an absorption case).

**Corpora.** The three probe corpora under
`experiments/lolo1-entity-v10/evaluations/` — v322 (object-present),
v323 (pre-push), v325 (object-removed).

**Ground truth (detector-free; the referee).** Per corpus, the
`tracker_ood_eval` extraction machinery reused unchanged: option-search
telemetry regrouped into counterfactual roots
(`probe_first_step_edges` → `collect_probe_roots`) and labeled by
`counterfactual_labels.label_counterfactual_root` itself.  Scored arms
are labeled arms with a non-empty corroborated controllable-cell set
(expected counts, from the §4.32 report: 2,920 / 3,765 / 7,044).  Each
scored arm carries: the factual endpoint frame, the duration-matched
NOOP control endpoint frame from the same saved emulator state, the
corroborated controllable cells (the GT component), and the byte-exact
changed-cell set — which also certifies, per arm, exactly which cells
did NOT change between the two endpoints.  No goal-prior symbol
participates in the ground truth.

**Conventions compared.** LEARNED: the pinned pixel-mask reconstruction
(`pixel-mask-head-v1.pt`, parameter sha256 `85e977f2…`, over frozen
tracker v4 `b2fdd8ba…` and backbone `642d66ed…`; anchor threshold 0.5 +
1-cell dilation, head threshold 0.5, Chebyshev halo 3 — the convention
preregistered for the spike, unchanged), mask pixels and reference slot
recovered through the unchanged substitution-replay helpers at the
pinned 0.5 threshold; an empty reconstruction leaves the frame
explicitly unmasked.  ASSISTED: `PixelHeartGoalPrior.detect_player` +
`player_pixel_mask`, fresh per frame; no detection ⇒ explicitly
unmasked.  Each convention masks every frame it evaluates with its own
mask for that frame; no quantity mixes conventions.  The pinned-digest
cross-checks of the spike's gate driver are repeated at load time.

**Bit (a) — manipulation detection, GT-refereed.**  Measurement =
deduplicated (factual digest, control digest, component cells) triple
(duplicate arm counts reported).  A convention DETECTS the manipulation
iff `object_tracks.world_effect_cells_state_signature` over the
component cells differs between the factual and control endpoints under
that convention's per-frame masks — the planner's own world-state
identity quantity.  The same evidence is lifted through
`object_correspondence` (per-cell signatures + `masked_cell_fingerprint`
as `CellEvidence` → `endpoint_relative_state` →
`observations_from_evidence`), deriving the endpoint-relative track
state on every factual/control pair; the track-state view
(current_cells non-empty) must agree with the signature view modulo
hash collision and the consistency count is reported.  Gated (per
corpus): learned detection rate over ALL GT measurements ≥ 0.95 AND
learned detection rate over the measurements the assisted convention
detects ≥ 0.95.  Both directions reported (assisted-vs-GT,
assisted-given-learned), so assisted defects are visible and a learned
convention that out-detects assisted is not penalized; if assisted
detects zero measurements the assisted-conditioned requirement is
vacuous (flagged) and GT remains the referee.

**Bit (b) — fingerprint stability under the learned convention
(self-consistency, not cross-convention equality).**  Instances: per
root, scored arm i, labeled sibling arm j of a different
(action, duration) with a non-empty changed set: every cell of
component_i ∖ changed_j — cells where the label rule itself certifies
arm j's factual and control endpoints are byte-identical while the
player stands at different positions.  Deduplicated per (factual
digest, control digest, cell).  Any feature motion at such a cell is
pure masking-convention noise: the same GT-tracked object bytes must
fingerprint consistently across player poses or the planner sees
phantom world-state changes.  Gated: learned normalized-L1 feature
distance ≤ 0.08 on ≥ 0.95 of measurements.  The assisted convention's
rate is reported for context only.

**Bit (c) — no player-absorption regression (the v316/v317 defect
class).**  Instances: per scored arm, every in-grid cell at Chebyshev
distance 1 from the GT component that is NOT in the arm's changed-cell
set, deduplicated per (factual digest, cell).  Such cells are
byte-certified player-free at the factual endpoint (any sprite spill
would have differed from the control, landing the cell in the changed
set), directly adjacent to where GT localizes the player's action — the
exact configuration in which the assisted mask's white-component
leakage absorbed adjacent objects.  A convention PRESERVES the adjacent
appearance iff its masked feature stays within L1 0.08 of the unmasked
feature of the same frame and cell.  Gated: learned preservation rate ≥
assisted preservation rate (both reported).  Interpretation note fixed
now: cells the arm itself changed are excluded because the mask SHOULD
alter features where player pixels moved; the defect class
(player-adjacent resting objects, e.g. the v322 corpus's pushed object
at rest) is covered by these instances across the corpus's probe arms.

**Instrument validity.**  Every bit requires ≥ 50 deduplicated
measurements per corpus; perfect agreement over fewer is vacuous, not a
pass (§4.31 lesson).  All thresholds are prior published operating
points reused by import — mask probability 0.5, appearance L1 0.08,
agreement rate 0.95, minimum 50 — plus the structural Chebyshev-1
adjacency radius; nothing is tuned against these corpora, and the unit
tests pin every constant.

**Verdict rule.**  PROMOTE-to-shadow (learned masking convention with
mask-divergence telemetry; engineering-internal, reversible, claim
boundary unmoved per §4.35) iff all three bits pass on all three
corpora; otherwise NO-PROMOTE with every failing mechanism named
per corpus and bit.  Reported, not gated: learned-vs-assisted pixel
mask IoU distributions over the unique factual endpoint frames
(divergence telemetry), empty-mask counts, unmasked-frame counts,
track-state summary statistics, censoring and dedup counts.

One preregistered run, CPU, deterministic content-digested report to
`experiments/lolo1-wp5/functional-gate-report.json` (digest prefix
`wp5-functional-mask-gate:v1:`), plus a byte-identical determinism
rerun to a scratch path (reported).  Honest outcome either way: PASS →
the strict pipeline's masking convention is the learned reconstruction,
shadow-promoted with divergence telemetry; FAIL → the failing
mechanism names which functional quantity the learned convention gets
wrong, per corpus.

### WP5-final functional gate — results (appended 2026-08-16, after one preregistered run)

**FAIL on all three corpora — NO-PROMOTE.**  Report
`experiments/lolo1-wp5/functional-gate-report.json`, content digest
`414c65763cfcc6731b8a38bba646806a58dbf39f18f7351f18cc91bdfb9bc086`,
byte-identical on the preregistered determinism rerun.  No
preregistration deviations.  Instrument validity held everywhere: every
bit cleared its 50-measurement floor by 20x or more (1,069–12,680
deduplicated measurements per bit per corpus), and the signature and
correspondence track-state views agreed on 100% of the 4,488 detection
measurements — the hash-collision consistency channel never fired.

| corpus | (a) GT-detect learned/assisted | learned given assisted | (b) stability learned/assisted | (c) preserve learned/assisted | bits |
|---|---|---|---|---|---|
| v322 object-present | **0.302** / 0.884 (1,175) | 0.342 | **0.766** / 0.964 (1,069) | **0.983 / 0.720** (6,101) | a,b FAIL; c PASS |
| v323 pre-push | **0.447** / 0.940 (1,563) | 0.471 | **0.820** / 0.970 (2,151) | **0.970 / 0.742** (12,680) | a,b FAIL; c PASS |
| v325 object-removed | **0.353** / 0.936 (1,750) | 0.375 | **0.766** / 0.961 (2,129) | **0.965 / 0.769** (9,031) | a,b FAIL; c PASS |

Failing mechanisms, each instance-verified on the report's own rows:

1. **Bit (a) — symmetric erasure of the counterfactual evidence.**  On
   200/200 sampled learned-miss measurements the reconstruction covers
   100% of every ground-truth component cell's pixel block in BOTH
   endpoints (verified instance: component `(6,6),(6,7)`, learned
   coverage 1.00/1.00 both frames, both features all-zeros, joint
   signatures equal; assisted coverage 0.52–0.66, signatures differ).
   The cause chain is the disclosed vacated/occupied training blur (the
   head fires on both ends of a displacement; residual-pixel probability
   0.738) plus the 1-cell anchor dilation and 3-pixel halo: the mask
   blankets the entire displacement region regardless of which endpoint
   the frame shows, so fully-masked pools encode zeros on both sides and
   the factual and control world states become IDENTICAL at exactly the
   cells where ground truth certifies the action's effect.  The assisted
   silhouette is tight enough to leave differing sub-cell residue.
   Learned detections are almost a strict subset of assisted ones
   (assisted-given-learned 0.99–1.00).  Honest instrument note: part of
   the miss set is pure locomotion rendered pose-invariantly invisible,
   which a future pipeline might even want — but the erasure applies to
   the whole corroborated controllable component, which for push arms
   includes the manipulated object's cells (they are what the tracker was
   trained on), so under the planner's existing downstream quantities
   this is a functional regression against the incumbent, not a
   defensible convention difference.  A pipeline that WANTS pose-invariant
   collapse must redesign the detection quantities and gate that change
   on its own.
2. **Bit (b) — all-or-nothing extent swings across poses.**  18–23% of
   byte-certified-identical cells breach the 0.08 bound under the
   learned convention (verified instance: cell `(7,10)`, identical bytes
   across the pair, learned mask coverage 0.17 in one frame and 1.00 in
   the other, L1 0.354 — 4.4x the bound; assisted coverage 0.23/0.25,
   L1 0.005).  The reconstruction's extent is stable in SIZE (§4.35) but
   not in PLACEMENT relative to cell boundaries: whole 16x16 pools flip
   between fully-erased and untouched as the player pose shifts, so the
   same resting object fingerprints differently — the phantom
   world-state-change mechanism the planner cannot tolerate.  Assisted
   stability is 0.96–0.97 on the identical instances.
3. **Bit (c) — PASS, and the direction is informative: the learned
   convention strictly beats the assisted one on the v316/v317 defect
   class.**  The assisted mask erases player-adjacent, byte-certified
   player-free cells on 23–28% of measurements (verified instances:
   assisted coverage 0.88–0.94 of the neighbouring cell, L1 0.32–0.42
   erased appearance, while learned coverage is 0.12 with L1 0.003);
   learned preservation is 0.965–0.983 versus assisted 0.720–0.769 on
   every corpus.  This is the first measured functional axis where the
   learned convention is strictly better than the incumbent — the
   absorption defect that motivated the bit is real, frequent, and the
   learned reconstruction does not have it.

Reported alongside (not gated): the assisted convention itself detects
only 88.4–94.0% of ground-truth manipulations — the incumbent misses
6–12% of GT-certified changes, a fact no replication gate could ever
have surfaced; 29 of v323's 1,308 unique factual endpoints have an
empty learned reconstruction (the known empty-anchor gap, 36/41
control/factual unmasked detection frames); mask divergence telemetry
matches the spike (IoU mean 0.36–0.42, learned size stable ~660–720 px
mean vs the assisted multi-modal 860–916 px mean).

Consequence: per the preregistered criterion the learned masking
convention is NOT adopted; tracker v4 + pixel head v1 stay
telemetry-only.  The functional gate itself is validated as the
standing WP5 promotion instrument — its bits are mechanism-sensitive
(none could letter-pass), ground truth referees instead of the
incumbent's bytes, and it prices both directions (bit (c) credits the
learned mask where it is genuinely better).  Paths this measurement
licenses next (each requiring its own preregistration, then THIS gate
rerun unchanged): (a) occupied/vacated disambiguation in the pixel
label path — §4.35's path (a), now with a functional measurement
naming it as the binding defect: a reconstruction that masks only the
occupied silhouette cannot symmetrically erase a displacement, directly
attacking the bit-(a) mechanism and most bit-(b) flips; (b) with that
disambiguation, revisit the anchor dilation and halo as part of a
convention v2 — extent constants are convention parameters and may
only change with the convention version, never tuned against this
gate's corpora between runs.

### Occupied/vacated disambiguation spike — preregistration (§4.37 plan-change; added before execution)

Basis: learnings §4.37 — the functional gate failed via SYMMETRIC
ERASURE: the union vacated∪occupied silhouette target teaches the pixel
head to cover the whole GT component in BOTH endpoints, zeroing the
factual-vs-control difference exactly where the effect lives (bit (a)),
and the resulting all-or-nothing extent swings drive most bit-(b)
flips.  This spike is §4.37's licensed path (a)+(b): occupied-only
label semantics in the pixel path, reconstruction convention v2, then
THIS gate rerun unchanged.  Everything below is fixed now, before the
v2 training run and before any gate execution; the implementation is
additive in `lolo_agent/pixel_mask_head.py` (new target mode +
convention v2) and `lolo_agent/pixel_mask_train.py` (new flags +
functional-gate driver), unit-tested on synthetic fixtures only at
preregistration time (52 tests, including the moved-sprite split, the
transformation-in-place and removal edge cases, determinism, and the
convention-v2 checkpoint/predictor plumbing).

**Label semantics v2 (`occupied-v2`; detector-free).**  At the FACTUAL
endpoint frame the training target is the OCCUPIED silhouette only —
the pixels where the controllable region IS in that frame.  The split
rule, applied inside each v1 corroborated controllable component
(`split_occupied_vacated`):

> A component pixel `p` is VACATED at the factual endpoint iff some
> corroborating sibling arm (different primitive action, non-empty
> changed-pixel set — the exact eligibility rule the corroboration
> count already uses) ALSO changed `p`, AND the full 3×3 pixel
> neighbourhood of `p` (clipped at frame bounds;
> `OCCUPIED_SPLIT_NEIGHBORHOOD_RADIUS` = 1) is byte-identical between
> the two arms' factual endpoint frames.  OCCUPIED = component ∖
> VACATED; pixels with no sibling evidence default to occupied (the
> destination-region case).

Why this is the temporal direction rule: two different actions displace
the controllable sprite differently, so they cannot both place the same
sprite patch at `p` — nine-byte agreement between their factual
endpoints certifies the local content as action-invariant, i.e. the
revealed scene without the sprite, so the factual frame shows
background at `p` (vacated).  A pixel the sprite occupies at the
factual endpoint disagrees with every sibling that moved the sprite
elsewhere.  Pinned edge-case behaviour (each with a unit-test fixture):
a pure move splits into destination silhouette (occupied) and revealed
origin (vacated) up to a conservative one-pixel occupied seam where a
3×3 window straddles another arm's sprite; a transformation in place is
fully occupied at the same cells; a removal has NO occupied pixels at
the source; factual-pose-transparent pixels (bounding-box pixels
showing background through the sprite) classify vacated, which is
correct occupied-silhouette semantics because the displayed bytes are
scene content.  Vacated pixels become explicitly weighted hard
negatives; residual pixels are unchanged from v1; arms whose occupied
silhouette is empty are excluded and counted
(`empty_occupied_arms`), mirroring v1's empty-mask exclusion.

Design-phase evidence (training corpus only — the gate corpora were
never touched): on a deterministic 40-root stride sample of the pinned
v4 label corpus (195 labeled arms), the rule yields occupied/component
pixel fraction 0.60 mean; on moving arms the occupied-minus-vacated
centroid points along the arm's action direction 124 / 6 / 65
(along / opposed / neutral-or-N/A), with the opposed and neutral cases
concentrated at blocked or short-displacement arms where direction is
undefined (pose-change splits); 0/155 non-empty components lose all
occupied pixels, though 2 duration-16 arms retain only a single seam
pixel because their destination sprite fell into the uncorroborated
residual (origin-only components; the disclosed
corroboration-granularity limit, inherited from v1 where those same
destination pixels were already residual hard negatives — verified on
`legacy-segment:cycle-000011` group 668 up/16: component rows 128–142,
residual destination block rows 113–127).  The
single-pixel-equality variant of the rule was rejected at design time:
it mislabels opaque sprite pixels that coincidentally match the
revealed background byte (11 direction-opposed arms vs 6 for the
neighbourhood rule on the same sample).

**Training run (fixed now; v1 budgets unchanged).**  Same corpus,
loader, split, and budgets as the v1 spike preregistration: labels
`wp5-labels-full-v4.jsonl` (manifest `ee8d4f8e…`, digest-verified) +
`experiments/lolo1-medium/dataset`; run-held-out hash-stable split
(modulus 5, seeds 17/18), caps 6,000 training / 1,500 validation arms,
20 epochs, batch 16, lr 1e-3, positive weight 8.0, residual weight 4.0,
MPS, internal wall-clock ceiling 2,100 s; frozen tracker v4
(`b2fdd8ba…`) and backbone (`642d66ed…`) with digest checks.  New,
preregistered: `--target-semantics occupied-v2` and vacated-negative
weight 8.0 (`--vacated-weight`) — the vacated pixels are exactly the
pixels the v1 head demonstrably fires on, so a false positive on a
vacated pixel is priced equal to a false negative on an occupied pixel
(the positive weight's mirror; fixed a priori, not tuned — one
preregistered run).  Same architecture (19,713 parameters; no
ensemble).  Checkpoint to
`experiments/lolo1-wp5/pixel-mask-head-v2.pt`, pinning additionally the
target semantics and the reconstruction convention (below); the v2
pixel-target corpus digest uses its own prefix and pins the vacated
sets, so v1/v2 corpora can never alias.

**Training gates (fixed now).**  The v1 spike's four
untrained-baseline gates unchanged — (1) held-out per-pixel loss,
(2) pixel ROC AUC, (3) silhouette-above-background mean probability,
(4) silhouette-above-residual when residual pixels exist — plus one
v2-specific gate: (5) mean probability on occupied silhouette pixels
above vacated pixels when vacated pixels exist (the disambiguation the
spike exists to deliver).  Reported, not gated: precision/recall/IoU at
0.5, Brier vs constant, the vacated-pixel mean probability.
`strict_lineage` checkpoint audit must be clean; the head module must
keep linting assisted-free.

**Reconstruction convention v2 (pinned now, before any gate run).**
Anchor cell threshold 0.5 (unchanged), ANCHOR DILATION 0 (reduced from
1), head pixel threshold 0.5 (unchanged), Chebyshev halo 3 (unchanged —
still the documented assisted-convention halo constant, mirrored by
value).  Rationale, recorded before execution: the v1 dilation existed
so the anchor could span the union silhouette (origin plus
destination); occupied-only targets need only the occupied sprite,
whose pixels lie in cells the tracker's own union-labelled training
targets already cover — measured on the training corpus (60-root
stride sample, 238 arms, never the gate corpora), the undilated
0.5-anchor contains 99.75% of occupied target pixels and fully
contains 235/238 arms (dilation 1: 100% both), and §4.37's bit-(a)
mechanism explicitly names the anchor dilation as an erasure
amplifier.  The halo is kept because bit (c) PASSED with it and it is a
recorded convention constant, not a fitted parameter.  The convention
version travels with the head checkpoint (`target_semantics`,
`anchor_cell_dilation`) and is restored onto the head at load, so the
unchanged gate composition applies convention v2 automatically and a
v2-supervised head can never be silently reconstructed under the v1
convention.  Constants pinned by unit tests; no post-hoc tuning of any
threshold, dilation, or halo after seeing real-corpus results.

**Gate rerun (preregistered).**  Rerun the WP5-final FUNCTIONAL gate
UNCHANGED — `lolo_agent.functional_mask_gate`'s own
`build_conventions` + `score_corpus` + `build_report`, same three bits,
same thresholds (0.5 mask probability, 0.08 appearance L1, 0.95
agreement rate, minimum 50, Chebyshev-1 adjacency), same ground-truth
extraction, same three probe corpora (v322/v323/v325), same verdict
rule (PROMOTE-to-shadow iff bits (a) and (b) pass with bit (c) not
regressing below assisted, on all three corpora) — with only the
learned convention's pinned head checkpoint substituted
(`pixel-mask-head-v2.pt`).  The driver is
`python -m lolo_agent.pixel_mask_train functional-gate`, which loads
the pinned artifacts (repeating the digest cross-checks), lets the
unchanged predictor apply the checkpoint-pinned convention, corrects
only the static provenance strings to describe the applied convention
honestly, and calls the gate module's own scorer; no gate code is
modified.  One preregistered scoring run, CPU, deterministic report to
`experiments/lolo1-wp5/functional-gate-v2-report.json` (the unchanged
gate's own digest scheme), plus a byte-identical determinism rerun to
a scratch path (reported).  Honest outcome either way: PASS →
recommend shadow-promotion of the learned masking convention with
mask-divergence telemetry (engineering-internal, reversible, claim
boundary unmoved per §4.35); FAIL → name the failing mechanism per
corpus and bit against v1's numbers.

### Pixel-mask head v2 — training results (appended 2026-08-16)

**All five preregistered training gates PASS.**  Run exactly as
preregistered (6,000/1,500 arms, 20 epochs completed, MPS, 694 s
wall-clock — ceiling not hit; no deviations).  Checkpoint
`experiments/lolo1-wp5/pixel-mask-head-v2.pt`, parameter sha256
`d486693181be83d010dd0d43a42f88d4a3988fd0d9fb5e0d87cd4d205fefad10`,
pinning labels manifest `ee8d4f8e…`, v2 pixel-target corpus digest
`d74acb6f…` (v2 prefix, vacated sets pinned), tracker v4 `b2fdd8ba…`,
backbone `642d66ed…`, target semantics `occupied-v2`, anchor cell
dilation 0; frozen-digest checks passed; `strict_lineage` checkpoint
audit clean (assisted=False, zero violations); head module still lints
assisted-free.

Target derivation: 5,990 of 6,000 selected training arms produced a
non-empty occupied silhouette (10 `empty_occupied_arms` excluded —
0.17%, the disclosed origin-only large-displacement class; 0 in
validation; 0 empty union silhouettes anywhere); all pixel/cell
cross-checks held.  Held-out prevalence 0.24% occupied, with 168,026
vacated and 20,671 residual pixels as explicit negatives.

| quantity (held-out, 92.2M pixels) | untrained baseline | trained v2 | v1 (union) |
|---|---|---|---|
| per-pixel loss | 0.7297 | **0.00317** | 0.00859 |
| pixel ROC AUC | 0.7512 | **0.99945** | 0.99843 |
| Brier (constant 0.00241) | 0.2683 | **0.00090** | 0.00292 |
| precision / recall @0.5 | 0.0024 / 1.0 | **0.686 / 0.960** | 0.500 / 0.932 |
| IoU @0.5 | 0.0024 | **0.667** | 0.482 |
| mean p: occupied / residual / background | ~0.518 each | 0.883 / 0.714 / 0.0011 | 0.828 / 0.738 / 0.0036 |
| mean p: vacated | 0.518 | **0.053** | n/a (in target) |

Reading: the disambiguation the spike exists to deliver is delivered —
occupied pixels fire at 0.883 while vacated pixels are suppressed to
0.053 (16.7x separation; under v1's union target these same pixels WERE
positives).  Gate 5 (occupied above vacated) passes with a wide margin,
and every v1 axis improved as a side effect (IoU 0.482 → 0.667 at
higher recall).  Metrics sidecar:
`experiments/lolo1-wp5/pixel-mask-head-v2.metrics.json`.

### WP5-final functional gate v2 (occupied-v2 head, convention v2) — results (appended 2026-08-16, after one preregistered run)

**FAIL on all three corpora — NO-PROMOTE.**  Report
`experiments/lolo1-wp5/functional-gate-v2-report.json`, content digest
`7d1e5703b07ac8d2b713ff41161bbec6f73188f13faca674f84f488086d910b1`,
byte-identical on the preregistered determinism rerun (scratch path,
file sha256 equal).  No preregistration deviations: the gate module ran
unchanged through its own `build_conventions`/`score_corpus`/
`build_report` (driver `pixel_mask_train functional-gate`), ground
truth identical to v1 (measurement counts match v1 exactly:
1,175/1,563/1,750 detection, 1,069/2,151/2,129 stability,
6,101/12,680/9,031 preservation), signature and track-state views
agreed on 100% of detection measurements, every bit cleared the
50-measurement floor.

| corpus | (a) GT-detect v1 → v2 (assisted) | given-assisted v1 → v2 | (b) stability v1 → v2 (assisted) | (c) preserve v1 → v2 (assisted) | bits |
|---|---|---|---|---|---|
| v322 object-present | 0.302 → **0.743** (0.884) | 0.342 → 0.784 | 0.766 → **0.975 PASS** (0.964) | 0.983 → **0.998** (0.720) | a FAIL; b,c PASS |
| v323 pre-push | 0.447 → **0.782** (0.940) | 0.471 → 0.809 | 0.820 → **0.936** (0.970) | 0.970 → **0.985** (0.742) | a,b FAIL; c PASS |
| v325 object-removed | 0.353 → **0.678** (0.936) | 0.375 → 0.703 | 0.766 → **0.943** (0.961) | 0.965 → **0.989** (0.769) | a,b FAIL; c PASS |

What moved, and what the residual mechanism is:

1. **The v1 mechanism — SYMMETRIC ERASURE OF DISPLACEMENTS — is
   resolved.**  Detection roughly doubled on every corpus; the learned
   mask is now sprite-sized (mean 423–471 px vs v1's 637–721; assisted
   859–916 multi-modal), learned-vs-assisted mask IoU rose to 0.43–0.49
   mean (v1: 0.36–0.42), and for the first time the learned convention
   detects GT manipulations the incumbent misses (58/34/36 per corpus;
   assisted-given-learned dropped from ~1.0 to 0.93–0.97 — the learned
   detections are no longer a near-strict subset).
2. **Bit (a) residual — IN-PLACE ERASURE, instance-verified on 250
   sampled misses per corpus:** 90–92% of misses have full learned-mask
   coverage of every GT component cell block in BOTH endpoints, and the
   misses are overwhelmingly in-place arms — the factual and control
   learned masks nearly coincide (mask IoU ≥ 0.8 on 250/250, 241/250,
   218/250 sampled misses; median 0.95).  These are blocked/contact
   arms where the player does not displace: occupied@factual equals
   occupied@control, so an occupied-only mask (plus halo 3) still
   blankets the pose-change/contact component that IS the ground-truth
   effect.  This is the degenerate case occupied/vacated
   disambiguation cannot reach by construction — when nothing vacates,
   there is nothing to disambiguate.  The assisted convention detects
   only 76–86% of these same sampled misses itself (its rates on the
   full corpora are 0.88–0.94, partly via its own frame-to-frame
   silhouette variance leaving differing residues).
3. **Bit (b) is a tail phenomenon now:** learned stability L1 mean
   0.0145–0.0214, at or near the assisted convention's 0.0135–0.0156
   (v1: 0.051–0.066); max 0.29–0.31 (v1: 0.35–0.46).  v322 passes
   outright at 0.975; v323/v325 miss the 0.95 bar by 1.4/0.7 points on
   residual whole-pool placement flips at anchor/halo boundaries.
4. **Bit (c) strictly widened:** learned preservation 0.985–0.998 vs
   assisted 0.720–0.769 — the occupied-only target plus the undilated
   anchor reduced neighbour-cell spill further (v1 learned:
   0.965–0.983).  The absorption advantage over the incumbent is now
   1.5–28x fewer violations.

Reported alongside (not gated): v323 empty learned reconstructions rose
29 → 47 of 1,308 unique factual endpoints (the known empty-anchor gap
plus more frames where no head pixel clears 0.5 inside the now-smaller
anchor; 68/80 factual/control detection frames explicitly unmasked);
v322/v325 have zero empty masks.

Consequence: per the preregistered criterion the learned masking
convention is again NOT adopted; tracker v4 + pixel head v2 stay
telemetry-only.  The label-semantics hypothesis of §4.37 is
CONFIRMED — occupied/vacated disambiguation removed the displacement
erasure it was designed to remove, moved every gated axis toward or
past the incumbent, and produced the first learned-only detections —
but the gate exposes a second, previously masked failure class:
in-place manipulations, where the effect lives entirely under the
sprite's unmoved footprint and ANY convention that masks the
controllable region at both endpoints hides it from the planner's
existing signature quantities.  Paths this measurement licenses next
(each requiring its own preregistration, then THIS gate rerun
unchanged): (a) an in-place-aware detection path — the downstream
quantities, not the mask, are what erase unmoved-footprint evidence,
so this is a §4.35-style explicitly gated convention change on the
detection quantities (e.g. scoring the masked-region residue itself),
never a silent mask tweak; (b) the bit-(b) tail (halo-boundary
placement flips) may close with probe-distribution strict collection
(the three-cycle recipe) without any convention change; (c) if pose
change under an unmoved footprint is deemed out of the manipulation
claim's scope, that scope change must be preregistered and priced,
not assumed.

### Detection quantity v2 — preregistration (§4.38 plan-change; added before execution)

Basis: learnings §4.38 — the residual bit-(a) failure class is IN-PLACE
ERASURE, and it is a property of the detection QUANTITY, not the mask:
for blocked/contact arms the world outside a correct controllable mask
does not differ between the factual and duration-matched-NOOP
endpoints, so "does the world outside the mask differ?" is structurally
blind under ANY mask that covers the unmoved footprint in both
endpoints.  This spike is §4.38's licensed path (a): an explicitly
gated convention change on the detection quantity itself, never a mask
tweak.  Everything below is fixed now, before the scorer touches any
gate corpus; the implementation is additive in
`lolo_agent/functional_mask_gate.py` (the v1 scorer, report builder,
and driver are byte-identical to the published runs; 50 unit tests,
including an in-place fixture the signature channel provably misses and
the differential catches, an animation-under-mask fixture that must NOT
fire, false-positive-discipline fixtures, and determinism).

**Detection quantity v2 (fixed now).**  Per deduplicated ground-truth
measurement (factual digest, control digest, component cells), a
convention DETECTS the manipulation iff either channel fires:

1. SIGNATURE channel — the v1 quantity verbatim: the joint
   `world_effect_cells_state_signature` over the component cells
   differs between the endpoints under that convention's per-frame
   masks (correspondence lift and consistency reporting unchanged).
2. MASKED-REGION DIFFERENTIAL channel — the byte content of the
   DUALLY-HIDDEN COMPONENT REGION differs between the endpoints.  The
   region is the intersection of the pixels the convention masks in the
   factual frame and in the control frame (a frame without an anchor
   slot hides nothing), restricted to the `feature_at` pixel blocks of
   the ground-truth component cells.  Rationale, fixed now: pixels
   hidden in only ONE endpoint already perturb that endpoint's pooled
   features (the signature channel sees them), so the dually-hidden set
   is exactly the evidence the v1 quantity cannot see; the component
   anchor is the neutrality discipline — masked content outside the
   ground-truth locus can never claim a detection of THIS manipulation.
   The channel fires on >= 1 byte-differing pixel
   (`DIFFERENTIAL_MIN_CHANGED_PIXELS` = 1, equivalently: the region
   content digests differ); no tolerance parameter exists, mirroring
   the `cell_difference` rationale (deterministic emulator, byte-exact
   endpoints, duration-matched phase).

Both conventions are scored under the same two-channel quantity — the
incumbent gets the identical upgrade, so the comparison is not rigged
in either direction — and both channels are logged per measurement per
convention (`signature_detected`, `differential_fired`, region and
changed-pixel counts), with per-corpus channel aggregates including
`differential_only_detected`, so the conventions change is fully
auditable, never blended.

**Gated bits (fixed now).**  Bit (a) gates the SAME two preregistered
conditions as v1 — learned v2 detection rate over ALL ground-truth
measurements >= 0.95 AND over the measurements the assisted convention
(also under v2) detects >= 0.95 — at the same thresholds (0.95
agreement, 0.5 mask probability, minimum 50).  Bits (b) and (c) are
UNCHANGED in definition, code path, and inputs; their per-corpus
numbers must reproduce the v2 report's values exactly (an internal
consistency check, verified and reported).  Verdict rule: PROMOTE-to-
shadow iff bits (a) and (b) pass with bit (c) not regressing below
assisted AND bit (d) (below) disciplined, on all three corpora;
otherwise NO-PROMOTE with every failing mechanism named.

**Bit (d) — differential false-positive discipline (new; the v312/v313
raw-change-pathology guard).**  A detection channel that fires on
animation inside the mask would repeat the raw-change pathology in new
clothes, so the new channel's false-positive risk is priced on
ground-truth-certified no-effect pairs and gated.  Population, per
corpus: deduplicated (factual digest, control digest) pairs of LABELED
arms whose corroborated controllable set is empty, in two
label-rule-certified classes — `identical_endpoints` (empty changed
set: byte-identical endpoints, the duration-matched no-change pairs)
and `uncorroborated_change` (changed cells exist but every one failed
leave-one-action-out corroboration: ambient/uncorroborated change, the
animation class).  Pairs any scored arm certifies as carrying a
controllable effect are excluded (count reported).  Censored arms
never participate (no certified control pairing).  The differential is
measured over the FULL dually-hidden region — no component exists to
anchor to, and the unanchored region is a strict superset of the
anchored one, so every measured rate upper-bounds the detection
channel's.

Design-phase evidence (training corpus only — the gate corpora were
never touched): on deterministic stride samples of the pinned v4 label
corpus (`records[::67]`: 299 roots, 192 no-effect pairs; and
`records[::11]`: 1,819 roots, 1,063 no-effect pairs, 4 conflicting
pairs excluded), with the pinned pixel-head-v2 learned convention and
the assisted convention both applied:

- `identical_endpoints` NEVER fires: 0/177 and 0/1,018, both
  conventions — the phase discipline holds byte-exactly.
- `uncorroborated_change` fires at PARITY across conventions: 12/15 =
  0.80 (stride 67) and 35/45 = 0.778 (stride 11) for BOTH learned and
  assisted — the firing is a property of the class (the uncorroborated
  change is usually the player's own motion under whichever mask), not
  of one convention's extent.

Consequence, recorded before scoring: a single combined-rate bound
would be class-mix sensitive (the combined rate measured 0.033–0.063
purely as a function of the class mix) and would conflate "the raw
ingredient is unsafe without its anchor" with "one convention is
noisier than the other".  The preregistered discipline is therefore
per class, plus a paired no-regression condition:

- (d1) `identical_endpoints`: the learned differential fires on
  exactly 0 pairs (rate bound 0.0) — any fire is an instrument defect.
- (d2) `uncorroborated_change`: the learned rate is at most 0.95 (the
  published agreement operating point reused by import as a ceiling —
  a channel firing above it on certified-null changed pairs is
  indistinguishable from raw frame differencing, the v312/v313
  signature; the training-distribution expectation is ~0.78).
- (d3) no regression: the learned convention fires on no more no-effect
  pairs than the assisted convention over the same paired population
  (bit (c)'s form, under the identical upgrade).
- Instrument validity: >= 50 deduplicated no-effect pairs per corpus;
  an empty class leaves its condition vacuous (flagged), never passed
  silently.  All fired pairs are listed individually in the report.

**Run (fixed now).**  Same three probe corpora (v322/v323/v325), same
pinned artifacts as the v2 gate run (`pixel-mask-head-v2.pt`
`d4866931…` over frozen tracker v4 `b2fdd8ba…` and backbone
`642d66ed…`, digest cross-checks repeated at load), same ground-truth
extraction, CPU.  Driver:
`python -m lolo_agent.functional_mask_gate --detection-quantity v2`
(the v1 CLI path is untouched; the only performance-affecting change is
a larger convention mask LRU, which cannot alter any scored quantity).
One preregistered run, deterministic content-digested report (version
2, digest prefix `wp5-functional-mask-gate:v2:`) to
`experiments/lolo1-wp5/functional-gate-v3-report.json`, plus a
byte-identical determinism rerun to a scratch path (reported).  Honest
outcome either way: PASS → recommend shadow-promotion of the learned
masking convention TOGETHER WITH detection quantity v2 (one explicitly
gated convention change, engineering-internal, reversible, claim
boundary unmoved per §4.35); FAIL → name the failing mechanism per
corpus and bit against the v2 run's numbers.  Expected a priori and
disclosed: bits (b)/(c) cannot move (identical code and inputs), so a
NO-PROMOTE via the v323/v325 bit-(b) tail remains possible even if the
new quantity resolves bit (a) entirely; that outcome would confirm the
quantity hypothesis while leaving promotion blocked on the already-
documented placement-flip tail.

### WP5-final functional gate v3 (detection quantity v2) — results (appended 2026-08-16, after one preregistered run)

**FAIL overall — NO-PROMOTE — but bit (a) passes at 1.000 on every
corpus and v322 becomes the first corpus ever to pass ALL gate bits.**
Report `experiments/lolo1-wp5/functional-gate-v3-report.json`, content
digest
`01a9b128c90d509dece3370da1996fa9d3b83b4a182c8b0ca5694625761076b7`,
byte-identical on the preregistered determinism rerun (scratch path,
file sha256 `f49c3b6f…` equal).  No preregistration deviations.  The
preregistered internal consistency check holds exactly: the bit-(b),
bit-(c), stability, preservation, and divergence-telemetry blocks are
byte-identical to the v2 report on all three corpora, ground truth is
identical (1,175/1,563/1,750 detection measurements), the per-corpus
signature-channel rates equal the v2 run's v1-quantity rates to the
last digit (0.742979/0.781830/0.678286 learned), and the
signature/track-state views agreed on 100% of measurements.

| corpus | (a) v1-quantity → v2-quantity (assisted) | learned diff-only | (b) unchanged | (d) no-effect pairs / uncorr. rate | bits |
|---|---|---|---|---|---|
| v322 object-present | 0.743 → **1.000** (0.884 → **1.000**) | 302 of 1,175 | 0.975 PASS | 145 / vacuous (0 uncorr.) | **all PASS** |
| v323 pre-push | 0.782 → **1.000** (0.940 → **1.000**) | 341 of 1,563 | **0.936 FAIL** | 3,155 / **0.902** ≤ 0.95 | b FAIL |
| v325 object-removed | 0.678 → **1.000** (0.936 → **1.000**) | 563 of 1,750 | **0.943 FAIL** | 204 / vacuous (0 uncorr.) | b FAIL |

What the run establishes:

1. **The §4.38 quantity hypothesis is CONFIRMED, completely.**  Under
   the two-channel quantity the learned convention detects EVERY
   ground-truth manipulation on every corpus (0 misses in 4,488
   measurements); the differential channel alone carries 302/341/563
   detections the signature channel misses — the in-place erasure
   class, eliminated exactly as designed.  In-place erasure was a
   property of the detection quantity, not the mask.
2. **The incumbent's own 6–12% miss class was ALSO in-place erasure —
   now mechanistically demonstrated.**  Under the identical upgrade the
   assisted convention likewise reaches 1.000 everywhere (differential-
   only 136/94/112): §4.37's report-only observation that the assisted
   pipeline misses GT manipulations is explained by the same quantity
   blindness.  Both conditioned rates are 1.000/1.000 — the two
   conventions now agree perfectly on detection.
3. **Bit (d) — the raw-change-pathology guard holds.**  Identical-
   endpoints pairs never fire (145/459/204 pairs, 0 fires, both
   conventions — the phase discipline is byte-exact on the gate
   corpora too).  Only v323 has an uncorroborated-change class (2,696
   pairs, 142 conflicting pairs excluded): the learned differential
   fires on 2,431 = 0.902, under the preregistered 0.95 ceiling, at
   EXACT parity with the assisted convention (2,431 — no regression;
   fired changed-pixels median 107 inside a median 399-pixel region).
   Honest notes: the condition is vacuous on v322/v325 (flagged in the
   report — the discipline is genuinely exercised only by v323), and
   v323's 0.902 sits above the training-distribution expectation of
   ~0.78 though under the ceiling — the probe corpora's uncorroborated
   pairs are denser in player-adjacent change than the training mix.
4. **The verdict is blocked exclusively by the pre-existing bit-(b)
   tail**, numerically unchanged from the v2 run (v323 0.936, v325
   0.943 vs 0.95; v322 passes at 0.975): whole-pool placement flips at
   anchor/halo boundaries, the §4.38 path-(b) residual that
   probe-distribution strict collection may close without any
   convention change.  Bit (c) is also unchanged (learned 0.985–0.998
   vs assisted 0.720–0.769).

Disclosed channel-dominance observation (report rows, not gated): the
differential fires on nearly every GT measurement (1,175/1,175,
1,447/1,563, 1,748/1,750 learned) — whenever a convention's masks
cover changed component content in both endpoints the differential
sees it, and the signature channel covers the remainder — so under
quantity v2 detection is carried primarily by the new channel.  That
is the intended design (the masked region is exactly where the v1
quantity was blind), it is fully auditable per row, and the bit-(d)
discipline plus the component anchor are what keep it from collapsing
into raw frame differencing.

Consequence: per the preregistered criterion the learned masking
convention plus detection quantity v2 are NOT promoted; tracker v4 +
pixel head v2 stay telemetry-only.  The detection axis of WP5 is now
closed — every remaining promotion blocker is the bit-(b)
fingerprint-stability tail on v323/v325.  Paths this measurement
licenses next (each requiring its own preregistration, then THIS gate
rerun unchanged): (a) §4.38 path (b) — probe-distribution-targeted
strict collection against the placement-flip tail, no convention
change required; (b) if a future planner adopts quantity v2
downstream, that adoption inherits this gate's evidence but any use of
the differential OUTSIDE GT-anchored evaluation must carry the bit-(d)
discipline with it, preregistered.
