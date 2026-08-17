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
