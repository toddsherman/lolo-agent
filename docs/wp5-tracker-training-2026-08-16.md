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
