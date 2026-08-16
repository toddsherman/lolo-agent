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
