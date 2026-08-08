# Medium experiment: 2026-08-08

This experiment tested whether the duration-aware durable loop could move the
dataset beyond boot imagery and whether a frozen planner could autonomously
reach interactive gameplay.

## Configuration

- 10 collect–train–evaluate cycles on MPS
- 20 roots and 3 branches per root per cycle
- planning and training horizon 3
- controller durations 1, 2, 4, 8, and 16 frames
- one training epoch per cycle
- 600 persisted visual sequences
- 1,800 real branched controller transitions

Artifacts are stored locally in `experiments/lolo1-medium/` and intentionally
excluded from Git because they contain checkpoint data and game imagery.

## Dataset result

The collector stored 257 unique RGB observations spanning 23 coarse visual
scenes. Its exploration was balanced without supplied game semantics:

| Dimension | Counts |
| --- | --- |
| Actions | 168–220 uses per controller action |
| Durations | 344–379 uses per duration |
| Nonzero visual transitions | 623 / 1,800 |

Evaluator-side visual inspection confirmed that collection reached:

- the title and start/password interface;
- the castle sequence;
- the opening story sequence;
- Floor 1 with Lolo at multiple positions;
- a game-over/password state.

These labels were assigned only after collection and were never exposed to the
agent or model.

## Model result

Held-out RGB L1 prediction error improved across all three modeled decisions:

| Cycle | Horizon 1 | Horizon 2 | Horizon 3 |
| ---: | ---: | ---: | ---: |
| 1 | 0.363579 | 0.376114 | 0.375117 |
| 3 | 0.209504 | 0.206669 | 0.205028 |
| 5 | 0.162166 | 0.163127 | 0.160852 |
| 8 | 0.132716 | 0.133896 | 0.133228 |
| 10 | 0.110654 | 0.112996 | 0.113037 |

Ensemble disagreement continued to increase with horizon, but its correlation
with prediction error was `-0.0820` at cycle 10. Deep imagined rollouts must
therefore remain advisory and real first-action verification remains mandatory.

## Frozen-planner findings

The original 20-decision evaluation was too short and remained in intro
animation. Longer frozen audits exposed three general exploration failures:

1. action-independent animations produced timestamp branches that could restore
   the agent backward through a cutscene;
2. password-character edits generated effectively unlimited exact-frame
   novelty inside one coarse scene;
3. the duration-expanded verification budget could test several durations of
   one button while omitting other buttons.

The planner was changed without introducing game-specific concepts:

- matched equal-duration control probes detect autonomous visual dynamics;
- autonomous dynamics select the longest neutral wait and do not archive
  equivalent timestamp alternatives;
- novelty is hierarchical, reducing exact-frame reward as one coarse scene is
  repeatedly explored;
- archive pruning preserves minority scenes instead of allowing one scene to
  consume all memory;
- least-tested distinct controller buttons are verified before extra durations.

With these changes, the frozen cycle-10 checkpoint progressed through menus and
castle into the opening story sequence within 210 decisions. It did not yet
reach Floor 1 in frozen planning, so this is a navigation milestone rather than
a puzzle-solving result.

## Delayed-return and passive-sequence follow-up

The planner now detects delayed returns to informative visual signatures,
temporarily credits the intervening scene/action/duration choices with return
cost, and restores a distinct archived branch from the loop. This is attempt
memory only and does not modify the frozen world model. A 360-decision audit
detected 35 delayed visual returns and recovered from all 35.

The same audit exposed brief static pauses inside otherwise action-independent
visual sequences. A four-decision neutral grace window was added, along with a
fix ensuring equal-duration `NOOP` and non-neutral probes both survive branch
selection. The corrected 280-decision audit recorded:

- 27 delayed visual returns and 27 successful return recoveries;
- 158 autonomous-motion decisions and 53 neutral grace waits;
- 1,482 real verified branches, 134 telemetry frames, and 44 coarse scenes;
- an unchanged frozen parameter digest of
  `35909b3723be9a36859852aa80a324e0e9d94ce722e34b12b78238afd1d22dfe`.

Post-run evaluator comparison found no exact match to the three known Floor 1
collection frames and no match to their coarse room signature. The agent
therefore still has not reached Floor 1 in frozen planning. This is a negative
result: delayed-return recovery and passive-sequence persistence are working,
but temporary memory still lacks a reliable way to retain and prefer the
deepest persistent visual frontier over older story/menu branches.

The corrected run has deterministic high-speed players at
`experiments/lolo1-medium/extended_evaluations/cycle-000010-control-fixed-280/replays/`.
Replay verification passed for all 5,254 recorded observations; the committed
view contains 3,545 timeline frames and the full planning view contains 13,857.

## Persistent-frontier follow-up

Persistent-frontier learning is now implemented as temporary discounted
successor novelty. First visits to exact and coarse visual signatures are
credited backward through recent state traces. A delayed return discards
provisional gains within the loop and records a negative sample. Save-state
restoration and candidate scoring use the resulting values.

An initial audit revealed that count-based novelty still gave small positive
rewards to identical repeated pixels. The frontier reward was tightened to
first visits only, with a regression test proving that repeated static frames
produce exactly zero value. Values were then conditioned on
visual-signature/action/duration choices so a tried branch can override the
optimism inherited from its parent state.

The final 320-decision action-conditioned audit recorded:

- 286 successor-novelty updates;
- 21 delayed-return penalty events covering 240 traces;
- 110 learned state/action/duration return samples;
- 34 frontier-trace restarts following archive jumps;
- 1,716 real verified branches and 157 unique telemetry frames;
- a passing frozen-model digest audit.

Evaluator-only post-run checks again found neither the Floor card's coarse
signature nor any exact or coarse Floor 1 match. The committed trajectory was
unchanged from the state-only frontier audit. Exact visual signatures are too
specific to transfer a learned choice value between neighboring animation
frames, even when those frames depict the same underlying situation.

The verified high-speed players are under
`experiments/lolo1-medium/extended_evaluations/cycle-000010-action-frontier-320/replays/`.
Replay checked all 6,075 observations. The committed view contains 3,623
timeline frames and the full planning view contains 15,703.

## Frozen-encoder abstraction follow-up

The temporary frontier now keys values by online clusters in the frozen visual
encoder rather than exact frame hashes. Clustering is constrained to frames
with the same coarse visual signature. Offline calibration over the previous
audit measured same-scene latent RMSE at 0.00089 median and 0.03287 at the 90th
percentile, versus 0.2649 at the 10th percentile for different scenes. A 0.04
threshold was therefore selected before this audit.

The first real audit exposed two independent control bugs. A known negative
action value was being erased by zero-valued state optimism, and remaining in
one coarse scene was treated as stagnation even while the screen continued to
animate. The corrected controller lets a sampled action value override state
optimism and triggers stagnation recovery only after repeated exact visuals.

The corrected 320-decision audit recorded:

- 157 exact telemetry frames assigned to 51 temporary latent clusters;
- 285 successor-novelty updates and 107 learned choice samples;
- 23 delayed visual returns, all followed by recovery;
- 35 total archive restores, 108 autonomous-motion decisions, and 107 grace
  waits;
- 1,710 real verified branches and a passing frozen-model digest audit.

The committed trajectory changed and reached the final visible story tableau,
showing Lolo beside the clouds and castle. Evaluator-only comparison still
found neither the Floor card nor an exact or coarse Floor 1 match, so the agent
has not yet entered the first puzzle room. A follow-up that increased waiting
time after every repeated-cluster recovery regressed to an early static logo;
that heuristic was rejected rather than retained.

The deterministic players are under
`experiments/lolo1-medium/extended_evaluations/cycle-000010-visual-stagnation-320/replays/`.
Replay verification passed for all 6,056 recorded observations. The committed
view contains 3,676 timeline frames and the full planning view contains 15,592.

## Behavioral-equivalence follow-up

Purely visual clusters are now only candidate sets. Each current saved state is
tested with two fixed, equal-duration controller probes. The agent compares the
resulting source-to-successor latent displacements and shares frontier evidence
only below a 0.02 mean RMSE threshold. Unprobed successors keep unique
provisional signatures; values and active traces migrate only after behavioral
compatibility is observed. Calibration on the preceding audit placed the 95th
percentile for distinct frames in one visual cluster at 0.020116.

The 320-decision frozen behavioral audit recorded:

- 201 exact frames and 59 coarse scenes, up from 157 and 44;
- 67 visual clusters refined into 72 behavioral clusters across 285 planning
  roots, with zero deferred classifications;
- 285 provisional-to-behavioral frontier migrations and 117 learned choice
  samples;
- 25 delayed returns and recoveries, 35 total archive restores, 151
  autonomous-motion decisions, and 79 grace waits;
- 1,710 real verified branches and an unchanged frozen parameter digest.

The committed trajectory reached the final story tableau at decision 131, but
did not match the evaluator-only Floor 1 coarse signature
`040604030303040302`. Behavioral refinement broadened exploration without
advancing beyond the prior deepest frontier. A follow-up that forced a third
equal-duration discrimination probe displaced too much ordinary exploration:
it fell to 112 exact frames and 38 scenes and never reached the final tableau.
That variant was rejected.

The deterministic players are under
`experiments/lolo1-medium/extended_evaluations/cycle-000010-behavioral-abstraction-320/replays/`.
Replay verification passed for all 6,056 recorded observations. The committed
view contains 3,861 timeline frames and the full planning view contains 15,967.

## Active-probing follow-up

The equal-duration probe budget is now active rather than fixed. One neutral
anchor remains stable while the second probe rotates through the least-observed
controls in each visual cluster. Once multiple behavioral hypotheses exist,
the agent instead selects the control with the largest observed
successor-centroid separation. A sole hypothesis can accumulate a new probe
through a unique neutral-anchor match; ambiguous partial profiles remain
provisional.

The matched 320-decision audit recorded:

- 287 probe selections: 219 coverage rotations and 68 hypothesis-separation
  choices;
- 92 conservative anchor expansions, 126 full-profile matches, 69 new
  behavioral clusters, and zero ambiguous deferrals;
- 179 exact frames, 56 coarse scenes, and 64 visual clusters;
- 146 learned choice samples, up from 117 with fixed probes;
- 22 delayed returns and recoveries, 33 total archive restores, 133
  autonomous-motion decisions, and 86 grace waits;
- 1,722 verified branches and an unchanged frozen-model digest.

The committed trajectory reached the final story tableau at decision 123,
eight decisions earlier than the fixed-probe run. It explored slightly fewer
exact frames and scenes and again did not match the evaluator-only Floor 1
signature. Active probing therefore improved behavioral identification and
time-to-frontier, but not the deepest game progress.

The deterministic players are under
`experiments/lolo1-medium/extended_evaluations/cycle-000010-active-probes-320/replays/`.
Replay verification passed for all 6,094 recorded observations. The committed
view contains 3,752 timeline frames and the full planning view contains 16,188.

## Temporal-option follow-up

The planner now traces action-independent sequences from their possible
initiating choice through neutral continuation to the next controllable state.
Endpoint novelty, coarse-scene span, duration, and return-to-source evidence
form a temporary option sample. Candidate and archive scores can reuse the
learned state/action/duration value without changing persistent parameters.

A preliminary 320-decision audit learned five samples. The largest assigned
`0.75` to a `START@4` choice followed by 51 passive decisions across 28 coarse
scenes. Causal inspection of the saved branches showed that every tested action
at that root had produced identical pixels: the button press coincided with a
timer transition but did not demonstrably cause it. Reusing the value later
over-selected `START` on the same static pixels. This unsupported result was
rejected.

The retained controller now credits an initiating choice only if another real,
same-duration action branch produced distinguishable pixels. The corrected
matched audit recorded:

- 38 passive-option starts, six controllable completions, and 32 traces
  discarded at explicit archive jumps or run close;
- 32 choices with immediate action-dependent counterfactual evidence;
- zero credited option samples, because none of those evidenced choices
  immediately preceded a completed passive sequence;
- 192 exact telemetry frames and 56 coarse scenes, versus 179 and 56 in the
  active-probe baseline;
- 159 learned persistent-frontier choice samples, 30 delayed-return recoveries,
  39 total archive restores, and 1,686 verified branches;
- an unchanged frozen-model parameter digest.

The trajectory again reached the final story tableau at decision 123. The
evaluator-only Floor 1 coarse signature `040604030303040302` was absent, so the
agent still did not enter the first puzzle. This is a negative temporal-option
result, but it establishes the causal guard needed to avoid learning value from
coincidental animation timing.

The deterministic players are under
`experiments/lolo1-medium/extended_evaluations/cycle-000010-causal-temporal-options-320/replays/`.
Replay verification passed for all 5,980 recorded observations. The committed
view contains 3,409 timeline frames and the full planning view contains 16,004.

## Delayed-counterfactual follow-up

The evaluator now clones one same-duration alternative whose immediate pixels
match the committed action. If passive dynamics begin on the next decision,
the factual and cloned states receive identical neutral inputs and durations.
Every intermediate contrast is logged, but option value is recorded only when
the states remain visually different at the next controllable endpoint.
Cloning keeps the original alternative in the ordinary archive, so the causal
experiment does not perturb later recovery choices.

The matched 320-decision audit recorded:

- 81 delayed counterfactuals armed and 68 paired neutral steps;
- six completed passive sequences, including the 51-decision, 28-scene story
  transition;
- zero endpoint differences and therefore zero delayed option samples;
- 192 exact frames, 56 coarse scenes, 39 archive restores, and 1,686 verified
  branches, matching the immediate-causal baseline;
- 2,116 native save states created and all 2,116 released;
- an unchanged frozen parameter digest.

The result confirms that the observed passive sequences were timer-driven in
this trajectory. In particular, the branch paired with the apparent
`START@4` story initiation stayed pixel-identical for all 51 matched steps, so
the button correctly received no value. The final story tableau again appeared
at decision 123, and the evaluator-only Floor 1 signature remained absent.

The deterministic players are under
`experiments/lolo1-medium/extended_evaluations/cycle-000010-delayed-counterfactual-options-320/replays/`.
Replay verification passed for all 6,427 recorded observations. The committed
view contains 3,409 timeline frames and the full causal-planning view contains
17,307.

## Evaluator-owned first-room bootstrap

The title and story sequence does not exercise the puzzle mechanics under
study, and earlier frozen runs repeatedly exhausted their decision budget in
that sequence. A trace-derived input sequence was therefore minimized by
replacing all non-`START` inputs with neutral frames and greedily removing
unnecessary `START` pulses. The retained fixture is:

```text
NOOP@254, START@4, NOOP@21, START@1, NOOP@944
```

It is guarded by the exact Lolo 1 ROM digest, endpoint frame digest
`cff8e18b...e4a78`, and Floor 1 coarse signature `040604030303040302`.
The evaluator applies it as attempt 0; attempt 1 starts with clean agent memory
on the resulting pixels. It is opt-in, fully logged, included in deterministic
playback, and excluded from agent action statistics.

A 20-decision native audit from this handoff recorded 120 verified branches,
26 exact telemetry frames, seven coarse scenes, and 140 save states created and
released. The first committed choice changed the first-room pixels, confirming
controller ownership after attachment. The checkpoint digest remained
`35909b37...1d22dfe`. Replay verification passed for all 427 recorded
observations; the committed view contains 1,497 timeline frames and the full
planning view contains 2,372.

Artifacts are under
`experiments/lolo1-medium/extended_evaluations/cycle-000010-first-room-bootstrap-smoke-20/`.

The subsequent 320-decision audit confirmed that initialization was no longer
the limiting factor. It recorded 1,668 verified branches, 233 exact telemetry
frames, nine coarse scenes, 42 archive restorations, and 1,997 save states both
created and released. Replay verification passed for all 6,082 observations.
The committed trajectory contains 4,367 frames; the full planning trace
contains 17,353.

It did not solve Floor 1. Visual inspection showed the initial upward move
entering the left enemy's path, repeated fade/restart sequences, and the life
display decreasing from five to three. The planner treated those changing
pixels as novelty and selected 163 neutral waits, so the result exposes a need
to learn temporary value for preserving controllability and avoiding
action-caused visual collapse without supplying a death label.

The durable experiment runner now applies the same opt-in bootstrap before
collection and frozen evaluation. A new `lolo1-puzzle` experiment initialized
from cycle 10 collected 60 sequences and 61 unique frames whose first source
was the exact Floor 1 endpoint; bootstrap transitions were telemetry-only. One
training epoch reduced batch loss from `0.1771` to `0.1453`, but held-out
horizon-three pixel L1 worsened from `0.1307` to `0.1377`. Its 80-decision
frozen audit still chose 49 neutral waits and did not solve the room. More
cycles alone are therefore not yet evidence of useful progress.

## Next research target

Learn progress potential and controllability survival from directed
reachability rather than passive duration or raw pixel novelty alone. The
temporary interaction graph can identify visual regions that are easy to leave
but not recover through tested controller transitions, action choices followed
by loss of control and a visual restart, and states whose descendants repeatedly
expand the reachable frontier. Ranking branches by this evidence may avoid
repeated lethal exploration without assuming object identity, death, room
completion, or success.
