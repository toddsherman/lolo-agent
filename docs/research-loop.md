# Evidence-gated research loop

Long unattended searches are no longer the default. Every experiment is a
bounded cycle:

1. Write one falsifiable hypothesis and the evidence that would change the
   plan.
2. Declare wall-clock, telemetry-event, per-cycle dollar, and campaign-dollar
   ceilings before execution.
3. Run through `lolo-research-cycle`; the supervisor terminates the entire
   process group when a ceiling is reached.
4. Inspect the generated logs and telemetry audit.
5. Record an immutable reflection with a `continue`, `revise`, or `stop`
   decision.
6. Start another cycle only if its hypothesis exactly matches the prior
   reflection's `next_hypothesis`.

This gate prevents an automated job—or a conversational request to
"proceed"—from silently extending a paid run. A model change must be tied to
measured evidence, and a negative result is a valid outcome.

## Run a cycle

Copy `configs/research-cycle.example.json` and use a unique `cycle_id` and
unique telemetry path. Then run:

```bash
lolo-research-cycle run \
  --plan /absolute/path/to/cycle.json \
  --campaign-dir /absolute/path/to/campaign
```

Each cycle contains the normalized plan, stdout/stderr, machine-readable and
Markdown reports, and lifecycle state. The campaign ledger accumulates
estimated compute spend. The estimate uses the declared hourly rate and wall
time, so storage, network, and idle-Pod charges must still be checked in the
provider dashboard.

## Reflect before continuing

Copy `configs/reflection.example.json`, replace every example statement with
the evidence from the report, and run:

```bash
lolo-research-cycle reflect \
  --campaign-dir /absolute/path/to/campaign \
  --cycle-id runpod-benchmark-001 \
  --reflection /absolute/path/to/reflection.json
```

The reflection cannot be overwritten. A `stop` decision permanently closes
that campaign. For `continue` or `revise`, the next plan must name the prior
cycle and repeat the next hypothesis exactly. This creates an auditable chain
between observation and spending.

## Reflection questions

- Did the expected evidence appear, and is it causal rather than correlated?
- Which failure mode consumed most branches or wall time?
- Did a learned representation transfer across states, or only memorize a
  location?
- What is the smallest change that distinguishes the leading explanations?
- What result will stop this line of investigation?
- Is the next measurement cheaper on the Mac, or does it need a GPU?

Large training runs should follow a cheap smoke test, then a short comparative
pilot. Increase a budget only after throughput, learning signal, and artifact
durability have all been demonstrated.

On RunPod, invoke the cycle through `scripts/runpod-bounded-cycle.sh`. It adds
an outer watchdog and stops the Pod at the end, covering paid resource time in
addition to the experiment subprocess.
