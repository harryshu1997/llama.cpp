# S15 observed BurstGPT B32 cohort results

## Verdict

`OBSERVED_BURST_CAN_FORM_OP15_B32_WITHIN_SYNTHETIC_5S_SLO`

This is an admission-feasibility result, not a physical-execution or energy
result.

## Frozen cohort

- source: normalized BurstGPT v2 burst trace, 11,550 real arrival records;
- eligibility: successful API-generation rows with input tokens > 0 and at
  least 8 observed output tokens;
- eligible rows: 7,783;
- measured OP15 B32 duration: maximum of seven processes = 2,933,672 us;
- synthetic relative SLO: 5,000,000 us;
- maximum admissible formation window: 2,066,328 us;
- densest eligible window: 60 requests from 247,000,000 to 249,000,000 us;
- selected cohort: first 32 requests in that window;
- selected formation delay: 1,000,000 us;
- predicted slack at launch on the final arrival: 1,066,328 us.

The trace has no request text, deadline, or priority. The fixed prompt, low
priority, and 5 s SLO are explicitly synthetic. Observed token counts are
selection metadata and are not presented as the physical input shape.

## Verification

- builder and validator use separate eligibility and window-selection logic;
- the normalized trace manifest binds the full trace digest and row count;
- every selected event binds its canonical source-row digest;
- every event binds the same fixed prompt digest used by the physical B32 gate;
- the model, device, layer range, seven latency samples, and conservative
  duration bind to the real-device gate report;
- 12/12 cohort tests pass, including duplicate request/key, changed source
  field, manifest row-count recomputation, nonidentical payload, false
  provenance, B31, forged duration, and forged schedule rejection;
- S15 runtime dispatch: 77/77 tests pass;
- S14 power-frontier policy: 18/18 tests pass.

## Remaining gate

The live launcher must execute these exact 32 request identities and input
manifest on OP15, persist raw route and placement evidence, and return exactly
one completion certificate per request. Server-side BGE concurrency and
selected-GPU energy are later matched controls.

Residency preflight may load the model, phone shard, and contexts before the
cohort arrives. It must not consume prompt bytes or start request computation
before those request IDs have arrived. The current LayerSplit `--wait-for-go`
path receives the prompt on its command line, so using it unchanged would prove
physical mechanics but not an arrival-faithful SLO result. A resident host-tail
input seam is required for that stronger claim.
