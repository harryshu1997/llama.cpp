# S23 dense observed-arrival runtime

## Objective

Replace the six-request all-at-once S22 trace with a provenance-bound burst
from BurstGPT v2 and execute it through the real OP15, OP12, and RTX 4060 Ti
StageNet workers.

## Scope

- Preserve observed arrival times and input/output token counts.
- Label priority, SLO, prompt content, and the short execution shape synthetic.
- Reuse finite sequence slots only after exact KV removal.
- Admit requests at their observed arrival, ordered by priority and deadline.
- Keep the existing fixed `[0,8)` prefix and shared `[8,48)` tail routes.

The first physical replay is an arrival-pressure mechanics test. BurstGPT does
not publish prompt text, and many rows exceed the current 512-token per-stream
context. It therefore executes one synthetic input token and four output steps
per request while retaining observed token demand as non-executed metadata.

## Checkpoints

- [x] CP0: select the earliest maximum-density 2-second BurstGPT window.
- [x] CP1: bind the source trace, manifest, row digests, and synthetic sidecar.
- [x] CP2: add dynamic head and shared-tail sequence-slot leasing.
- [ ] CP3: pass offline fail-closed tests and reproduce trace bytes.
- [ ] CP4: execute all 60 arrivals on OP15, OP12, and one RTX 4060 Ti.
- [ ] CP5: profile a context-compatible cohort with observed token lengths.

## Claims

CP4 may establish only arrival-faithful routing, slot reuse, continuous
ready-row batching, SLO outcomes, and physical three-device execution. It does
not establish full BurstGPT request replay, output quality, or energy savings.
