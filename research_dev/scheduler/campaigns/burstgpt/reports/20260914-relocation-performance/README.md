# Relocation: memory, energy and context priorities

The user accepts the observed coherent cross-hardware wording difference and
asks that it no longer block performance work. Per-layer numerical diagnosis
is paused. This is an experimental acceptance decision, not proof that every
possible numerical difference is harmless. Existing artifacts and exact-token
FAIL records remain unchanged.

## Narrow gate change

The existing gate now supports `--output-comparison semantic-sanity`. Both arms
must still complete their requested output lengths and pass the existing HTTP
semantic-sanity checks. Token agreement remains in the result, but mismatch
alone does not fail this explicit mode. The default `exact` mode remains for
reproducing historical gates. No scheduling, attachment, fraction, generation,
memory, lease, proof or cleanup policy changed.

Future performance gates use:

```text
--output-comparison semantic-sanity --diagnostic-top-logprobs 0
```

Malformed/degenerate output, request failure, incomplete output, stale ownership,
invalid proofs and memory overcommit still fail closed. Semantic sanity is not
a full task-accuracy benchmark. No old physical result was relabelled PASS.

Changed existing code:

- `campaigns/burstgpt/remote_resident_gate.py`: explicit completion acceptance mode.
- `tests/test_remote_resident_gate.py`: mismatch recording, default exact mode,
  incomplete/failed requests and invalid-mode regressions.

15 focused gate tests PASS in 7.509 s. The saved v4 result remains PARTIAL;
evaluating its completed rows under the new explicit mode passes while retaining
the exact first-divergence record. This is a software replay of an acceptance
decision, not a new physical run. No broad suite, native rebuild or deployment.

## Starting evidence

The real phone already owns 2,831,155,200 bytes of Gemma FFN tensors. The server
unmaps 2,831,056,896 complete pages with zero overlap. This is host RAM, not GPU
VRAM: process GPU memory remains 12,654,215,168 bytes. It is not yet proof of
greater usable context length.

`HOT_REQUEST_DIAGNOSTIC.json` analyzes the two already-warm requests from the
non-logprob v4 run. The first request is excluded because only its phone arm
includes a desktop load transition. The source result remains unchanged.

| Assumed active phone power | Desktop fleet energy | Phone-path fleet energy | Diagnostic saving |
| --- | ---: | ---: | ---: |
| 3 W | 3.328 kJ | 2.780 kJ | 16.48% |
| 4.5 W | 3.328 kJ | 2.826 kJ | 15.10% |
| 6 W | 3.328 kJ | 2.872 kJ | 13.72% |

These are two short requests (55 output tokens), one sample each, not a frozen
matched performance experiment. Server package/board energy is measured. Phone
activity is conservatively assumed for the entire assisted request boundary;
the desktop arm includes the same phone at the declared 0.875 W idle power.
Preparation and final cleanup are excluded here. This diagnostic cannot establish
end-to-end savings or amortization. Aggregate request duration is 28.985 s versus
30.617 s, so the phone path is 5.63% slower despite its lower estimated energy.
Probability-instrumented runs are deliberately rejected by this analyzer.

## Next bounded measurements

1. Freeze a same-revision paired performance specification. Use the same model,
   native libraries, prompt/token IDs, seed, output length, GPU/CPU placement,
   context, batch and disclosed file-cache policy on both arms. Keep normal
   identity/admission checks; no numerical-probability instrumentation.
2. Measure a warm execution interval only after both respective parents are
   READY. Record TTFT, prefill/decode latency, CPU/GPU energy, phone activity and
   throughput. The current gate's first reduced request includes loading; it
   cannot be compared directly with its full-arm execution-only interval.
3. Separately measure complete start-through-cleanup intervals. Include phone
   staging/loading, verification, desktop loading, execution and cleanup. Do not
   derive steady state by subtracting overlapping intervals. Report phone
   sensitivity at 3/4.5/6 W and compute break-even reuse only if the measured
   steady-state saving is positive.
4. Assess long-context capacity by memory pool before increasing context. Use
   the scheduler's live host/GPU admission and actual KV/workspace allocations.
   Then run a real long prompt plus decode; a server starting with a larger
   context setting is not a successful capacity result. Keep the full parent as
   the paired capacity control and disclose any identical KV-placement changes.
5. After that evidence, consider additional FFN relocation under the existing
   memory/session planner. Measure whether it improves energy, capacity or
   speed; do not assume all three improve simultaneously.

The current `_capacity_gate` launches servers but does not execute a long prompt.
Phone mode correctly refuses that path. It must not be enabled as-is or reported
as proof of longer usable context. The measured 2,560-token Gemma parent has
non-sliding KV on both CPU and GPU (20 MiB each), plus separate sliding-window
KV buffers (264 MiB CPU, 216 MiB GPU). Reclaimed host RAM receives no GPU memory
credit, and GPU/context limits can still bound the result. No larger-context
capacity or speedup is claimed now.

No 24/84-request trace, new physical inference, process interference, reboot,
commit or push in this step. A bounded matched performance test is next; broad
KV-cache redesign is not part of this acceptance change.

Before-image archive SHA-256:
`b30ce3ade3367e79c8a413aee9ebc5f2787d60e3d653792da5586b0b9b36e43e`.
Source physical result SHA-256:
`9ca16f45a21e4015831f01e772211313e1eacd48b81c2862f4675f05846e25c5`.
