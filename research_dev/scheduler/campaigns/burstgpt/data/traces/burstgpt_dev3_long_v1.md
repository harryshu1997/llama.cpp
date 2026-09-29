# Longer-generation development replay

Use `burstgpt_dev3_long_v1.json` when checking adaptation over a longer decode.
The short `burstgpt_dev4_5min_v1.json` and the 24-request parent are preserved.

| Combined index | Model | Arrival (s) | Input tokens | Output tokens |
| --- | --- | ---: | ---: | ---: |
| 36 | Gemma | 1 | 915 | 292 |
| 37 | Llama | 61 | 915 | 292 |
| 50 | Qwen | 91 | 277 | 71 |

These three requests are an ordered subset of `burstgpt_sparse_locality24_v1.json`.
Only replay arrivals change. The existing request contents, prompt tokens, output
lengths, source timestamps, and artifact assignments are unchanged. In particular,
there is no generated-token cap or artificial extension of a short request.
Total work is 2,107 input tokens and 655 output tokens, compared with 132 output
tokens in the short development trace.

The requests are from the existing BurstGPT-derived semantic-source and Llama
overlay files, not original BurstGPT prompt text. Llama request 37 is a derived
overlay request. Source paths and hashes are recorded in
[the short trace's provenance](burstgpt_dev4_5min_v1.md#provenance).

The long Gemma decode supplies a sustained adaptation opportunity. Llama and Qwen
arrive later to change live demand while Gemma may still be executing. This is not
a prescribed placement schedule: phone assignment, attachment, fractions, and
transitions remain scheduler decisions. It does not guarantee a reverse replacement,
simultaneous desktop execution of different models, or phone-only Llama selection.
Qwen's 71-token request is medium length, not a substitute for a long-Qwen soak test.

## Runtime budget and startup boundary

Keep the five-minute target only for a verified warm start: initial Gemma desktop
residency and compatible phone shards must already be physically READY before the
clock starts. This JSON does not perform that preload or alter the runner's paid
boundary. Merely having weight files on disk is not a warm start. If the launcher
cannot preserve and verify that residency, report a cold run instead; do not hide
loading afterward by subtracting it from the result.

Measured inference times from the matched 2026-09-07 v6 audit:

| Request | Desktop (s) | Adaptive (s) |
| --- | ---: | ---: |
| Gemma 36 | 155.048819 | 153.777118 |
| Llama 37 | 1.745302 | 1.777872 |
| Qwen 50 | 56.134053 | 54.949544 |

Taking the slower observation per request gives 212.960744 seconds of serial
inference. Adding the one-second first arrival, the observed 5.561847-second Llama
load and 65.744803-second Qwen load gives about 285.3 seconds, assuming phone
preparation overlaps. This leaves only 14.7 seconds of headroom, so it is a sizing
estimate, not a latency guarantee. A cold initial Gemma load adds about 73.3 seconds,
making the corresponding estimate 358.6 seconds (about six minutes).

Using both the 341-token Qwen request 43 and 292-token Gemma request 36 would require
about 388-406 seconds of serial inference in the reference observations, before
loading or queueing. Do not silently truncate either to claim a five-minute run.

Keep the existing live-VRAM desktop parents, F16 large-model artifacts, Llama Q4
artifact, CUDA graph mode, and phone shard-index flags. Use:

```sh
--replay-schedule research_dev/scheduler/campaigns/burstgpt/data/traces/burstgpt_dev3_long_v1.json
```

Remove conflicting request-index/arrival-scale options. For manifests, update
`trace.replay_schedule_path`, clear `trace.request_indices`, and set
`trace.arrival_scale` to null.

For a five-minute warm-start development cutoff, prefix the resolved runner
command with `timeout --signal=INT 300s` as described in the short trace's run
notes. A cutoff with incomplete requests is a failure, not a successful sample.
Normal cleanup may take additional time. If the test must include cold preparation,
allow a longer budget explicitly rather than implying this trace can guarantee
completion in five minutes.

No physical run of this variant has been performed. Any A/B comparison must use
these same three requests, arrivals, runtime identities, and matched initial
residency. Warm-start energy must be labelled separately from preload-inclusive
energy; neither the four-request nor 24-request baseline is a valid denominator.
