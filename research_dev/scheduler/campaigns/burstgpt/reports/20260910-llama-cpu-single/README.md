# One full Llama request on desktop CPU

One physical request completed on the RTX desktop's Intel i9-12900K CPU.
This is a latency measurement, not a scheduler qualification or matched energy
comparison. No phone execution or existing worker process was changed.

## Request and execution

- Model: Llama-3.2-1B-Instruct-Q4_0.
- Saved request: combined request 37, overlay request 4.
- Input: the original 915 token IDs; output: all 292 requested tokens.
- Seed 42, temperature 0, ignore EOS, prompt cache disabled.
- Eight CPU threads and eight prompt-processing threads, one server slot.
- CPU-only: `--device none --n-gpu-layers 0 --no-kv-offload --fit off`.
- Context 2048, batch 2048, microbatch 512.
- One completion submission, no HTTP warm-up or repeated request.
- Ordinary server initialization is recorded separately from request latency.

The rendered prompt exactly matches both earlier phone-memory probe requests.
The server reported zero cached input tokens, 915 evaluated prompt tokens and
292 generated tokens. The output is readable; this is not a downstream quality
evaluation.

| Metric | Measurement |
| --- | ---: |
| Startup to healthy endpoint | 0.908 s |
| Prompt processing | 1.677 s |
| Time to first streamed token | 1.680 s |
| Decode time | 8.188 s |
| Decode throughput | 35.661 tokens/s |
| Decode time per token | 28.042 ms |
| HTTP request wall time | 9.869 s |

This supersedes the two-token CPU warm-up as evidence for this full request.
That earlier warm-up reported 71.84 tokens/s and was not representative of the
full prompt and decode.

## Historical speed references

| Backend | Decode tokens/s | Decode time, 292 tokens |
| --- | ---: | ---: |
| CPU, this run | 35.661 | 8.188 s |
| CUDA, previous phone-memory probe | 206.608 | 1.413 s |
| Phone OpenCL, previous phone-memory probe | 4.689 | 62.278 s |

The CPU decode was 7.61 times faster than that earlier phone configuration.
These are same-prompt historical references, not fresh matched runs or proof
of the fastest phone configuration. No repeated-run variance was measured.

RAPL package energy over this request interval was 1,112.439 J. GPU board and
whole-phone energy were not measured in this run, so this is not fleet energy
and must not be compared directly with earlier fleet totals.

## Evidence

Remote root: `/mnt/storage/s42-llama-cpu-single-20260910-jhjinU`.

- `probe_cpu.py`: the executed one-request probe.
- `physical/RUN_COMMAND.json`: exact command, server, library and model hashes.
- `physical/REQUEST.json`: exact HTTP request body.
- `physical/SOURCE_REQUEST.json`: unmodified source request metadata and tokens.
- `physical/completion.raw`: full streamed response.
- `physical/server.stderr`: request and intermediate decode timings.
- `physical/RESULT.json`: measured summary.
- `physical/CLEANUP.json`: the owned server exited with code zero.

Model SHA-256:
`4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad`.

Raw stream SHA-256:
`7c4556eb2ce415db1abc5b61e33b3ce443876b0a89ba8bdd3fcfac4f4f26d68b`.

The desktop had no inference process before this probe. The probe terminated
only its own server, and no inference process remained afterward. No scheduler
production code, phone state, GDM process, commit or push was changed.
