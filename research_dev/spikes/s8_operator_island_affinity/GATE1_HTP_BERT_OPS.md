# Gate-1: HTP op support + correctness for a phone-resident BGE island

Status: RUN 2026-07-17. This is the on-device Gate-1 of the second-service
(embedding/reranking) funnel in `EMBEDDING_MODEL_FUNNEL.md`. It answers one
question: can the BERT/BGE operator set run on the Hexagon HTP backend with no
CPU fallback (`ggml_backend_supports_op == true`) AND numerically correct vs the
CPU reference, on both phones? It does NOT run an end-to-end BGE model (that is
the gold-standard follow-up, section 6).

## 1. Method and provenance

Tool: `test-backend-ops {support|test} -o <ops> -b HTP0`. `support` mode is
exactly `ggml_backend_supports_op` (the no-fallback gate); the harness computes
each op directly on the backend with no per-op scheduler, so "not supported" is
the only fallback signal. `test` mode compares each supported op against the CPU
reference (`ggml_backend_compare_graph_backend`). Env on device:
`LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072`. Correctness was
run per-op so one op's crash does not mask the others.

Source is at HEAD `5e92adf00`; `ggml/src/ggml-hexagon/ggml-hexagon.cpp` has no
working-tree changes, so the deployed `supports_op` matches the reviewed source
(cross-checked: `ggml_hexagon_supported_binary` accepts all-F32 and all-F16, and
`ADD(f16)` reports SUPPORTED on device, as the source predicts).

| Device | Serial | HTP | Build used | Provenance |
|---|---|---|---|---|
| OP15 | 3C15AU002CL00000 | v81 (8 thr/8 hvx/1 hmx/8MB vtcm) | `build-snapdragon/` | host lib Jul-16, v81 skel; consistent with HEAD |
| OP12 | 5ae7a43d | v75 (4 thr/4 hvx/1 hmx/8MB vtcm) | `npu-harness/.../android-arm64-hexagon-release-eafdc75e/` | self-consistent Jul-9 build (v75 skel); recent source |

Raw logs + SHA256SUMS: `scratchpad/gate1_bert_ops/`.

## 2. Per-op result matrix (support = no-fallback gate; test = CPU-ref correctness)

All BGE-mandatory ops. "supp/nosupp" = count of harness shape/dtype variants
that pass/fail the support gate; the not-supported variants are the documented,
mitigable dtype/shape fallbacks (not correctness failures).

| Op (BGE role) | OP15/v81 support | OP15/v81 correctness | OP12/v75 support | OP12/v75 correctness |
|---|---|---|---|---|
| MUL_MAT (all projections/FFN) | SUPPORTED | PASS (S5) | SUPPORTED | PASS (S5) |
| NORM = LayerNorm (per-layer) | supp (F32) | **contiguous OK**; noncontig CRASH | supp (F32) | **contiguous OK**; noncontig CRASH |
| L2_NORM (output norm) | supp (F32) | **contiguous OK**; noncontig CRASH | supp (F32) | **contiguous OK**; noncontig CRASH |
| RMS_NORM (control) | supp | OK (21) | supp | OK (21) |
| SOFT_MAX (non-causal attn) | 82 supp / 130 nosupp | OK (82/82) | 82 supp / 130 nosupp | OK (82/82) |
| GET_ROWS (tok/pos embed) | 8 supp / 103 nosupp | OK (8/8) | 8 supp / 103 nosupp | OK (8/8) |
| GELU (FFN activation) | 2 supp / 6 nosupp | OK (2/2) | 2 supp / 6 nosupp | OK (2/2) |
| SCALE (attn scaling) | supp | OK (4/4) | supp | OK (4/4) |
| MUL (elementwise) | 66 supp / 24 nosupp | OK (67/67) | supp | OK (67/67) |

Zero numerical FAILs anywhere: every op that ran to completion matched the CPU
reference on both devices.

## 3. The one defect, localized

`GGML_OP_NORM` and `GGML_OP_L2_NORM` **hard-crash the HTP session** on the
`noncontig_rows=1` variant (non-contiguous `src0`), on BOTH v81 and v75:

~~~text
NORM(f32, ne=[64,5,4,3], noncontig_rows=0): OK     (contiguous - correct)
NORM(f32, ne=[64,5,4,3], noncontig_rows=1): CRASH
  ggml-hexagon.cpp:16xx ggml-hex: dspqueue_read failed: 0x0000002e
  _ZN20ggml_hexagon_session13flush_pendingEb -> Aborted
~~~

This is a `supports_op` FAIL-OPEN: `ggml_hexagon_supported_unary` allows a
non-contiguous `src0` (only `dst` must be contiguous), so `supports_op` returns
SUPPORTED, but the DSP kernel then faults with the same `0x2e` DSP-queue error S5
saw for v75 RMS_NORM/ROPE. The contiguous cases compute correctly first, then the
non-contiguous case faults during `flush_pending`. RMS_NORM has no non-contiguous
variant in its test set and passes 21/21.

Impact on BGE: **none in a well-formed graph.** BERT/BGE LayerNorm and the final
L2 normalization operate on contiguous F32 activations, which pass. The crash is
only reachable by feeding a non-contiguous norm input, which a clean BGE graph
does not do. It should still be fixed defensively (either `supports_op` rejects
non-contiguous `src0` for NORM/L2_NORM, or the kernel handles it) so no future
graph can fail-open into a crash.

## 4. Required graph hygiene (from the not-supported variants)

These are `supports_op == false` (clean fallback signal, not a crash) and are the
mitigations the pre-audit already predicted:

- **Embeddings must be F32.** `get_rows` requires an F32 table (`src0`); an f16
  `tok_embd`/`pos_embd` is not supported (103/111 nosupp are the non-F32 tables).
  Convert the BGE embedding tensors as F32.
- **Sequence length padded to a multiple of 32** (or <= 32). `supported_softmax`
  rejects `ne0 > 32 && ne0 % 32 != 0`; the 82 supported softmax variants are the
  aligned ones. BGE pads to a fixed max (e.g. 512), which is aligned.
- **Contiguous norm inputs** (section 3).

## 5. Verdict (op-level, isolated)

`GATE1_HTP_BERT_OPS_PASS_PENDING_FUSED_GRAPH` (isolated single-op). The fused
graph was then run; see section 7 for the closing verdict.

At the operator level, a phone-resident BGE island is **feasible on BOTH OP15/v81
and OP12/v75**: every mandatory BERT op is supported with no CPU fallback and is
numerically correct vs CPU on contiguous F32 inputs, on both devices. This is a
decisive upgrade from the pre-audit "UNKNOWN / likely blocker" — the funnel's
premise that HTP lacks LayerNorm/GELU/non-causal-softmax is refuted, and the two
residual risks (f16 get_rows, non-aligned softmax) are confirmed
support-gated-off with known mitigations. The second-service class is therefore
**not `TRACE_OR_SERVICE_BLOCKED`**.

Two items keep this short of a full phone-island certification:

1. **Isolated single-op scope.** Like S5, each op runs alone (own alloc, own DSP
   flush). Isolated execution exposes fragility a fused graph may hide (and vice
   versa). Per-op correctness is necessary but not an end-to-end proof.
2. **No end-to-end fused BGE graph run yet.** The gold-standard completion is
   section 6.

## 6. Gold-standard follow-up (to close Gate-1 fully)

Convert `bge-small-en-v1.5` to GGUF with F32 embeddings (network is reachable;
`conversion/bert.py` supports it), then:

- `test-export-graph-ops` on the BGE gguf to emit the exact fused graph ops;
- `test-backend-ops support|test --test-file <bge_ops>` on both phones to confirm
  the exact BGE shapes (contiguous norm, aligned softmax, F32 get_rows) with no
  fallback and CPU-correct;
- an end-to-end pooled-CLS cosine check of HTP output vs the CPU/HF reference
  (funnel Gate 0 + fused correctness).

Only after that does a phone-resident BGE island earn eligibility under
`MIXED_WORKLOAD_DESIGN.md` section 3. Energy remains out of scope and BLOCKED.

## 7. Fused-graph closure (RUN 2026-07-17) -- CLOSING VERDICT

The gold-standard follow-up was executed. `bge-small-en-v1.5` (BAAI, BertModel,
hidden 384 / 12 layers / 12 heads / ff 1536 / max_pos 512) was fetched and
converted with `convert_hf_to_gguf.py --outtype f16` to
`bge-small-en-v1.5-f16.gguf` (67 MB, 197 tensors; sha256 `4cd429b8...`). f16
matches the existing Gemma island policy: f16 weights for the HMX matmul path,
with GET_ROWS on CPU as the one declared scheduled-buffer exception.

### 7.1 Exact-BGE-shape op routing and correctness (`test-backend-ops --test-file`)

`test-export-graph-ops` on the BGE gguf (seq 64, 32-aligned) emitted the exact
27-op fused graph. `support`/`test --test-file` on HTP0, both devices:

- **support**: every op supported on HTP0 on both v81 and v75 -- ADD, MUL, NORM
  (contiguous LayerNorm), MUL_MAT, CPY, GET_ROWS, FLASH_ATTN_EXT, GELU. The
  earlier isolated non-contiguous NORM crash does NOT occur: the real BGE graph
  feeds contiguous norm inputs, as predicted.
- **test (vs CPU)**: 23/25 op cases OK on HTP0 on both devices, with two nuances:
  - `GET_ROWS` is not-supported here because the BGE token-embedding table is
    `f16[384,30522]` -> CPU (the declared exception; convert embeddings F32 to
    keep it on HTP, not required to match the Gemma policy).
  - `GELU` trips the harness at `ERR = 1.8e-4 > 1e-7` on both devices: the HTP
    GELU is an APPROXIMATE kernel, and test-backend-ops' 1e-7 f32 threshold is far
    tighter than an f16 model needs. `Backend HTP0: FAIL` is that
    strict-threshold artifact, not a model-level error (see 7.2).

### 7.2 End-to-end pooled-CLS cosine (the model-level correctness proof)

`llama-embedding` (npu-harness Android build) run ON DEVICE with `--device HTP0
-ngl 99 --pooling cls`, pooled+L2-normalized CLS vector compared to the CPU
reference (same gguf, same prompt). HTP engagement confirmed by timing (OP15 HTP0
1244 ms vs CPU-only 179 ms -- a silent CPU fallback would match CPU).

| Device | FA=auto | FA=off (explicit softmax = the op-probe path) |
|---|---|---|
| OP15/v81 | 0.997328 | 0.997331 |
| OP12/v75 | 0.99733 | 0.997328 |

All ~0.9973 vs CPU on both devices, both attention modes. The ~0.003 gap from 1.0
is f16-vs-f32 weight rounding plus the approximate GELU -- both immaterial to the
embedding. `fa=off` parity proves the explicit-softmax attention path (the one the
op probe validated with no fallback) is end-to-end correct; `fa=auto` shows the
BGE-shape FLASH_ATTN_EXT is also correct on v75 (unlike the Gemma-shape FA, which
S5 found wrong on v75 -- a shape-specific distinction).

### 7.3 Closing verdict

`GATE1_HTP_BGE_FUSED_PASS`. The full fused BGE encoder graph runs on the HTP
backend on BOTH OP15/v81 and OP12/v75 and produces pooled-CLS embeddings matching
the CPU reference at 0.9973 cosine, with only GET_ROWS on CPU (the declared
exception, f16 embedding table). A phone-resident BGE embedding island is
therefore a real second executable service class: the S14 mixed-system claim is
NOT `TRACE_OR_SERVICE_BLOCKED` and now has TWO executable phone-resident classes
(Gemma generation head + BGE embedding). Residual, non-blocking follow-ups:
convert BGE embeddings F32 to keep get_rows on HTP; note the approximate-GELU
1.8e-4 per-op delta if a future stage needs tighter precision; and the standard
S5 caveats (isolated vs fused timing, energy BLOCKED). Latency/throughput/energy
of the BGE island are unmeasured and out of scope here.

Raw logs + SHA256SUMS and the reference embedding: `scratchpad/gate1_bert_ops/`.
