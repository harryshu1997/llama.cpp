# OP15-fenced GPU prefetch physical receipts

This bundle contains the 2026-08-10 RTX 4060 Ti plus OP15 qualification for a
disabled-by-default, protected Gemma weight transfer during Qwen FFN phone
windows.

`PINNED_PREFETCH_ABBA_V1.json` is the matched 256 MiB
observe-prefetch-prefetch-observe result. `PREFETCH_QUALIFICATION_V1.json`
re-hashes that result, the complete tied-output-tensor pilot, and every raw
resource-sample file. `raw/` retains the Qwen request results, phone energy
captures, bridge and helper logs, server logs, one-second resource samples,
and raw response streams used by the analyzers.

The admitted claim is only
`FENCE_TRANSFER_QUALIFIED_NO_WEIGHT_ADOPTION`. The helper owns the CUDA
destination, so Gemma cannot execute from it. No incremental dynamic energy
saving is claimed. The next gate is a Gemma-owned destination that is staged,
verified, published, and used before matched full-trace energy comparison.
