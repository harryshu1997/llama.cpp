# Canonical B8 history materializer

This pre-phase tool materializes all 64 corpus items in the frozen V2.2
MMLU64 order. It formats each prompt with the pinned candidate prompt template
and invokes one exact, digest-bound `llama-tokenize` executable against the
pinned Qwen3-14B Q4_K_M artifact. The captured command template is
`llama-tokenize -m MODEL --ids -f PROMPT_FILE --log-disable`.

The output records canonical UTF-8 prompt bytes and hashes, token arrays,
logical item and sequence IDs, and eight consecutive B8 groups. Each group
uses wire request IDs 1 through 8, whole-position-wave prefill partitions of
at most 64 rows, and seven exact B8 decode call descriptors. The final prefill
row for each request yields continuation token 0; the seven decode calls yield
continuation tokens 1 through 7. `mechanics_b8` is exactly the first quality
group.

Acquisition producers must consume these recorded call descriptors without
padding, truncation, repartitioning, or reordering.

The materializer and validator reopen and hash the corpus, candidate, model,
and tokenizer. The validator independently reruns tokenization. These tools do
not execute a model route and do not authorize A_ONLY acquisition.
