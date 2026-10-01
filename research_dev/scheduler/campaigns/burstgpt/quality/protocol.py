"""Frozen parameters of the output-quality noninferiority protocol (see PROTOCOL.md next to this file).

Changing any value here after the first confirmatory run invalidates the protocol; a change needs a new
PROTOCOL_ID and a new PROTOCOL.md."""

PROTOCOL_ID = "ws4-quality-gsm8k-ni-v1"

# dataset: GSM8K test split (MIT license), pinned to one commit of openai/grade-school-math
GSM8K_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_URL = {
    split: ("https://raw.githubusercontent.com/openai/grade-school-math/" + GSM8K_COMMIT
            + "/grade_school_math/data/" + split + ".jsonl")
    for split in ("test", "train")
}
GSM8K_SHA256 = {
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
}
GSM8K_ROWS = {"test": 1319, "train": 7473}

# item assignment: one seeded permutation of the split; shard s takes the next items in this order
ASSIGNMENT_SEED = 20260929
LLAMA_PER_SHARD = 2
GEMMA_PER_SHARD = 32
QWEN_PER_SHARD = 32
PLANNED_SHARDS = 8
INTERIM_SHARDS = 4
MAXIMUM_SHARDS = 12

# output budget: fixed per model by the pilot (train split, never scored): the smallest candidate whose
# token prefix holds the first answer of all but floor(PILOT_TRUNCATION_LIMIT x n) pilot outputs
OUTPUT_TOKEN_CANDIDATES = (256, 320, 384, 512)
PILOT_OUTPUT_TOKENS = 512
PILOT_SEED = 20260930
PILOT_PER_MODEL = 24
PILOT_TRUNCATION_LIMIT = 0.05

# confirmatory test: pooled Qwen + Gemma paired accuracy difference, treatment minus baseline
MARGIN = 0.03
DESCRIPTIVE_STRICTER_MARGIN = 0.02
ALPHA_ONE_SIDED = 0.025
TARGET_POWER = 0.80
PLANNING_DISCORDANCE = 0.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260929
MAXIMUM_MISSING_SHARE = 0.02
STRICT_ARM_MINIMUM_COVERAGE = 0.90
PER_PROTOCOL_MINIMUM_COVERAGE = 0.90

EXTRACTION_ID = "gsm8k-first-final-answer-v1"
CONFIRMATORY_ROLES = ("qwen", "gemma")
CONTROL_ROLES = ("llama",)
