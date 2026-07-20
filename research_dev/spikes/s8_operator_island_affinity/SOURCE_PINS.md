# Source Pins (S8-V0b-P0)

Status: sources fetched, inspected, and pinned OUTSIDE git. This records the exact
provenance and the column-level facts observed by a full single pass over each
file. It is the input to the versioned source configs under `configs/`. The
normalizer is NOT implemented here.

Sources live outside the repository at `/home/myid/zs89458/Documents/s8_sources/`
(not committed, not under the worktree). Re-fetch with the commands in section 3.

## 1. BurstGPT v2.0 -- BurstGPT_3.csv

| field | value |
|---|---|
| source id | `burstgpt-v2` |
| provenance | `real` |
| download URL | https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv |
| release/revision | tag `v2.0` (immutable release asset) |
| filename | `BurstGPT_3.csv` |
| bytes | 231682327 |
| sha256 | `2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f` |
| lines (incl header) | 5344022 |
| data rows | 5344021 |
| license | CC-BY-4.0 (repo `LICENSE`) |

Observed facts (full pass, `csv` module, UTF-8, no BOM):
- Header (exact, 8 columns): `Timestamp,Session ID,Elapsed time,Model,Request tokens,Response tokens,Total tokens,Log Type`
- `Log Type` distinct: `Conversation log` (233617), `API log` (5110404).
- `Model` distinct: `GPT-4` (790239), `ChatGPT` (4553782).
- `Timestamp`: all match `^[0-9]+\.[0-9]+$` (float seconds); min 19440110.0, max
  28943983.0; **0 non-monotonic pairs** -> policy `require_nondecreasing`.
- `Response tokens == 0` (failures): 387963 (7.26%). KEPT in the primary trace,
  flagged `source_fields.burstgpt_failed:true`. A no-failure replay is a SEPARATE
  sensitivity config only (explicit `exclude_zero_output` filter), never the
  default.
- Negative tokens: 0.
- Blank `Session ID`: 5110404 (exactly the `API log` rows) -> mapped to `null`.
  `Conversation log` rows carry a UUID session.

## 2. RAGPulse -- data/0_trace.jsonl

| field | value |
|---|---|
| source id | `ragpulse` |
| provenance | `real_decomposed` |
| download URL | https://raw.githubusercontent.com/flashserve/RAGPulse/99a62769a91d5ebd17a2d4ddbbc88c1d16edc0e8/data/0_trace.jsonl |
| commit (immutable) | `99a62769a91d5ebd17a2d4ddbbc88c1d16edc0e8` (last touched `data/0_trace.jsonl`, 2025-11-17) |
| filename | `data/0_trace.jsonl` |
| bytes | 1923473 |
| sha256 | `cd371571bef3320907147f8901729e37f412aafb067ec8eb153e2828e1801e65` |
| lines | 7107 |
| records | 7106 |
| license | MIT (repo) |

Observed facts (full pass, JSONL, UTF-8):
- The pinned file ends with exactly `\n\n`: 7106 JSON records followed by one
  empty terminal physical line (7107 LF-terminated physical lines total). The
  source config permits only this single terminal blank; interior or multiple
  blank lines remain invalid.
- Every record has exactly these top keys: `timestamp` (str), `input_length`
  (int), `output_length` (int), `session_id` (str), `hash_ids` (object).
- `hash_ids` has exactly 5 keys in every record:
  `sys_prompt`, `passages_ids`, `history`, `web_search`, `user_input`, each a
  list of ints. `passages_ids` and `web_search` may be empty `[]`; the other
  three were always non-empty in this file.
- `timestamp`: all match `^[0-9]+$` (integer seconds); min 0, max 604603
  (~7 days); **1 non-monotonic pair** -> policy `sort_stable` (deterministic
  stable sort by `(t_us, source_row_id)`; the inversion count is recorded in the
  sidecar). This is why the per-source timestamp policy exists: a global
  `require_nondecreasing` rule would reject RAGPulse.
- `session_id`: never null, never blank (all 7106 present, non-empty).

## 3. Re-fetch / re-verify commands

```sh
DATA=/home/myid/zs89458/Documents/s8_sources
curl -L -o "$DATA/burstgpt_3.csv" \
  https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv
sha256sum "$DATA/burstgpt_3.csv"   # 2299986a...a43b8f ; 231682327 bytes

RP=99a62769a91d5ebd17a2d4ddbbc88c1d16edc0e8
curl -L -o "$DATA/ragpulse_0_trace.jsonl" \
  "https://raw.githubusercontent.com/flashserve/RAGPulse/$RP/data/0_trace.jsonl"
sha256sum "$DATA/ragpulse_0_trace.jsonl"   # cd371571...801e65 ; 1923473 bytes
```

The exact column/key mappings, value maps, defaults, timestamp grammar, and
per-source policy are frozen in `configs/burstgpt.config.json` and
`configs/ragpulse.config.json`, validated by
`schemas/source_config.schema.json`. Nothing derived here is a deadline, priority,
model profile, or performance result.
