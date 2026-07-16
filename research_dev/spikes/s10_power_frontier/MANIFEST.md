# S10-V0 MANIFEST (ASCII)

Verdict: FAIL. HEAD 933c722f6 (unchanged). No commit/push/stage/reset/revert.
Date: 2026-07-15. Devices: A6000 x2 (driver 580.159.03); OP12 5ae7a43d (HTP v75,
USB 6-2); OP15 3C15AU002CL00000 (HTP v81, USB 8-3); adb 127.0.0.1:5037.

## Directory layout

```
research_dev/spikes/s10_power_frontier/
  PLAN.md                     frozen contract (pre-existing)
  RESULTS.md                  verdict-first results (this run)
  MANIFEST.md                 this file
  schemas/                    instance.schema.json, certificate.schema.json
  oracle/    model_data.py (shared sim/energy), oracle.py (exact enumerator)
  checker/   checker.py (STANDALONE certificate validator; imports no solver/sim)
  policies/  policies.py (C0..C5)
  fixtures/  frozen/ (primary + sweep/ 44), mutation_tests.py, gen_fixtures.py,
             generated_examples/ (first 5 gen instance+cert pairs)
  scripts/   s10_gpu_ffn_atlas.cpp (CP2 harness), build_atlas.py, freeze_instance.py,
             run_cp4.py, verdict.py, cp0_ffn_repro.sh
  artifacts/ all raw evidence (below)
```

## Code hashes (sha256)

```
09b945e2...fd34 oracle/model_data.py
73244a98...60cf oracle/oracle.py
a66eac29...16a3 checker/checker.py
59150753...d371 policies/policies.py
ba92b9ef...ee05 fixtures/gen_fixtures.py
e2ee1978...b8a2 fixtures/mutation_tests.py
66413a08...3f74 scripts/build_atlas.py
f79f8525...e9b7 scripts/freeze_instance.py
d6cec985...bc5b scripts/run_cp4.py
df2a4536...8326 scripts/verdict.py
74ab23f3...9979 scripts/s10_gpu_ffn_atlas.cpp
be6c836e...66af schemas/instance.schema.json
c8455ba3...fcab schemas/certificate.schema.json
```
Full list incl. artifacts: `artifacts/cp_file_hashes.txt`.

## Reproduction commands and exit codes

Prereq: `source npu-harness/.venv/bin/activate`. Run from the spike dir unless noted.

CP0 (from repo root):
```
bash scripts/cp0_ffn_repro.sh 3C15AU002CL00000 9101 19101 artifacts/cp0_ffn_op15.json   # PRESTAGED_FFN_PASS
bash scripts/cp0_ffn_repro.sh 5ae7a43d        9102 19102 artifacts/cp0_ffn_op12.json   # PRESTAGED_FFN_PASS
ctest --test-dir build-phone-pim -R phone-pim        # 3/3 (pre and post edit)
ctest --test-dir build-phone-pim-asan -R phone-pim   # 3/3 (pre and post edit)
```

CP2 (from repo root; build the standalone harness, then measure):
```
gcc -O2 -I examples/gguf-hash/deps -c examples/gguf-hash/deps/sha256/sha256.c -o /tmp/sha256.o
g++ -O2 -std=c++17 -I ggml/include -I examples/gguf-hash/deps -I examples/gguf-hash/deps/sha256 \
    -I examples/phone-pim research_dev/spikes/s10_power_frontier/scripts/s10_gpu_ffn_atlas.cpp \
    examples/phone-pim/phone_pim_ffn.cpp /tmp/sha256.o -L build-cuda/bin -lggml -lggml-base \
    -o /tmp/s10_gpu_ffn_atlas
LD_LIBRARY_PATH=build-cuda/bin /tmp/s10_gpu_ffn_atlas --model scratchpad/phone_pim/12b-f16-mid-2-3.gguf \
    --model-bytes 464114176 --sha256 5cfba18d...360d --prefix blk.2 --backend CUDA0 \
    --sweep 1,2,4,8,16,32,64,128,256,512,1024 --reps 300     # -> cp2_a6000_ffn_latency.jsonl
python scripts/build_atlas.py                                # -> cp2_atlas.json  (exit 0)
```
(The compiled binary + sha256.o are reproducible and were removed from the tree.)

CP1/CP3/CP4/verdict (from spike dir):
```
python scripts/freeze_instance.py                # primary x2 + 44 sweep instances (exit 0)
python oracle/oracle.py --instance fixtures/frozen/primary_favorable.json --out artifacts/cert_primary_favorable_C4.json
python checker/checker.py --instance fixtures/frozen/primary_favorable.json --certificate artifacts/cert_primary_favorable_C4.json  # valid, exit 0
python fixtures/mutation_tests.py                # 13/13 caught, exit 0
python fixtures/gen_fixtures.py --n 1200         # 1200 valid, 0 wrongly rejected, 1200 corruptions rejected, exit 0
python scripts/run_cp4.py                        # every policy cert checker-validated; prints gate table (exit 0)
python scripts/verdict.py                        # verdict FAIL (exit 0) -> artifacts/cp_verdict.json
```

Observed results: CTest 3/3 + 3/3 (pre and post edit); mutation self-test 13/13
CAUGHT; generated fixtures 1200/1200 valid, 1200/1200 corruptions rejected; CP4
opportunity gate met only under favorable+lone+slack (non-robust), mechanism gate
FAIL (no measured batch/cap/low-power lever), physical gate ENERGY_BLOCKED.

## Artifacts (artifacts/)

- cp0_integrity.txt, cp0_gpu_inventory.txt, cp0_gpu_power_probe.txt, cp0_hashes.txt,
  cp0_usb.txt, cp0_phones.txt, cp0_pretests_{release,asan}.txt, cp0_preexisting_tracked.diff,
  cp0_ffn_op15.json, cp0_ffn_op12.json
- cp2_summary.txt, cp2_atlas.json, cp2_a6000_ffn_latency.jsonl, cp2_a6000_power_trace.csv,
  cp2_a6000_sustain.jsonl
- cp3_mutation_selftest.txt, cp3_fixtures_summary.jsonl
- cp4_results.jsonl (46 rows: primary x2 + 44 sweep), cp4_gate_summary.json
- cp_verdict.json, cp_file_hashes.txt, cp_postedit_tests.txt
- cert_primary_{favorable,conservative}_C4.json (example oracle certificates)

## Frozen model / worker provenance

- Island: gemma4 dense FFN blk.2, n_embd 3840, n_ff 15360, f16 weights 353,925,120 B.
- Model shard: scratchpad/phone_pim/12b-f16-mid-2-3.gguf, 464,114,176 B, SHA-256
  5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d.
- Certified worker on both phones: 0a50ca72..e749 (verified intact, untouched).

## Protected scope statement

No edits to tools/server, llama-server, model graphs, KV internals,
ggml_backend_sched, HTP/OpenCL/CUDA kernels, or protocol-v3 wire. No new build target
added to the tree. No commit, push, stage, reset, revert, or PR. All new files are
under research_dev/spikes/s10_power_frontier/. Pre-existing dirty tree preserved
byte-for-byte (cp0_preexisting_tracked.diff, re-verified identical after all edits).
