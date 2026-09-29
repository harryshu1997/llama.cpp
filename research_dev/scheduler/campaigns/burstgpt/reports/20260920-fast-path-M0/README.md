# Fast-path M0: retune the host baseline

Status: **PASS - all M0 checks completed**. The resumed tuned pair retains the
phone energy advantage, with identical outputs at ubatch 1024. Proceed to M1.
OP15 uses ADB 5037 exclusively. Single runs unless stated otherwise.

| Required M0 check | Result |
| --- | --- |
| Prefill energy minimum within 18 GiB | PASS: ubatch 1024, 45.6422% less prefill host energy than 128; no OOM |
| Decode J/token minimum | PASS: eight unpinned threads, 100.24743267234257 J/token, including the step 1 reference |
| Warm 9.63 GB restore < 2 s and unchanged argmax | PASS: 0.225794 s; every native arm has identical argmax and logits |
| Tuned 0% versus 100% phone split | PASS: 11.9685% request host-energy saving; all 64 tokens identical at ubatch 1024 |

## Prefill sweep

The pair-v1 document and suffix produce 9,737 prompt tokens; each request asks for
64 output tokens. Qwen3-14B dequantized F16, 16 GPU layers, 32,768 context cells,
one slot, batch 2048, eight decode and eight batch threads. The existing KV planner
keeps CPU KV on layers 0-31 and GPU KV on 32-39 (4 GiB / 1 GiB allocation).
Sweep ubatch 128, 256, 512 and 1024, each in a fresh 18 GiB user scope with swap
disabled. The existing gate hashes the model then advises its file cache away
before each server launch. No phone executes in this sweep.

Host energy is measured RAPL package plus NVML board at 10 Hz. Request, prefill,
decode and paid load-plus-request windows remain separate. The gate also records
assumed phone-idle energy at 0.875 W; it is not included in measured host energy.
The tuned phone comparison below states assumed active power separately.

Deployment: `/mnt/storage/s42-fast-path-M0-20260920-LuVndR/` on
`zhihao@172.20.74.85`. Step 1 freezes the latest validated split-KV runtime from
`/mnt/storage/s42-split-kv-20260918-WwiIPY/cuda-build/bin`; no native rebuild yet.
Runtime/model hashes, exact server argv, listener ownership and the typed launch
contract are recorded per arm. CPU settings enter the runtime-launch digest.

Exact scope commands are in `RUN_UBATCH_SWEEP.sh`; inputs are in `config/`.
Launch from the shared workspace with:

```sh
ssh -o BatchMode=yes zhihao@172.20.74.85 \
  'bash /mnt/storage/s42-fast-path-M0-20260920-LuVndR/RUN_UBATCH_SWEEP.sh'
```

The gate takes the shared execution lock with `LOCK_EX | LOCK_NB`, chooses a
free server port, verifies the listening PID and argv, and closes its server in
`finally`. The four completed sweep records are in `physical/`.

| ubatch | Prefill s | Prefill host J | Assumed phone idle J (0.875 W) | Decode ms/token | CUDA compute MiB | Host compute MiB | OOM events |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 289.281266473 | 30889.412287359897 | 253.12110816387502 | 776.9830625 | 194.50 | 16.50 | 0 |
| 256 | 180.360740975 | 22080.806689886318 | 157.815648353125 | 779.85734375 | 219.00 | 33.01 | 0 |
| 512 | 133.143075633 | 18654.31646966702 | 116.500191178875 | 792.758921875 | 268.00 | 66.01 | 0 |
| 1024 | 108.995955457 | 16790.798791870282 | 95.37146102487499 | 783.36284375 | 366.00 | 132.02 | 0 |

Selected ubatch: **1024**, the measured prefill-energy minimum, 45.6422% below
128. Every arm completed 9,737 + 64 tokens. All scopes hit the 18 GiB cap and
reclaimed file-cache pages. Recorded peak charge is 19,327,352,832 B (128/512)
or 19,327,356,928 B (256/1024, a one-page transient overshoot); no swap or OOM.
The full numeric summary is `UBATCH_SUMMARY.json`.

Outputs at 128/256/512 are identical. The 1024 greedy output first differs at
zero-based index 44 (token 45). This is preserved, not classified as token
equality or task-accuracy qualification. The restore comparison must hold the
chosen launch settings fixed when checking argmax equality. Each arm is a
single run; loading times varied and are excluded from the prefill table.

## Thread sweep

Measured at ubatch 1024 and batch 2048, with batch threads fixed at eight.
`THREAD_SWEEP_PLAN.json`, `config/threads-*.json` and `RUN_THREAD_SWEEP.sh`
record the exact arms. Affinity is applied by the existing typed launch
contract through `taskset`.

| Decode threads | Allowed CPUs | Topology |
| ---: | --- | --- |
| 8 | 0,2,4,6,8,10,12,14 | Eight physical P-cores |
| 12 | 0,2,4,6,8,10,12,14,16-19 | P-cores plus four E-cores |
| 16 | 0,2,4,6,8,10,12,14,16-23 | All physical cores |
| 24 | 0-23 | All cores and SMT siblings |

| Decode threads | Decode ms/token | Decode host J/token | Prefill s | Request host J | Assumed phone idle request J (0.875 W) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 785.849125 | 101.64222320444287 | 105.816266647 | 23071.54805155423 | 136.597795366375 |
| 12 | 746.824171875 | 101.10500244764776 | 133.974220304 | 24159.7306504333 | 159.05086325974997 |
| 16 | 717.660296875 | 102.886021720831 | 115.998526861 | 23868.574325467005 | 141.688812941375 |
| 24 | 716.754234375 | 103.8009401296758 | 191.482668498 | 26310.483987628893 | 207.686591763 |
| 8, unpinned (step 1 reference) | 783.36284375 | 100.24743267234257 | 108.995955457 | 23206.634482900205 | 139.240901594375 |

Selected decode threads: **8, unpinned**, the lowest measured decode J/token
at ubatch 1024, including the already completed step 1 reference. Twelve was
the minimum within the pinned sweep (0.53% below eight pinned threads), but
still used 0.86% more J/token than the unpinned reference. These are single-run
observations, not statistically established differences. The affinity change also
affects batch workers: prefill and full-request energy did not improve over the
eight-thread arm. All five greedy outputs are identical and all scopes have
zero OOM/OOM-kill events. Exact records are in `THREAD_SUMMARY.json` and
`physical/threads-*/`.

```sh
ssh -o BatchMode=yes zhihao@172.20.74.85 \
  'bash /mnt/storage/s42-fast-path-M0-20260920-LuVndR/RUN_THREAD_SWEEP.sh'
```

## Software checks before step 1

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. \
  python3 -m unittest test_kv_decode_relocation_gate test_llama_server_adapter \
  test_kv_placement test_decode_split_selection
PYTHONPATH=/tmp/fast-path-pyflakes python3 -m pyflakes \
  research_dev/scheduler/campaigns/burstgpt/reports/20260917-decode-relocation-kv-headroom/kv_decode_relocation_gate.py \
  research_dev/scheduler/tests/test_kv_decode_relocation_gate.py
```

Local: 69 tests passed; pyflakes clean. Rig: 69 tests passed with one
environment-dependent skip. Logs preserve the initial test-import typo and
the rig check attempted before its source copy completed; both were corrected
before physical execution.

## Restore implementation and checks

The existing mapping release keeps `MADV_DONTNEED`; typed launch fields
`ffn_host_share_drop_cache` and `ffn_host_share_populate` independently control
file-cache advice and explicit population. Both default to 1 for legacy
behavior. M0 trials explicitly select 0/1 (keep cache and populate) or 0/0
(keep cache and fault on access). Invalid values, unsupported model backing,
missing release ownership and old servers that do not confirm the requested
policy fail closed. The settings enter the launch and selection identities.
Deferred population ends the release credit at the normal local boundary;
its returned byte count does not assert physical residency.

`BUILD_RESTORE.sh` builds a fresh CUDA runtime with the existing FFN transport
enabled. `MATERIALIZE_TRANSPORT.sh` verifies the current phone binary hashes,
preserves the six measured USB receipts and binds the rebuilt host stack and
its actual transport source to a new identity. This is identity materialization,
not a new transport throughput measurement.

`RUN_RESTORE_PROBES.sh THREADS CPU_LIST` measures the legacy release, retained
cache with population, and retained cache with demand faults. The last two
have warm and pressured arms; pressure is a touched 24,576-cell KV allocation
held in an 18 GiB scope. These are memory-occupancy probes with a seven-token
prompt and two decode steps, not served 24k prompts. Restoration happens before
the repeated local decode; that decode's duration includes deferred page faults.
The probe sweep uses 12 pinned threads (the pinned sweep's minimum); the tuned
server pair uses eight unpinned threads after including the step 1 reference.

| Restore policy | KV cells touched | Restore s | Following local replay s | Argmax / logits identical |
| --- | ---: | ---: | ---: | --- |
| Legacy cache drop, populate | 0 | 35.698425 | 13.219456 | yes / yes |
| Keep cache, populate | 0 | 0.225794 | 2.272551 | yes / yes |
| Keep cache, populate | 24576 | 20.015170 | 19.690335 | yes / yes |
| Keep cache, demand faults | 0 | 0.000002 | 2.435795 | yes / yes |
| Keep cache, demand faults | 24576 | 0.000003 | 40.755803 | yes / yes |

**Warm restore subcheck: PASS**, 0.225794 s < 2 s for 9,625,706,496 bytes.
Every arm returned the same argmax `[4180, 13, 2160]`, with zero logit difference
and zero OOM/OOM-kill events. The pressure arms touched 4,026,531,840 B of KV
(3 GiB host, 0.75 GiB GPU). The chosen policy is keep cache plus explicit
population (0/1). Deferred population clears the logical release state; its
microsecond return does not populate pages, and pressured execution remains
expensive. These single runs start with different file RSS, so the table is
not a controlled estimate of a small speed difference between the pressure
variants. Full residency and cgroup records are in `RESTORE_SUMMARY.json` and
`physical/restore/`.

Exact rig invocation:

```sh
ssh -o BatchMode=yes zhihao@172.20.74.85 \
  'bash /mnt/storage/s42-fast-path-M0-20260920-LuVndR/RUN_RESTORE_PROBES.sh 12 0,2,4,6,8,10,12,14,16-19'
```

The local probe built successfully. The existing CPU build has the server
target disabled; its initial combined build command therefore failed after
building the probe, and that log is retained. The rig CUDA server and probe
builds passed after adding the omitted `scripts/` build support to the deploy
copy; the first build's missing-UI-script failure is retained separately.
The final local check passed 82 tests in 6.422 s (dormant native policies, lazy
KV, release contracts, gate identities, launcher, selection and coordinator);
pyflakes is clean on all touched Python files. All four native cache/population
combinations preserve tiny-model logits and argmax exactly. The rig repeated
all 82 tests successfully in 6.457 s, with pyflakes clean. Logs are in
`software/` and `physical/unit-restore-rig.log`.

Transport materialization completed successfully. New identity:
`sha256:4ee0362b6b7f2629e0508d36377dbee9e4b956ec4160ea4b4c8e3cf8245e69ca`.
The phone kernel, session, worker, resident workers and router were checked
against the prior qualification. Their hashes and all six receipt hashes
match; rebuilt host hashes and the actual current transport source are in
`TRANSPORT_QUALIFICATION_IDENTITY.json`.

The tuned 0%/100% phone comparison was attempted via `RUN_TUNED_PAIR.sh`, using
`config/tuned-pair.json`: ubatch 1024, batch 2048, eight unpinned decode and
batch threads, phone FFN maximum four rows, and cache/population policy 0/1.
Both arms were configured for the rebuilt runtime and the original 9,737 + 64
request, combined first then control. The first launch refused the occupied
rig lock before starting either arm (`physical/tuned-pair-lock-busy.log`).
The retry waited with the existing bounded nonblocking option. The unrelated
Gemma policy-C campaign then reported an expert-service readiness failure and
no forced cleanup. After obtaining the lock, this gate failed at its first
phone preflight command (`adb -P 5037 -s 3C15AU002CL00000 shell 'uname -r'`):
the device was not found. Neither phone preload nor desktop launch began.
The raw failure and power samples are in `physical/tuned-pair/combined/`.

At 2026-09-20 02:23:35 EDT, ADB port 5037 still listed no devices, this gate's
scope was inactive, and no M0 server or probe remained running. No foreign
process, USB session or phone boot state was changed. At that time the required comparison was unavailable, so M0 stopped for the
user's decision. The user subsequently restored OP15 and authorized the resumed
pair below. The earlier failure is retained in `physical/tuned-pair/` and
`physical/CHECK_M0_PREVIOUS_PREFLIGHT.json`.

## Resumed tuned pair: PASS

On 2026-09-20 the user returned OP15 to desktop ADB 5037 and stopped the stray
ADB server on 5038. This run used only 5037. Read-only preflight found a sleeping
foreign Gemma service; it was left untouched. Our g2 FFN gadget was free.
Preflight and cleanup records are `physical/RESUME_PREFLIGHT.json` and
`physical/RESUME_CLEANUP.json`. The existing check passed all 82 tests in
7.818 s with pyflakes clean immediately before the pair.

`RUN_TUNED_PAIR.sh` verified the materialized host binary/library/source hashes,
then ran combined followed by control in one fresh 18 GiB scope under the rig
lock. Both arms used ubatch 1024, batch 2048, eight unpinned decode/batch threads,
the same KV plan and the freshly built server. Each generated a new 64-token
output from the same 9,737-token prompt. Phone release uses keep-cache + populate
(0/1); the control makes no dormant release. The old ubatch-128 output was not
used as a reference, as instructed by the user.

| Decode split | Prefill s | Request host J, measured | Decode ms/token | Decode host J/token | Decode host W | Phone W, assumed | Request phone J, assumed |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0%, tuned host | 109.141627444 | 22967.92225269669 | 779.72196875 | 98.32215973656352 | 126.09580004114238 | 0.875 idle | 139.164461989 |
| 100%, OP15 on FFN layers 0-17 | 123.924118139 | 20219.003291561952 | 789.379421875 | 55.27746710360097 | 70.02513588918862 | 4.5 decode; 0.875 otherwise | 335.779260380125 |

Measured request host energy falls **11.968513872916764%**. Decode host energy
falls 43.77923832053022%; decode time rises 1.2385765070185517%. The explicit sum
of measured host energy and the separate phone assumption falls 11.045547170260539%.
Even assuming 4.5 W for the phone throughout the entire combined request gives
785.004188634 assumed phone J and a 9.101446930361534% saving in that explicit sum.
These are one pair's observations, not repeated-run estimates. Prefill energies
were 16675.304029556628 J control and 16681.24539693149 J combined despite the
longer combined prefill time.

Cold load/preparation plus request is a separate interval:

| Decode split | Paid s | Paid host J, measured | Phone J, assumed active only during decode | Phone J, assumed active for entire paid span |
| --- | ---: | ---: | ---: | ---: |
| 0% | 277.068830509 | 25672.974286478286 | 242.435226695375 | 242.435226695375 |
| 100% | 332.861744657 | 23522.097304269624 | 474.39358360949996 | 1497.8778509565 |

Both outputs have hash
`sha256:949f68abbc25705dd76f1d858524599a2e66128e5c3cd90d8b27f8c2642d515b`;
the complete token lists are identical. The phone control applied at token index
2, so native proofs require and show 62 assisted rows per layer, 18 layers,
372 calls in each of HTP0/HTP1/HTP2, 1,116 completed transport requests and zero
resets. The server released 9,625,706,496 B in 0.101666 s. No OOM or swap;
combined peak was 19,327,352,832 B, and the scope's later control peak was one
page above the cap at 19,327,356,928 B. The phone closed normally and restored
to ADB 5037. At 15:46:41 UTC the scope was inactive, the lock was free and no
owned server remained. No foreign process was modified.

Full precision summary: `TUNED_PAIR_SUMMARY.json`; acceptance: `physical/CHECK_M0.json`.
Raw records, including both new reference outputs, are in `physical/tuned-pair-resume/`.
Reproduce the local acceptance calculation with:

```sh
bash research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M0/physical/TUNED_PAIR_ANALYSIS.command
```

Build, identity and validation commands, from the shared workspace:

```sh
bash research_dev/scheduler/campaigns/burstgpt/reports/20260920-fast-path-M0/DEPLOY_RESTORE.sh
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M0-20260920-LuVndR/BUILD_RESTORE.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M0-20260920-LuVndR/MATERIALIZE_TRANSPORT.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M0-20260920-LuVndR/CHECK_RESTORE.sh'
ssh -o BatchMode=yes zhihao@172.20.74.85 'bash /mnt/storage/s42-fast-path-M0-20260920-LuVndR/RUN_TUNED_PAIR.sh'
```
