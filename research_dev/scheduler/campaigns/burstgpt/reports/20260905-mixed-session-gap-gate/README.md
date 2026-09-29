# Current-shard mixed-session gate, 2026-09-05

Status: FAIL on the unchanged retained-session maximum-gap assertion.
Forward replacement and retained service succeeded. Gemma execution,
reverse replacement, injected rollback, and the warm matched A/B were not
reached. No trace was run. This is not a completed mixed-model acceptance
gate and provides no new energy-saving claim.

The phone is back on normal USB at 5,000 Mbps. Gate-owned desktop processes
have exited. GDM is untouched, using 3,178 MiB; live free VRAM is again
12,770 MiB. No commit, push, worker rebuild, shard generation, or old-artifact
overwrite occurred.

## What passed before the timing failure

| Observation | Measured result |
| --- | --- |
| Cold Qwen probe | Completed, 4,878 phone calls |
| Qwen reuse/replacement probe | Completed, 3,876 phone calls |
| Dynamically selected replacement | HTP2, Qwen generation 1 -> Gemma generation 2 |
| Retained HTP0/HTP1 | Qwen, generation 1 unchanged |
| Retained calls before/during/after replacement | 210 / 114 / 1,512 on each retained session |
| Physical replacement load-to-READY | 9.665491 s |
| Final physical load counts | HTP0=1, HTP1=1, HTP2=2 |
| Phone reset recoveries / fallback executions | 0 / 0 |

Both probes use the original 341-output-token Qwen 43 shape, with the
existing harness's distinct request IDs and seeds. They are lifecycle probes,
not an A/B pair. The Gemma 3 shape was used as scheduler demand for planning;
its physical execution was not reached. Gemma's terminal call count is zero.
The existing fraction sweep contains 0/25/50/75/100%, with no fraction-induced
reload. Fraction-weighted coverage acceptance follows the failing gap check
and was not reached; no new coverage PASS is claimed.

The completed forward stage's authoritative target matches the physical map
exactly. Recomputing its authorization from the recorded source and target
maps reproduces the stored assignment hash:
`sha256:842aa5ab43727646360a00eb7cbf16482ed198643f0ec7fd00b618366e393747`.
Artifact, endpoint, layer mask, column width, resident bytes, geometry,
operator plan, and session generation are checked. Generation-1 Qwen proofs
remain valid after HTP2 becomes Gemma generation 2. The final router terminal
contains both HTP2 identities, with zero Gemma calls and no stale execution.

The Qwen session mask is acknowledged before loading: drain policy bound at
78.551400 s, QUIESCED with allowed sessions HTP0/HTP1 at 92.772888 s,
preparation stage started at 92.926078 s, and helper rebound at 103.032310 s.
These are the online controller's timestamps. The request executes from
42.000550 through 238.517625 s, spanning the entire replacement. Its desktop
parent stays unchanged. The retained sessions are not physically reloaded
or re-verified.

Cold preload verifies first-session service during the next load: HTP0 has
119 direct call events while HTP1 loads. A deliberate desktop campaign
restart after the cold request preserves the physical phone state. There is
one phone router launch/terminal, not a global phone restart. Its five host
connections are request/mask handshakes, not endpoint process restarts.

| Session | Cold physical load-to-READY | Generation after replacement |
| --- | ---: | ---: |
| HTP0 | 11.236648 s | 1 |
| HTP1 | 11.537864 s | 1 |
| HTP2 | 18.940445 s | 2 |

First-session readiness is 11.236648 s and all-session readiness 54.010667 s
after the first physical LOAD_AUTHORIZED. The latter includes this gate's
explicit serving check between stages. The full cold phase is 270.549279 s,
including desktop setup and its concurrent validation request; it is not
reported as phone load time or preparation-only energy.

All loads use the existing indexed F16 FFN files. The mixed target contains
9,248,440,320 tensor bytes against an admitted FFN limit of 9,625,939,968 bytes.
HTP2 opens Gemma's stored layers 16-23, all CPU-resident under the exact parent
whose first GPU layer is 26. Stored/active width is 15,360 columns and the
file is 2,831,157,504 bytes. Retained Qwen shards each have 17,408 columns and
3,208,644,448 file bytes. Full source, index, parent, and shard hashes and
all phase timestamps are retained in PARTIAL_GATE_AUDIT.json.

## Why the strict gap check fails

| Retained session | Median before load | Maximum before load | Maximum during load | Required limit |
| --- | ---: | ---: | ---: | ---: |
| HTP0 | 14.906 ms | 428.747 ms | 463.378 ms | 29.811 ms |
| HTP1 | 15.649 ms | 418.508 ms | 461.254 ms | 31.298 ms |

These are unsampled phone monotonic timestamps, not desktop log-arrival times.
The reference is exactly 30 preceding call intervals. The maximum includes
the intervals crossing each transition boundary.

Each retained shard serves six consecutive layers per decode token. Of the
30 reference intervals, 25 are between layers within a token, and five are
between tokens. The median consequently measures the fast within-token path,
while the maximum usually measures the intervening desktop/other-session
work before the next token's first layer. The no-load reference already
violates the prospective 2x-median limit by more than an order of magnitude.

All 3,876 online native FFN calls were joined to the phone's generation-keyed
call counters. Cold totals plus online per-session totals equal the terminal
phone totals exactly. Both worst gaps cross a token boundary:

- HTP0: layer 5 -> 0. During its 463.378 ms physical gap, 454.480 ms elapsed
  on the host before issuing the next RPC; that RPC took 8.635 ms, including
  7.909 ms of phone computation.
- HTP1: layer 11 -> 6. During its 461.254 ms gap, 451.157 ms elapsed before
  the next RPC; that RPC took 10.125 ms, including 9.674 ms of computation.

The largest within-token gaps during loading are 20.798 and 20.142 ms.
The largest token-boundary gaps increase by 8.08% and 10.21% relative to the
pre-load maxima. That is descriptive, not an isolated estimate of load
interference: the active session mask and other desktop work also matter.
The small clock-alignment differences are retained in the two-clock raw
RPC decomposition; the acceptance calculation itself uses only phone time.

No acceptance rule was changed. Optimizing USB/HTP load arbitration alone
cannot remove the normal host-side token gap already present before loading.
Artificially slowing within-token calls to inflate the reference median
would not be a valid fix. A proposed next measurement is to check within-token
and same-layer token-to-token cadence separately, with enough like-for-like
reference intervals at the same fraction/mask. That requires user agreement;
it has not been implemented or used to relabel this run PASS.

## Code and tests

Only measurement/persistence and focused tests changed this turn:

- `campaigns/burstgpt/offline_residency_gate.py`: read phone per-call monotonic
  times; retain phase clocks and failed-gap measurements; gather 31 new calls
  for the required 30-interval pre-load reference. The post-load service check
  remains one call; request completion still ends the wait immediately.
- `tests/test_burstgpt_replay.py`: clock-domain, failed-gap-persistence, and
  missing/foreign-generation call regressions; extend the existing phase fixture.
- `research_dev/talks.md` and this report directory.

The previous desktop-derived timing helper is left in place as diagnostic
code. No transaction, authorization, adaptive fraction, route-selection,
worker, or shard-format code was changed. The existing
S42_RESIDENT_CALL_LOG_PERIOD=1 is explicit in COMMAND.json and scoped to the
gate subprocess, not a new hidden global setting.

Focused command:

```sh
env PYTHONPATH=research_dev/scheduler/tests:. python3 -m unittest test_burstgpt_replay test_session_cow_transaction test_offline_phone_residency test_replay_determinism
```

77 tests pass in 105.176 s. No complete-suite rerun was needed for this
measurement-only change. Both replay goldens are unchanged:

- v3: `sha256:f78d2b2c37a3880a523eba4f5315ada0207678c841d633229782bfa3a05c1829`
- v8: `sha256:965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

## Artifacts and remaining work

All raw physical files are under `/home/zhihao/` on the desktop:

- `s42-ffn-mixed-session-20260905-v1-gate/run/FAILURE.json`,
  SHA `feb6c563db493ca1f21a2f65f23b74d1c9d8d47abe15713d1a790b9241e2ea38`.
- The same run directory's `COLD_PRELOAD_RESULT.json`,
  SHA `0b52ff6a6892e908eb139c2e25d47be87f502ed91d6955132a208867da297bbe`.
- The same run directory's `ONLINE_QWEN_RESULT.json`,
  SHA `23ff95411a7082e022fa6ffb5fde7be296c07b67cee4b180eb3c6c96f48035f7`.
- `s42-ffn-mixed-session-20260905-v1-inputs/` holds configuration, commands,
  preflight, source manifest, phone logs, and all decoded diagnostic scripts.
- `s42-ffn-mixed-session-20260905-v1-deploy/` is the preserved tested deployment.

Canonical source manifest identity:
`sha256:fc1dc7c570d9f79a1860aecc6dce0d553f52d7d5f1fda6b80e231c86fffe863b`.
Its file SHA is
`a7c387756555a1ed88350f29bfff01287b257eed5c33b8612499f0340ddc011c`.
Preflight confirms the same qualified Qwen 16-GPU-layer and Gemma
22-GPU-layer parents; all six deployed shard hashes pass. The catalog identity
is unchanged from the earlier matched screen:
`sha256:d0403d09b786dd985767b57106079efdeb4f7fd7ade0596fef84db3f8ecab5ce`.

Local copies: [partial audit](PARTIAL_GATE_AUDIT.json),
[raw gap diagnostic](FORWARD_GAP_DIAGNOSTIC.json), and
[call-cadence diagnosis](CALL_CADENCE_DIAGNOSTIC.json).
These retain their original desktop bytes and hashes.

The next decision is the interruption metric. Reverse/rollback, Gemma calls,
and full warm matched A/B remain unvalidated on this build. No long trace
or new savings claim is authorized by this partial result.
