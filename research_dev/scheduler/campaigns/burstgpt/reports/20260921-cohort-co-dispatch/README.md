# Cohort co-dispatch queue fix

2026-09-21 03:42 UTC. PASS. Local controller checks; no physical run or energy
measurement. M2's accepted envelope is enforced at at most four members per
cohort. Phone kernels are unchanged.

The unchanged `test_cold_phone_cohort_shares_transition_identity` hung before
the fix (bounded at 15 s, exit 124) and now passes in 1.212 s. Queue snapshots
exclude each member itself when computing `active_conflict`.

| Version | Member | Queue state | causal_ready | active_conflict | Predecessors |
| --- | --- | --- | --- | --- | --- |
| Before | cold-cohort-a | ACTIVE | True | False | none |
| Before | cold-cohort-b | QUEUED | False | True | cold-cohort-a |
| After | cold-cohort-a | ACTIVE | True | False | none |
| After | cold-cohort-b | ACTIVE | True | False | none |

Before, the two requests held 16 distinct request-owned lease tokens. After,
both reference the same eight tokens owned by `decode-cohort-1`. Both tickets
are ACQUIRED; the queue calls that state ACTIVE. Raw states are in
[BEFORE_QUEUE.json](checks/BEFORE_QUEUE.json) and
[AFTER_QUEUE_FINAL.json](checks/AFTER_QUEUE_FINAL.json).

Admission now passes its existing typed cohort binding into the queue and
keeps it synchronized through formation, sealing, dissolution and checkpoint
restore. Matching cohort identity and shared tokens exempt only fellow members
from lane conflicts; `_bind_causal_predecessors` still orders genuine conflicts.
The fifth member receives a separate reservation and waits until the fourth
active member releases the shared lanes.

The current tree also excluded every cold plan from cohort formation. Identical
phone-only preparation can now share a cohort; its transition, participant and
adapter identities enter the cohort digest. Different preparation identities
and desktop preparation remain separate. A longer follower extends execution
leases without stretching preparation into execution. Survivor handoff uses
live leases after preparation has completed. Replan generation validation is
preserved; replacement admission updates the binding within that transaction.

| Check | Result |
| --- | --- |
| Original hanging test, assertions and method source unchanged | PASS, 1 test, 1.212 s |
| Two members dispatched by concurrent threads | PASS |
| Fifth waits through the first three completions, acquires after the fourth | PASS |
| Cold preparation completion and survivor lease handoff | PASS |
| Different transition, participant or adapter identity cannot join | PASS |
| Requested adaptive runtime, queue, controller and cohort modules | PASS, 133 tests, 18.603 s |
| Phase-scoped energy and runtime resource modules | PASS, 41 tests, 0.915 s |
| pyflakes on touched modules, tests and observer | PASS, no output |

All tests ran from the tests directory with the repository on PYTHONPATH.
Sibling test imports and physical-residency hooks were left unchanged. An
intermediate run found a replan-generation error; the final run passes without
weakening that check. This report does not claim a new physical performance
result. M3 still waits for the OP12 move; M4 is next.

## Commands and records

Before the edits, from the tests directory:

```sh
cd /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/tests
PYTHONPATH=/home/myid/zs89458/Documents/llama.cpp-release PYTHONDONTWRITEBYTECODE=1 timeout 15s python3 /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/REPRODUCE.py /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/checks/BEFORE_QUEUE.json > /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/checks/before.log 2>&1
```

Final checks, from the same directory:

```sh
PYTHONPATH=/home/myid/zs89458/Documents/llama.cpp-release PYTHONDONTWRITEBYTECODE=1 timeout 180s python3 -m unittest -v test_adaptive_runtime test_runtime_queue test_runtime_controller test_decode_cohort > /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/checks/requested-final.log 2>&1
PYTHONPATH=/home/myid/zs89458/Documents/llama.cpp-release PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 -m unittest -v test_phase_scoped_energy test_runtime_resources > /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/checks/phases-final.log 2>&1
PYTHONPATH=/home/myid/zs89458/Documents/llama.cpp-release PYTHONDONTWRITEBYTECODE=1 timeout 20s python3 /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/REPRODUCE.py /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/checks/AFTER_QUEUE_FINAL.json > /home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/checks/after-final.log 2>&1
```

Lint, from the repository root:

```sh
cd /home/myid/zs89458/Documents/llama.cpp-release
python3 -m pyflakes research_dev/scheduler/_internal/runtime_queue.py research_dev/scheduler/_internal/runtime_decode_cohort.py research_dev/scheduler/_internal/runtime_controller_ops/admission.py research_dev/scheduler/_internal/runtime_controller_ops/cohorts.py research_dev/scheduler/_unified/automated_requests_ops/commit.py research_dev/scheduler/tests/test_runtime_queue.py research_dev/scheduler/tests/test_adaptive_runtime.py research_dev/scheduler/tests/test_runtime_controller.py research_dev/scheduler/tests/test_decode_cohort.py research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/REPRODUCE.py > research_dev/scheduler/campaigns/burstgpt/reports/20260921-cohort-co-dispatch/checks/pyflakes-final.log 2>&1
```

[VALIDATION.json](checks/VALIDATION.json) records the before/after summary and
unchanged regression-method hash. [CHANGES.patch](CHANGES.patch) contains only
this task's code and test edits relative to the shared tree on entry.
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records before/after file hashes;
`source-before/` and `source-after/` preserve the corresponding files.
