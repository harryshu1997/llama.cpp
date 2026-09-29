# BurstGPT campaign tools

The campaign runs requests through `UnifiedScheduler` and its physical adapters.
It does not own route selection, adaptive fractions, leases, or session state.

| Module | Responsibility |
| --- | --- |
| `launch.py` | Resolve campaign configuration and construct launch commands |
| `arguments.py` | Runner CLI options, dependency syntax, execution confirmation and argument validation |
| `trace_inputs.py` | Validate source traces, merge models, select requests and build deterministic replay schedules |
| `common.py` | Shared campaign error, canonical JSON, hashing and artifact reads |
| `runner.py` | Construct the scheduler and rig, submit arrivals, gather results and clean up |
| `preflight.py` | Check physical readiness without launching inference |
| `offline_residency_gate.py` | Bounded physical session-residency checks |
| `compare_ab.py` | Validate comparison identities and compare measured results |
| `analyze_phone_wait_timeline.py` | Analyze physical execution and helper coverage |

`runner.py` retains explicit imports of the extracted helpers for existing
callers. Add new input and CLI helpers in their owning modules instead of
duplicating them in the runner. Preserve the current CLI defaults and JSON
schemas when making organization-only changes.

Run focused software checks from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. \
  python3 -m unittest test_campaign_inputs test_burstgpt_replay \
  test_development_trace test_cuda_reference test_architecture
```

Local cleanup is not a deployment. An active physical comparison must retain
its frozen source, inputs and binaries through the remaining arms. Deploy a
reviewed refactor to a new versioned directory after the comparison completes.
