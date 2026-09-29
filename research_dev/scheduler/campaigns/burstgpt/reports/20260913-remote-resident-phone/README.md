# Remote-resident phone preflight and proof repair

## Outcome

OP15 is connected again, but the real-phone relocation gate has not run.
The canonical preflight fails closed with
`phone transport qualification software identity differs`.
No inference, weight load, USB changeover, reset or worker shutdown was performed.
The existing TCP stand-in results are preserved and are not phone results.

The saved transport identity pins the September 11 server and libraries. The
current remote-resident build differs in the server and all seven host library
hashes. The deployed phone worker and session script still match. The complete
expected/observed host hashes are in
`physical/preflight-v1/PHONE_OWNER_PREFLIGHT.json`.

Live NVML reports 16,409,165,824 bytes free. The existing qualified desktop
placement is feasible: 23 GPU layers, context 2560, batch 2048, ubatch 512,
parallel 1, CUDA graph mode `default`. Its placement hash is
`sha256:a87d0996e54e49c0280e35e138a375464404efa62c3d236cedcba4f4ed8c3920`.
No placement or memory threshold was changed.

Remote artifacts:
`/mnt/storage/s42-remote-resident-phone-20260913-v1-Zo12Rb/`.
This is a new directory, with an isolated Python deployment and the existing
native binaries. It does not overwrite the stand-in deployments or results.

## Reproduced defects and scoped changes

1. A remote-resident parent has execution mode `desktop`, so `begin_execution`
   left its phone contract empty and `finish_execution` treated its real phone
   calls as unauthorized desktop calls. The new regression failed on that
   empty contract before the fix. Remote-resident parents now require the
   exact remote layer mask, complete FFN columns, payload shape and call coverage.
2. The owning session's operator-plan identity was lost when the READY layout
   bound the remote-resident declaration. Binding now carries that physical
   operator plan independently of the desktop execution plan. Request proof
   generation uses it, and the execution marker pins the entire owner identity
   so a generation, shard or operator change cannot be substituted at completion.
   Missing operator bindings fail closed. These runtime bindings do not change
   generation-free placement hashes.
3. The route validator compared the session's maximum supported width for
   equality with Gemma's FFN width. The real catalog supports 17,408 columns;
   Gemma requires exactly 15,360. The capacity check now accepts a larger
   maximum, while the READY shard must still have exactly the required width.
   The exact physical-catalog width regression failed before this correction.
   The READY shard's endpoint is also checked against the declared owner.
4. The gate advertised a phone owner, but its transport and owner-stop branches
   were placeholders. Added a canonical `--preflight-only` path that persists
   qualification differences and labels success `PREFLIGHT_PASS`, never a gate
   pass. Early gate errors now produce `FAILURE.json`. Phone preparation and
   execution wiring remain unfinished; the documentation no longer claims they
   work merely because the phone has reconnected.

## Files changed

Production and gate code, relative to `research_dev/scheduler/`:

- `_internal/plan_contracts/remote_resident.py`
- `_internal/route_generation/remote_resident.py`
- `adapters/ticket.py`
- `adapters/llama_server.py`
- `adapters/llama_server_contracts.py`
- `adapters/llama_server_ops/proofs.py`
- `campaigns/burstgpt/remote_resident_gate.py`

Tests:

- `tests/test_remote_resident_contract.py`
- `tests/test_remote_resident_routes.py`
- `tests/test_remote_resident_launch.py`
- `tests/test_remote_resident_gate.py` (new)

This report and `research_dev/talks.md` record the work. No native code, shard
files, adaptive controller, existing lifecycle, transport wire format,
qualification records, frozen results or baselines were changed. Nothing
committed or pushed.

## Validation

Focused command (76 tests):

```sh
PYTHONPATH=.:research_dev/scheduler/tests python -m unittest \
  test_remote_resident_gate test_remote_resident_launch \
  test_remote_resident_contract test_remote_resident_routes \
  test_remote_resident_accounting test_llama_server_adapter -q
```

The two saved-run replay goldens are preserved; their values are:

- session COW v3: `sha256:5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4`
- sparse-locality v8: `sha256:241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917`

`FOCUSED_TESTS.log` and `REPLAY_TESTS.log` contain the final command outputs.
Compile, pyflakes and whitespace checks pass for the edited files. The broad
scheduler suite was not rerun in this bounded step.

## Next gate prerequisites

1. Establish a valid measured transport qualification binding for the rebuilt
   runtime; do not copy the old identity or edit only its hashes to force a pass.
2. Complete the existing gate's phone preparation/close callbacks through the
   canonical session controller, with memory admission for the full ubatch
   workspace, READY publication and ticket-bound terminal proofs.
3. Run only the short real-phone correctness and memory gates, then owner-loss
   recovery. Keep the old stand-in and every failed attempt immutable.
4. Test additional useful KV capacity separately. Omitting the selected
   CPU-resident FFNs frees host RAM, not GPU VRAM; no capacity or energy gain
   has been established by this preflight.

No 24-request or longer trace is authorized by this milestone.
