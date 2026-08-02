# S39 joint B8 prototype

This directory is an exploratory, non-qualification path for one live
Qwen3-14B B8 execution on the phone collective while the same model is
resident and executing on CUDA.

It deliberately does not produce V2.4 or V2.6 receipts. It skips reboot,
thermal readiness, task quality, memory qualification, and authority. Its
only purpose is to establish live execution, placement, cleanup, and
diagnostic token agreement before returning to the frozen acquisition.

The first real run completed on 2026-07-27. See `RESULTS.md`.

The prototype corrects only its copied launch plans:

- desktop connects to the OP15 relay address;
- the relay head is OP15 localhost;
- the relay tail is the OP12 Wi-Fi address.
- Android component stat records use live nanosecond values;
- the B8 envelope uses 8 streams, `n_batch=64`, and `n_ubatch=64`;
- the deployed three-option relay interface is used;
- the deployed CUDA runtime uses `-m`, `--devices`, and `-ngl`;
- stale CUDA layer-bound command-line options are removed.

The frozen V2.4/V2.6 plans and contracts remain unchanged.
