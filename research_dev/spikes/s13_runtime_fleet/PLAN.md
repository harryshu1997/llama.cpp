# S13 Live Phone Fleet Runtime

## Goal

Replace the two-phone simulator boundary with the smallest compiled runtime
that sends real llama.cpp operator-island work to OP12 and OP15.

## Scope

This checkpoint is limited to the existing Gemma4 dense FFN island in
`examples/phone-pim`. It does not edit `llama_decode`, model graph construction,
KV internals, or the backend scheduler.

## Checkpoints

- [x] Reproduce the existing FFN island independently on both phones with the
      production llama.cpp graph oracle.
- [x] Add a reusable protocol client with complete response correlation and
      typed remote errors.
- [x] Require `HELLO -> STATUS` on every new session and adopt the STATUS
      residency generation before any mutating command.
- [x] Keep one persistent client session per phone.
- [x] PREPARE the exact model digest and island identity on every phone.
- [x] Build deterministic finite inputs with a local CPU FFN oracle and check
      every returned result against it.
- [x] Give every phone one initial job, then drain a shared queue by
      completion-driven work stealing.
- [x] Report cold setup separately from active dispatch makespan.
- [x] Run release, sanitizer, warning-clean, protocol, store, stream, and
      adversarial client tests.

## Stop Boundary

This checkpoint does not claim mixed-model scheduling, server fallback,
latency improvement, capacity relief, or energy savings. It does not implement
the S12 WiFi-input/USB-output topology. The live test uses ADB TCP forwards, so
both activation directions currently traverse the selected ADB transport.

The next runtime checkpoint must add a bounded executable job descriptor and
server fallback before it consumes a real trace. A later checkpoint may add
multiple model identities and dynamic provisioning. Direct `llama-server`
integration remains blocked until there is an exact graph continuation seam or
a complete supported operator-island route.
