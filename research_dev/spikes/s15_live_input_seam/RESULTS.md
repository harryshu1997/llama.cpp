# S15 resident host-tail input seam results

Verdict: `HOST_INPUT_SEAM_MECHANICS_PASS_REAL_B32_NOT_RUN`

## Change

An opt-in `pipedriver --prompt-after-load` path now:

1. loads the model and context;
2. emits `DRIVER_INPUT_READY`;
3. accepts one bounded prompt line from stdin;
4. emits `DRIVER_INPUT_ACCEPTED` without printing the prompt;
5. continues through the existing pipebatch route.

Command-line `-p` is forbidden in this mode. Existing invocations are
unchanged.

## Verification

- CUDA build: PASS;
- CPU build: PASS;
- Android Snapdragon build: PASS;
- ASan/UBSan build: PASS;
- release tests: 4/4 PASS;
- ASan/UBSan tests: 4/4 PASS;
- existing fixed-route regression suite: 105/105 PASS;
- tiny-model ordering test proves the process is alive at
  `DRIVER_INPUT_READY` before the harness writes prompt bytes;
- empty input, a command-line prompt, and a non-pipedriver use all fail closed.

Source SHA-256:
`fd03e735eaa2b52b13bba6c6fb5e8c16726b7c54bcac74eaae5d9239d0a1bfde`

Host CUDA binary SHA-256:
`8910c81991999884572d9390077c0f18b76adf46b39eb021b23a736e1948f8f2`

Android binary SHA-256:
`723866bf51db0435177188f8e39623ff28cef5c9652df7612b57a0bda6242809`

## Limit

The old OP15 B32 profile binds a different frozen host binary. This change
does not inherit that certification. The next real-device run must recertify
the B32 route with this host binary before making an arrival-faithful SLO
claim. The current seam accepts one shared prompt, not 32 arbitrary prompts.
