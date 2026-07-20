# S15 resident host-tail input seam

`llama-layersplit --mode pipedriver --prompt-after-load` loads the model and
context, emits `DRIVER_INPUT_READY`, and then reads exactly one bounded prompt
line from stdin. This gives the scheduler a real launch boundary after
residency preparation without placing future request payload on the process
command line.

The option is intentionally narrow:

- it is valid only for `pipedriver`;
- it cannot be combined with `-p`;
- it accepts one nonempty line of at most 16 KiB;
- all old invocations retain their existing behavior;
- it changes no model graph, backend scheduler, or stage wire protocol.

The seam still supports only the current same-input batch harness. Arbitrary
per-request prompts need a later versioned input protocol.
