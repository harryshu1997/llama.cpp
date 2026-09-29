# Archived gate launch scripts

One-off desktop launch scripts from the 2026-08-30 to 2026-09-01 physical gates
(dimensional, phone-coverage v3-v12, helper-rebind v11, session-granular v3).
They hard-code the desktop deploy directories of those runs and are kept only
so the talks.md log entries that cite them stay reproducible. New gates use
`../launch_gate.sh`.

Also archived: the one-off replay and deploy-preparation helpers
(`causal_replay_v9.py`, `dimensional_five_replay.py`, `phone_subset_replay.py`,
`prepare_helper_rebind_v12.sh`, `replay_helper_rebind_v13.py`) that used to
live at the repository root as `.tmp_*` files.
