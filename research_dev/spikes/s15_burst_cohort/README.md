# S15 observed BurstGPT B32 cohort

This checkpoint freezes the first physical scheduler workload after the B32
device gate. It does not simulate arrivals. It selects 32 eligible generation
requests from the densest real BurstGPT v2 burst window while preserving their
normalized arrival timestamps.

BurstGPT does not contain request text, priorities, or deadlines. Therefore:

- arrival timestamps and token counts are observed;
- the common prompt `Explain batching.` is synthetic and matches the physical
  B32 correctness gate;
- low priority and the relative 5 s SLO are synthetic sidecar fields;
- the artifact tests batching/admission mechanics, not original-request replay.

The OP15 seven-process B32 maximum is 2,933,672 us. The selected cohort forms
in 1,000,000 us, so a launch at the last arrival is predicted to finish
1,066,328 us before the earliest synthetic deadline. Physical execution and
server energy are separate checkpoints.

Preloading weights and contexts is allowed; pre-reading the fixed prompt before
its request arrivals is not. A launcher that starts a prompt-bound process
before replaying the arrivals is a physical smoke test, not an arrival-faithful
SLO pass.

Run:

```sh
python3 build_cohort.py
python3 validate_cohort.py
python3 -m unittest -v test_cohort.py
```
