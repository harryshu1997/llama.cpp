# Isolated CPU-overflow A-B-B-A evidence

This directory preserves the 2026-08-12 physical qualification on the RTX
4060 Ti desktop and OP15. The large-model policy is `cpu-overflow` in every
arm. Only the small-model policy changes between `static-cpu` and
`runtime-scheduler`.

`CPU_OVERFLOW_ABBA_V1.json` is the aggregate report. Its two matched fleet
energy savings are 5.746% and 4.086%. The mean is 4.916%, but the paired 95%
confidence interval is -5.634% to 15.466%, so `energy_claim_eligible` is
false. The runtime arms physically executed 8/10 and 7/10 Llama requests on
OP15 and met 10/10 small-model SLOs; both static arms used CPU for 10/10 and
met 4/10.

The `calibration` directory contains the phase-conditioned profile and its
held-out audit. The audit covers eight measured phase/policy classes with
seven training and three held-out requests per class and reports zero of 24
held-out upper-bound violations.

Each arm directory contains its combined result, whole-phone energy receipt,
and physical endpoint qualification. `SHA256SUMS.txt` binds the copied
artifacts. Generated token-content hashes are execution outcomes; identical
work is established by the trace identities and exact requested and actual
per-request output shapes recorded by the comparator.
