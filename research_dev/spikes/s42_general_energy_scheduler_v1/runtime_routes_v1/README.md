# S42 epoch-bound route compiler V1

## Result

The compiler emits two direct, physically measured cohort routes for one
exact RTX 4060 Ti plus OP15 epoch:

| Route | Scope | Quality | Mean duration | Mean fleet energy |
| --- | --- | --- | ---: | ---: |
| CPU control | task | exact local execution | 736.468 s | 137.217 kJ |
| CPU plus OP15 dense-FFN split | operator inside the fixed cohort | approximate | 630.594 s | 114.217 kJ |

The treatment reduced mean makespan by 14.38% and accounted fleet energy by
16.76%. Each of the three alternating treatment repeats beat its paired
control. Its conservative direct-cohort bounds are the largest observations:
632.916 s and 114.516 kJ.

Both routes are compiled with epoch key:

```text
sha256:27d004d2958af9df39e4543eda7853cca68f9293df7aba3515bed3585775a1e5
```

The key covers hardware IDs, both model hashes, both runtime manifests,
residency, transport, phone worker and bridge hashes, the I3 split policy,
continuous-batch settings, and the exact 74-request trace geometry. Any
mutation fails compilation.

## Fail-closed boundary

The graph-to-profile audit found 288 local full-operator Gemma kernel matches
and 240 isolated phone split-kernel matches. The latter do not become new
routes. They still lack a measured CPU complement, a qualified composed
transfer and merge path, and held-out per-shape full-model validation. The
Qwen graph has no exact per-operator row in this profile.

The direct I3 treatment remains valid as an exact-workload cohort certificate
because it was measured end to end in three alternating pairs. Its numerical
quality is `approximate`: the MMLU64 scores were both 27/64, but exact-token
equality was not established. It cannot serve exact-quality requests.

Compiled routes intentionally have `activation.ready=false`. Stage 5 must
probe online thermal, failure, cancellation, KV, sampler, and contention
gates before dispatch. The CPU route is the treatment's atomic fallback.

The historical S41 preflight recorded models, server manifests, devices,
policy, and workload, but did not hash the bridge and phone worker. This epoch
adds their current prospective hashes. That makes future dispatch fail closed
on those artifacts, but does not retroactively prove their historical binary
identity. The cohort certificate is consequently an observed behavior
certificate, not a broad binary reproducibility claim.

## Files

- `route_compiler.py`: strict graph, profile, certificate, and epoch compiler.
- `CERTIFICATES_4060TI_OP15_V1.json`: three-pair direct physical evidence and
  dispatch contracts.
- `I3_CERTIFIED_EPOCH_4060TI_OP15_V1.json`: exact static epoch bindings.
- `COMPILED_4060TI_OP15_I3_ROUTES_V1.json`: compiled receipt.

The adapted Gemma and Qwen graphs are raw physical captures and remain ignored
under `raw/runtime_routes_v1/`. Their SHA-256 values are bound by the
certificate.

## Reproduce

```sh
python3 \
  research_dev/spikes/s42_general_energy_scheduler_v1/runtime_routes_v1/route_compiler.py \
  --profile research_dev/scheduler/profiles/MEASURED_4060TI_OP15_KERNEL_PROFILE_V1.json \
  --certificates research_dev/spikes/s42_general_energy_scheduler_v1/runtime_routes_v1/CERTIFICATES_4060TI_OP15_V1.json \
  --epoch research_dev/spikes/s42_general_energy_scheduler_v1/runtime_routes_v1/I3_CERTIFIED_EPOCH_4060TI_OP15_V1.json \
  --graph gemma=research_dev/spikes/s42_general_energy_scheduler_v1/raw/runtime_routes_v1/gemma-placement-graph-mixed-v5.json \
  --graph qwen=research_dev/spikes/s42_general_energy_scheduler_v1/raw/runtime_routes_v1/qwen-placement-graph-mixed-v2.json \
  --output compiled-routes.json
```

The compiler refuses to overwrite an output.
