# Model energy calibration V1

This directory measures resident model-inference energy for Qwen3-14B and
Gemma4-12B. It reuses the validated S41 RAPL, NVML, request, and phone-clock
primitives but runs one model route at a time so mixed workloads do not
confound model cost.

The primary estimator is now the S42 `operator_sum_v1` model. It composes
measured kernel throughput, bandwidth, launch time, and power with operator
compute and memory work. The existing bounded cohort fit remains a full-route
validator and residual diagnostic; it is not the generalization mechanism.

The server boundary is measured directly:

```text
E_server = E_CPU_package_RAPL + integral(P_GPU_board_NVML dt)
```

This excludes motherboard, AC conversion, fans, storage, and DRAM outside the
CPU package. For the first scheduler prototype, a phone-offload case may add a
separate conservative estimate:

```text
E_accounted = E_server + 5 W * phone_reserved_seconds
```

The 5 W term is labeled `estimated`; it is not allowed to become a measured
fleet-energy claim. Server-only cases have zero phone-reserved time.

Files:

- `PLAN.md`: scope, gates, and execution order.
- `../OPERATOR_ENERGY_MODEL_V1.md`: active operator-energy specification.
- `CASES_V1.jsonl`: frozen calibration and held-out cohort cases.
- `MODELS_V1.json`: exact initial model identities and architecture features.
- `run_model_energy.py`: resident single-model physical runner.
- `run_server_energy_arm.sh`: server-only acquisition without touching a phone.
- `run_model_energy_arm.sh`: synchronized desktop and OP15 acquisition wrapper.
- `attach_phone_energy.py`: clock-aligned per-case whole-phone reducer.
- `fit_model_energy.py`: fail-closed model-specific profile fitter.
- `test_run_model_energy.py`: runner validation tests.
- `test_attach_phone_energy.py`: phone-boundary tests.
- `test_fit_model_energy.py`: fitting and held-out-gate tests.

Fit the OP15 route with the temporary 5 W assumption by passing
`--assumed-phone-power-w 5` to `fit_model_energy.py`. Omit that option for the
two server-only routes.
