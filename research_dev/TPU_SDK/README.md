# Pixel Tensor SDK installation and FFN investigation

Installed locally on 2026-09-24 UTC from the user-provided archive
`Latest_SDK_v2.0_2026_09_02-20260924T004730Z-1-001.zip`.

- SDK root: `installed/v2.0-20260902/google_tensor_ml_sdk`.
- Isolated environment: `installed/v2.0-20260902/.venv`, Python 3.12,
  `ai-edge-litert==2.2.0`. No global Python or system package changes.
- Entry point: `installed/v2.0-20260902/run-python` sets
  `GOOGLE_TENSOR_COMPILER_LIB` to the SDK directory and invokes the environment.
- Archive, plugin and dependency provenance: `INSTALL_MANIFEST.json` and
  `PIP_INSTALL.json` in the installation directory. Original archive retained.
- The archive contains a compiler library and toolchain, not a llama.cpp
  backend. The LiteRT wheel supplies the compiler adapter and model tooling.

## Validated use

From the repository root, with a new output directory for each invocation:

```sh
research_dev/TPU_SDK/installed/v2.0-20260902/run-python \
  research_dev/TPU_SDK/compile_model.py input.tflite output-new-directory
```

The script targets Tensor G5, explicitly sets FP16 truncation (`half`), and
retains compiler partition coverage and output hashes. `--precision` and
`--sharding` expose the documented compiler settings for isolated experiments.
Compilation success alone is insufficient evidence of TPU execution. Check
full operator coverage, then run on the physical phone with NPU-only execution
and verify dispatch and numerical results.

The installation smoke test compiled ADD (1/1 operations) and executed 24
exact calls on Pixel 10 Pro. The real Qwen layer18 tail512 FFN compiled all
6/6 operations into one partition and passed 88 physical calls. See the
[M3 investigation report](../scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/PIXEL_TPU_SDK_RESULTS.md)
for complete measurements and limits.

## Real FFN model

`extract_ffn_weights.py` reads an existing F16 GGUF and exports gate/up/down
matrices from a single layer and column interval. It records the actual source
shard hash, which differs from the parent model's protocol artifact identity.
`export_ffn_model.py` expands those F16 values exactly to FP32 TFLite constants:

```text
g = x @ gate.T
u = x @ up.T
y = (g * sigmoid(g) * u) @ down.T
```

Input and output dimensions are `[batch, 5120]`; intermediate width is the
selected FFN column count (up to 17408). No bias or residual is added.
Eight deterministic inputs use an archived layer18 qualification vector,
scaled/sign variants, and new synthetic inputs. These are numerical test
vectors, not a captured production activation distribution. NumPy FP32 and LiteRT CPU results must agree
before compilation. The TPU uses the compiler's explicitly selected precision;
passing relative error is not proof of identical full-model output tokens.

The bounded physical probe adapts the existing LiteRT TPU-add probe. It holds
the Pixel lock, uses ADB port5037, accepts a finite number of requests, sends
each response in one write, and exits normally. It requires every graph node
to be a compiled dispatch node and requests only NPU execution. Runtime
invocation measurements include vendor/runtime overhead; they are not pure
hardware-kernel timestamps. No production server or scheduler integration is
performed by these scripts.

## References

- [Google Tensor SDK overview](https://developers.google.com/edge/litert/next/tensor-sdk).
- [Compilation flags](https://developers.google.com/edge/tensor-sdk/compilation-flags).
- [LiteRT NPU execution](https://developers.google.com/edge/litert/next/npu).
- [Official AOT example](https://github.com/google-ai-edge/litert-samples/blob/main/samples/litert/colab/LiteRT_AOT_Compilation_Tutorial.ipynb).

The installed LiteRT 2.2.0 backend source was also checked directly: its
Google Tensor compiler step accepts `GOOGLE_TENSOR_COMPILER_LIB` as the
directory containing `liblitert_plugin_compiler.so`.
