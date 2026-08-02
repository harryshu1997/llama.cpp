# Same-Quantization Identity Check

## Result

`IDENTITY_ADMISSION_PASS_NUMERICAL_QUALITY_STILL_FAILS`

The current-source StageNet binary was built for CUDA and Android. The binary
SHA-256 deployed to both phones was
`7bbac3ec1fdcc021e4f47d3bdb97625a796bcf3d73b0532101eed849d5ea48a9`.

Four live configurations returned the following model identity through the
new protocol capability:

| Worker | Layers | GGUF file type | Model SHA-256 |
| --- | --- | ---: | --- |
| CUDA test worker | `[0,1)` | 7 (`Q8_0`) | `7b56cbd0...3d492848` |
| OP12 HTP0 | `[0,1)` | 7 (`Q8_0`) | `7b56cbd0...3d492848` |
| OP15 HTP0 | `[1,2)` | 7 (`Q8_0`) | `7b56cbd0...3d492848` |
| RTX 4060 Ti controller | `[0,8)`, `[8,16)`, `[16,48)` | 7 (`Q8_0`) | `7b56cbd0...3d492848` |

The full digest is
`7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848`.
Both phones loaded the full file from storage and prepared only the declared
layer window. The RTX 4060 Ti loaded an independently transferred copy with
the same full-file digest. All queried workers exited cleanly.

The desktop's older Q8_0 file has digest `f20e7ff1...d12afa72`. A live
negative query proved that it is rejected even though its GGUF file type is
also 7. The desktop controller now defaults to the exact shared artifact.

## Verification

- StageNet client tests: 50 passed.
- S31 route and launch-evidence tests: 16 passed.
- S33 quality and evidence tests: 26 passed.
- CUDA and Android `llama-layersplit` builds: passed.
- RTX 4060 Ti three-slice controller session and artifact hashes: passed.
- Shell syntax and `git diff --check`: passed.

The digest is supplied to the worker only after the launch controls recompute
the file hash. The worker derives `general.file_type` from the loaded GGUF.
The route rejects missing identity, unequal file type, unequal digest, and a
digest that differs from the requested model.

This check prevents mixed F16/Q8 or different-file routes. It cannot repair
the measured HTP-versus-CUDA arithmetic divergence for the same Q8_0 file;
S33's quality failure remains authoritative.
