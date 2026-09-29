# OP15 same-process Gemma weight adoption

Admission: `SAME_PROCESS_WEIGHT_ADOPTION_QUALIFIED_ENERGY_ABBA_PENDING`.

The RTX 4060 Ti held Qwen with 15 CUDA layers while two OP15 HTP sessions
executed Qwen FFN layers 0 through 11. During 54 complete OP15 FFN groups, a
Gemma-owned CUDA allocation received and read back the complete
2,013,265,920-byte tied `token_embd.weight`. Gemma then published the model and
served BurstGPT request 50 with the exact established 41-token SHA-256.

| Gate | Result |
|---|---:|
| Copied and verified | 2,013,265,920 bytes |
| Protected windows | 54 |
| Copy p90 | 51.539 ms |
| Protected overrun p90 / max | 0 / 3.983 ms |
| Minimum free GPU memory | 995,688,448 bytes |
| Margin beyond 512 MiB reserve | 458,817,536 bytes |
| Cgroup memory peak | 31,435,599,872 bytes |
| Qwen / Gemma peak process swap | 0 / 0 bytes |
| Cgroup swap limit / OOM events | 0 bytes / 0 |
| Gemma request 50 | exact 41-token hash |

The paid Qwen interval consumed 3,194.296 J across CPU package, GPU board, and
whole phone, but this is not an incremental energy result. Its control must
retain the same Gemma CPU fallback and include source preparation, transition,
restore, and equal completed trace work inside the same boundary. The checked
qualification therefore leaves dynamic energy savings as `null`.

`ADOPTION_QUALIFICATION_V1.json` is rebuilt from the raw result, bridge,
phone-energy, process-memory, cgroup, Gemma loader, and Gemma execution
receipts. `raw/` preserves those hash-bound artifacts.
