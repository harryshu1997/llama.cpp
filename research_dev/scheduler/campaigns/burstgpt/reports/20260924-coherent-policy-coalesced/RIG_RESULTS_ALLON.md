# All-on scheduler development trace, reviewed 2026-09-24

Completion PASS; strict token identity FAIL. These are single runs of the same
9-request `longtail_dev_v2` trace (1,420 output tokens), with the new dispatcher
enabled in both arms. They are not a full `longtail_v1` evaluation.

| Arm | Host CPU kJ | Host GPU kJ | Host total kJ | Duration s | Assumed phone kJ | Exact outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Desktop + dispatcher | 56.056 | 26.266 | 82.323 | 865.004 | 0.757 | reference |
| OP15 + coherence + reprovisioning + dispatcher | 20.784 | 24.901 | 45.686 | 811.949 | 2.050 | 7/9 |

Measured host saving: **44.504%**; elapsed time falls **6.133%**. Phone energy
is an assumed model, not a measurement. Both arms completed 9/9 requests with
zero rejections. Prompts, model identities, seeds, token budgets and trace
arrival/SLO fields match. Qwen request `004` first differs at output index 57;
Gemma request `005` first differs at index 146 (zero based). Their cause was
not diagnosed in this review. This result fails the all-outputs-exact gate.

Batching PASS in the native logs: Qwen has 1,854 three-row and 108 two-row USB
calls; Gemma has 96 two-row calls. All four Qwen requests and three of four
Gemma requests used the phone. Qwen phone windows at batch three report
36.26 host J/token (the analyzer's aggregate, not a median). Phone-policy
window coverage is 86.9% of Qwen output and 85.7% of Gemma output. The
`active_slots_peak=1` summary disagrees with both native row counts and saved
request intervals; it must not be interpreted as absence of batching.

Large-model loads still consume 210.5 s versus 203.5 s in the desktop arm,
with six launches and four model switches in each. Gemma's final short
request remains unassisted. The pending reprovision follow-ups and USB
selector work are unfinished scratch changes; neither was merged here.

Evidence:

- `ALLON_MATCHED_COMPARISON.json`: existing `analyze_longdecode_pair.py`,
  applied to both complete RESULT files and saved token streams.
- `PIXEL_REVIEW_ALLON.json` and `.md`: existing `analyze_allon.py`, including
  per-request, native batch, window, probe, load and dispatcher details.
- Desktop: `/home/zhihao/s42-trace-longtaildev2-baseDP-20260924-inputs/run-dev2baseDP-1/run`.
- OP15: `/home/zhihao/s42-trace-longtaildev2-allon-20260924-inputs/run-dev2allon-1/run`.

No Pixel execution or incremental two-phone energy saving is included.
