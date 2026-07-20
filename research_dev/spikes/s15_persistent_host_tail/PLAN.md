# S15 persistent host-tail gate

Status: `PERSISTENT_OP15_B32_HOST_AND_PHONE_PASS_ENERGY_UNKNOWN`.

## Goal

Keep both halves of the certified OP15 `[0,8)` plus A6000 `[8,48)` B32 route
resident across multiple request sessions. The first exchange must DETACH and
reset request-local phone state; the second must STOP both processes cleanly.

## Gate

- One A6000 tail PID and one OP15 worker PID/nonce across both exchanges.
- One current model/context load per device, outside the exchange windows.
- Exactly 32 requests and eight generated tokens per exchange.
- Exact same-batch CUDA token IDs for all 64 requests.
- OP15 HTP0 placement with CPU only for GET_ROWS.
- A6000 tail placement entirely on the selected CUDA0 backend.
- Session IDs 1 and 2, contiguous step totals, DETACH reset, then STOP.
- One bounded command and one canonical result per launch ID.
- Each exchange completes within 4,000,000 us.
- No selected-GPU, phone, USB, server-wall, or total-system energy claim.

## Boundary

This gate drives the C++ JSONL contract directly. It does not yet pass through
the typed `PersistentPreparedTransport` and `StageNetSessionAdapter`, and it
uses one repeated synthetic prompt rather than a mixed arrival trace. Those are
separate downstream gates.
