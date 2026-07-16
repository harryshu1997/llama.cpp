# S10-V0-R-E2 matched-timeline contract (frozen)

Status: FROZEN MECHANICS; PHYSICAL AGGREGATION BLOCKED. Schema version 1
(E2 namespace). ASCII, integer-only.

E2 is a POST-HOC comparison of two realized execution timelines. It is not a
solver, not a model, and not an input to one.

Units: time `us`, power `mW`, energy `nJ` (= mW * us), bytes. No floats anywhere,
including in raw artifacts after normalization.

## 0. Why E2 exists, and the architecture rule

E1 binds per-device, per-route terms and ADDS them. A SERVER_WALL or GPU_BOARD
measurement is an AGGREGATE timeline of a whole boundary. Feeding an aggregate
back into an additive solver double-counts and attributes shared power to
individual routes. E1 therefore rejects every `MEASURED` instance, and that
rejection is correct.

**Architecture rule, frozen:** an E2 timeline or comparison MUST NOT be consumed
by the E1 additive solver, and E2 MUST NOT import E1's binder, boundary, or
validator decision logic. E2 reuses E1 only for canonical JSON and strict loading,
read-only. `tests/test_e2.py` asserts this by construction.

E2 answers exactly one question: given two realized timelines that did the SAME
closed work with the SAME SLO outcomes at the SAME measurement boundary, is the
treatment's gross energy conservatively lower?

## 1. Allowed labels (closed set)

    GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED
    SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED
    RELIEF_FAIL
    MEASUREMENT_INVALID

`SYSTEM_ENERGY_SAVING` **is not a value in any E2 schema or enum**. It cannot be
expressed, so it cannot be emitted. Phone, USB supply, charger, and external-device
energy remain `UNKNOWN` and are never zero.

A positive SERVER_WALL result is a **break-even budget**:

    delta_E_server = E_server_control - E_server_qpim
    phone_plus_external_break_even_budget = delta_E_server

It states only that the excluded phone/USB/charger/relay energy may consume at most
that budget before the system stops breaking even. Those terms are unmeasured, so
the sign of the total is UNKNOWN. A positive delta is NOT a total-system saving.

A GPU_BOARD result has only `boundary_delta_nj`. It MUST set
`server_wall_delta_nj` and `phone_plus_external_break_even_budget_nj` to null. A
board sensor cannot establish how CPU, DRAM, storage, fans, or PSU losses changed,
so its delta is not a server delta and is not a phone-energy budget.

## 2. Scope -> label, typed not named

Capability is typed. A closed `instrument_kind` enum maps to exactly one allowed
scope. A free-form `instrument` string is a label and grants nothing.

| `instrument_kind` | allowed scope | strongest label |
|---|---|---|
| `NVML_BOARD` | `GPU_BOARD` | `GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED` |
| `RAPL_PACKAGE` | (none) | nothing - component counter, never a wall |
| `EXTERNAL_WALL_METER` | `SERVER_WALL` | `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` |
| `BMC_INPUT_POWER` | `SERVER_WALL` | `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` |
| `SYNTHETIC` | `GPU_BOARD` or `SERVER_WALL` for MECHANICS ONLY | **no physical label, ever** |

`SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` will additionally require evidence of
complete wall coverage. `ServerWallCapability` v1 resolves a hashed proof and
cross-binds its instrument identity, provenance, six coverage declarations, and
uncertainty floor. That is sufficient to test the binding mechanics, but it is
only a hashed declaration. It does not bind topology, calibration validity, or raw
acquisition evidence. The validator therefore rejects every `MEASURED` v1
capability with `E_CAPABILITY_UNCERTIFIED`. A later capability version is required
before a physical server-wall label can be emitted. See INSTRUMENT_AUDIT.md: this
host also has no wall instrument.

Per the audit, this host has TWO A6000 boards. A `GPU_BOARD` timeline names its
`board_uuids` set, and a comparison requires the SAME set on both sides.

## 3. Frozen integration rule

Normalized samples are `(timestamp_us: int, power_mw: int)`, strictly increasing in
`timestamp_us`.

**Zero-order hold, left-edge, integer-exact:**

    lo_i = max(timestamp_us[i], window_start_us)
    hi_i = min(timestamp_us[i+1], window_end_us)
    energy_nj = sum over i in [0, n-2] of
                power_mw[i] * max(0, hi_i - lo_i)

The first and last intersecting intervals are clipped exactly to the declared
window. Samples outside the window only bracket it and contribute no energy. The
last sample contributes no interval; it only closes the previous one. Energy is
integrated over the effective window `[window_start_us, window_end_us]`. E2 v1
requires that window to equal the execution-marker interval exactly. It must also
satisfy `samples[0].timestamp_us <= window_start_us` and
`window_end_us <= samples[-1].timestamp_us` so the window is BRACKETED by real
samples on both sides. Extrapolating past the first or last sample is
`E_UNBRACKETED`.

Rationale for left-edge ZOH: a sensor reports the value that held until the next
update, so holding it forward is the only rule that needs no interpolation
assumption and stays exactly integral. Trapezoid would invent intermediate values
and introduce a division. This rule is frozen and tested exactly.

**Gross energy only.** No idle baseline is invented, estimated, or subtracted. A
"relief" that only appears after subtracting a modeled idle floor is not measured.

## 4. Matching requirements (all mandatory, all exact)

Control and treatment MUST agree exactly on:

- `workload_digest` and `trace_digest`;
- `slo_policy_digest`;
- `offered_work` and `completed_work`;
- terminal outcomes: `met`, `tardy`, `rejected`, `canceled` (each exactly equal);
- `scope`;
- `instrument_kind` and `instrument_identity`;
- `device_identity`, `source_revision`, and `build_revision`;
- `normalizer_digest` and `synchronization`;
- `included_rails` and `excluded_rails` (as sets);
- `board_uuids` (as a set) when scope is `GPU_BOARD`;
- `clock_epoch_id` (a monotonic clock is per-boot; markers from different boots are
  incomparable); and
- `effective_window_us` within `WINDOW_TOLERANCE_US`.

They MUST differ in `role`: exactly one `OPTIMIZED_SERVER_ONLY_CONTROL` and one
`Q_PIM_TREATMENT`. They MUST differ in `policy_digest` (a comparison of one policy
against itself is `E_SAME_POLICY`).

Control and treatment MUST bind distinct raw-power and execution artifacts by ID,
path, and digest. They must also have distinct `run_nonce` and
`paid_payload_sha256` values. Re-encoding one observation or changing only
out-of-window padding does not create an independent run. Reuse is
`E_DUPLICATE_RUN`.

`completed_work` equality is not enough on its own: aggregate accounting must be
CLOSED, i.e.
`met + tardy + rejected + canceled == offered_work`, on both sides
(`E_WORK_NOT_CLOSED`). Otherwise a treatment could "save energy" by leaving work
in flight at the window edge.

This is structural closure, not proof of identical realized work. The workload,
trace, and SLO digests are opaque in E2 v1, and there is no resolved per-request
terminal/output ledger. Therefore E2 v1 cannot prove model identity, actual output
tokens, output correctness, or that the same request completed on each side. No
aggregate or physical label is allowed until a later contract resolves those
artifacts.

Any mismatch is a hard error and yields `MEASUREMENT_INVALID`. E2 never compares
different work, different SLO outcomes, or different boundaries.

## 5. Conservative decision (integer-only)

    control_lower   = control_energy_nj   - control_uncertainty_nj
    treatment_upper = treatment_energy_nj + treatment_uncertainty_nj

Relief requires `treatment_upper < control_lower`. Uncertainty is never optional and
never dropped: a timeline with `uncertainty_nj` absent or zero while its
`instrument_kind` has a nonzero floor is `E_UNCERTAINTY`. For `NVML_BOARD` the floor
is the vendor-stated +/- 5 W, i.e. `uncertainty_mw >= 5000`, integrated over the
window **per measured board**. A timeline summing two boards has a conservative
floor of 10000 mW times the window; it cannot reuse the one-board floor.

The 10 percent gate uses **integer cross multiplication only**:

    relief requires:  treatment_upper * 10 <= control_lower * 9

which is `treatment_upper <= 0.9 * control_lower` with no float and no division.

Both conditions must hold. A conservative margin that is positive but under 10
percent is `RELIEF_FAIL`, not a partial win.

`boundary_delta_nj = control_energy_nj - treatment_energy_nj` names only the
declared measurement boundary. For `SERVER_WALL`, `server_wall_delta_nj` and the
phone-plus-external break-even budget equal that delta. For `GPU_BOARD`, both are
null and MUST NOT be inferred from the board delta.

## 6. Repetition set

- Minimum `MIN_PAIRS = 8` balanced pairs.
- Order is a FIXED ROTATION declared before execution (`ABBA`-style alternation),
  not post-selected. The declared order digest is bound into the set.
- **Every attempted measurement pair is represented**, including failures. A pair
  whose run failed is present with `status: FAILED` and a reason code. Any failed
  measurement pair invalidates the physical result; it is never excluded while the
  remaining pairs are used to make a claim. `attempted_pairs` must equal the number
  of listed pairs.
- No timeline ID or record digest may appear twice anywhere in measured pairs or
  warm-ups (`E_DUPLICATE_RUN`). Renaming a reused record does not create an
  independent run.
- Pair indices are exactly contiguous `0..attempted_pairs-1`; gaps cannot hide a
  failed attempt. Each pair binds both timeline IDs and both record digests.
- Warm-ups are declared separately as `{timeline_id, record_sha256}` records. A
  warm-up may not reappear in a measured pair and never enters the aggregate.
- The frozen aggregate method is `SUM_ALL_PAIRS_V1`: sum control energy and its
  uncertainty across every measurement pair, do the same for treatment, then apply
  the section 5 inequalities once to those four sums. No median, trimming, retry
  selection, or favorable-pair filtering is permitted.
- A pair comparison is diagnostic only and MUST emit `MEASUREMENT_INVALID` with
  `PAIR_ONLY_NO_AGGREGATE_CLAIM`. Only a future aggregate evaluator that validates
  the complete repetition set and recomputes every pair may emit a physical relief
  label. That evaluator is not implemented in this checkpoint, so physical labels
  are currently unreachable.

`validate_repetition_set` is a structural validator. It checks the declared order,
indices, statuses, identities, digests, and global non-reuse rules, but it does not
resolve all timeline records and cannot prove that the declaration existed before
execution. The synthetic fixture has concrete timeline records only for pair 0;
the remaining seven pairs are shape-only placeholders. It is not aggregate
evidence. A future aggregate path needs a separately anchored pre-run plan and an
attempt ledger, then must resolve every referenced record.

Rationale: post-selecting favorable trials is the single easiest way to manufacture
a 10 percent win, and it is invisible in the final numbers. The set records what was
attempted, so a drop is a diff, not a silence.

## 6b. Raw and execution artifacts v2, and read-once

A raw artifact is `e2.raw.v2`. In addition to `samples` and `pstates`, it binds
the timeline, role, provenance, run nonce, raw/execution artifact IDs,
instrument/scope identity, device and clock identity, synchronization, rails,
boards, and source/build revisions. The timeline record is rejected unless every
bound field matches exactly.

- `pstates` is REQUIRED and carries ONE nonempty printable-ASCII status string per
  sample. Nulls, lists, empty strings, control bytes, non-ASCII strings, and values
  longer than 64 bytes are schema failures.
- The artifact is read ONCE. The bytes that are hashed are the bytes that are
  parsed and integrated. Hashing a path and then re-opening it is a
  time-of-check/time-of-use gap: two opens can see two files, and a record's
  declared energy can then describe bytes other than the ones it pins.

Each timeline also binds a separate `e2.execution.v2` artifact. Its hashed bytes
carry the raw/execution IDs and raw digest, run nonce, provenance, paid-payload
digest, timeline/role/policy identities, workload/trace/SLO/schedule digests,
work, outcomes, clock epoch, markers, source/build/device identities, and terminal
status. The timeline, raw artifact, and execution artifact therefore form one
cross-bound run rather than three independently resealable declarations.

`paid_payload_sha256` hashes the exact clipped hold intervals as
`[left_us, right_us, power_mw, pstate]`, after coalescing adjacent intervals with
the same power and state. It ignores formatting, redundant equal samples, and
out-of-window padding. Control and treatment with the same paid-payload digest are
rejected as the same observation. E2 v1 integrates the complete declared marker
interval; a favorable subwindow inside it is not admissible. The execution record
does not yet prove initial/final cache, residency, thermal, or outstanding-action
state, so it cannot prove that prefetch/warm-up and transfer/cleanup work were paid.

## 6c. Quality is measured over the paid window

`independent_updates` and `max_gap_us` are computed by `window_quality` over the
paid interval and its real brackets, NOT over the whole artifact. Updates are
power changes strictly after `window_start_us` and strictly before
`window_end_us`. A sample exactly at the end only brackets the integral; its new
value applies after the interval and cannot buy an update.

The gap gate uses the actual timestamp distance between the last sample at or
before the start, every interior sample, and the first sample at or after the end.
It does not clip that distance to the marker. A stale held value from long before
the paid window therefore cannot masquerade as a fresh observation.

Samples outside the window contribute zero energy, because the integral clips to
the window. Counting changes over the whole artifact therefore made padding free:
appending busy samples outside the paid window bought unlimited
`independent_updates` for a window that still contained only 57 real changes. The
quality of a measurement must be assessed over the same interval its energy is
taken from. Status uses the same interval rule: `pstates[i]` applies to
`[sample[i], sample[i+1])`, and only statuses whose hold intervals intersect the
paid half-open window are compared. An end-marker-only change is outside; an
interior status change is `E_STATUS_CHANGE`.

## 6d. A label is derived, never supplied

`build_comparison` takes control, treatment, a trusted root, and a
`RepetitionSet`; it does NOT take a label, reason, decision detail, set digest, or
order digest. It validates the timeline evidence and set, requires the set's
scope/instrument to match the timelines, finds the exact pair by both timeline IDs
and record digests, derives the set/order bindings, and seals the result. A pair
absent from the set or a set for another boundary is `E_BINDING`. Malformed
timeline input is converted to `ComparisonError`; the builder does not leak a
traceback or partially emit a record.

This is not style. When the function accepted a label and checked only that it
was one of the four allowed strings, importing the module was enough to stamp a
digest-valid, schema-valid `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` onto
unvalidated records whose treatment burned more energy, on a host with no wall
instrument. The next checkpoint's aggregate evaluator will call this function; it
must be unable to be handed a conclusion.

`MatchedComparison` v1 is diagnostic-only: its schema fixes `result_label` to
`MEASUREMENT_INVALID` and closes `reason_code` to the two diagnostic reasons.
`validate_comparison` requires the resolved control, treatment, and repetition
set; reruns validation and decision arithmetic; and recomputes every copied source,
energy, uncertainty, gate, result, and reason field. Resealing a changed number is
therefore `E_INCOHERENT`, not valid evidence. The CLI refuses `--out` until the
aggregate evaluator exists. It never fills repetition provenance with empty-list
hashes.

A timeline whose own `status` is not `OK` can never reach the decision
(`E_PAIRS`). The schema permits `FAILED`/`INELIGIBLE` so a repetition set can
RECORD an attempt that went wrong, not so one can be compared.

## 7. Sample-quality gates (frozen BEFORE any measurement)

    MIN_INDEPENDENT_UPDATES = 100   per timeline, counted as CHANGES in the
                                    sensor value, not as rows
    MAX_SAMPLE_GAP_US       = 250000  (250 ms)
    MIN_WINDOW_US           = 1000000 (1 s)
    WINDOW_TOLERANCE_US     = 50000   (50 ms)
    MAX_SAMPLES             = 1000000

`independent_updates` is a conservative change-count gate, not proof that samples
are statistically independent. Per INSTRUMENT_AUDIT.md, NVML on
Ampere returns a 1-second average, so polling at 10 Hz yields ~10x duplicate rows.
Counting rows would let a sampler manufacture "3220 samples" from ~57 real
observations. The existing A6000 trace has 322 rows and **57 value changes**,
and is rejected on exactly this gate. The gate is not lowered to admit it.

A status change (e.g. NVML pstate or a counter reset) inside the window is
`E_STATUS_CHANGE` / `E_COUNTER_RESET`.

## 8. Synthetic timelines

A timeline whose `provenance` is `SYNTHETIC`, or whose `instrument_kind` is
`SYNTHETIC`, exercises mechanics and **emits no physical result**. Its comparison
label is forced to `MEASUREMENT_INVALID` with reason
`SYNTHETIC_NO_PHYSICAL_CLAIM`. A synthetic artifact can never support a MEASURED
comparison (`E_PROVENANCE`).

## 9. Stable diagnostics

    E_SCHEMA            schema/type violation, duplicate key, unknown field
    E_VERSION           unsupported schema or record version
    E_ARTIFACT_MISSING  raw artifact absent, unreadable, or outside the trusted root
    E_ARTIFACT_HASH     raw artifact bytes do not match the declared SHA-256
    E_ARTIFACT_PATH     path traversal, absolute path, or symlink
    E_NONMONOTONIC      timestamps not strictly increasing, or duplicated
    E_UNBRACKETED       window not bracketed by real samples
    E_GAP               sample gap exceeds MAX_SAMPLE_GAP_US
    E_UPDATES           fewer than MIN_INDEPENDENT_UPDATES sensor changes
    E_WINDOW            window shorter than MIN_WINDOW_US, or inverted
    E_STATUS_CHANGE     instrument/device status changed inside the window
    E_COUNTER_RESET     monotonic counter went backwards
    E_SCOPE             instrument_kind not permitted for the declared scope
    E_INCOMPLETE_WALL   SERVER_WALL without demonstrated complete coverage
    E_CAPABILITY_UNCERTIFIED measured wall declaration lacks acquisition evidence
    E_SCOPE_MISMATCH    control and treatment differ in scope/rails/boards
    E_WORK_MISMATCH     different offered or completed work
    E_WORK_NOT_CLOSED   met+tardy+rejected+canceled != offered_work
    E_SLO_MISMATCH      different terminal outcomes or SLO policy
    E_WORKLOAD_MISMATCH different workload or trace digest
    E_WINDOW_MISMATCH   effective windows differ beyond tolerance
    E_CLOCK_EPOCH       markers from different monotonic clock epochs
    E_DEVICE_MISMATCH   control and treatment name different measured devices
    E_BUILD_MISMATCH    control and treatment use different source/build revisions
    E_RAW_BINDING       raw evidence identity or paid payload differs from record
    E_EXECUTION_BINDING record work/markers/identity differ from execution evidence
    E_BINDING           pair is not exactly bound by the repetition set
    E_ROLE              roles not exactly one control and one treatment
    E_SAME_POLICY       control and treatment share a policy digest
    E_UNCERTAINTY       uncertainty missing, or below the instrument floor
    E_DUPLICATE_RUN     one run reused across pairs
    E_ORDER             execution order not the declared balanced rotation
    E_PAIRS             fewer than MIN_PAIRS, failed pair, or attempted != listed
    E_PROVENANCE        synthetic evidence supporting a measured claim
    E_SYSTEM_CLAIM      SYSTEM_ENERGY_SAVING appeared anywhere
    E_ADDITIVE_REUSE    an aggregate timeline was offered to the additive solver
    E_DIGEST            record digest mismatch

## 10. Honest limits

- E2 validates STRUCTURE and ARITHMETIC. It cannot detect a fabricated raw
  artifact: bytes that hash correctly and parse cleanly are accepted. The gate makes
  a lie recorded, digest-pinned, and attributable, not impossible.
- `ServerWallCapability` v1 has the same attribution limit. Its proof is a hashed
  declaration, not calibration or acquisition evidence, so measured capabilities
  are rejected rather than promoted.
- `RepetitionSet` v1 is structurally checked but not independently time-anchored.
  It cannot prove precommitment and does not resolve all run records. No aggregate
  or physical label is implemented.
- The forbidden-label scan is DEFENCE IN DEPTH ONLY. It catches case and
  punctuation variants inside one field. It CANNOT catch a phrase split across two
  non-adjacent free-form fields, because canonical JSON sorts keys and no
  contiguous substring scan can. That is acceptable: what actually blocks the
  claim is that `result_label` is a closed enum in both the schema and
  `ALLOWED_LABELS`, so no combination of free-form strings can BE a label.
  `tests/test_e2.py` records this limit as an executable test rather than a
  footnote.
- `route_schedule_digest` is recorded provenance, deliberately NOT matched:
  control and treatment run different schedules by definition, so requiring
  equality would be incoherent. `normalizer_digest` IS matched between the pair
  and pinned to the frozen normalizer identity.
- E2 says nothing about phones. It never has, and its schema cannot express it.
- A `GPU_BOARD` relief result is relief on ONE boundary. The host CPU, DRAM, and PSU
  losses are outside it and may have moved in the opposite direction.
- No measurement is authorized by this checkpoint.
