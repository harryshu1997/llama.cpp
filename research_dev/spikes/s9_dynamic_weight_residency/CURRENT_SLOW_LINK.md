# Current USB Link Baseline

Status: USB negotiation and practical ADB staging goodput MEASURED on both phones.
The historical filename is retained to avoid path churn. These measurements are an
end-to-end provisioning baseline, not a raw USB or memory-to-memory transport rate.

Captured 2026-07-13 on the host at HEAD `933c722f6`.

## 1. Physical topology

Both endpoints negotiate USB 3.2 Gen 1 at 5000 Mbps. They are attached to separate
SuperSpeed root buses, so the current two-phone topology must not be modeled as one
statically divided controller.

```text
OP12  serial 5ae7a43d          sysfs path 6-2  Bus 006  endpoint 5000M
OP15  serial 3C15AU002CL00000  sysfs path 8-3  Bus 008  endpoint 5000M
```

Each host root advertises 10000M, but each phone endpoint advertises 5000M. Thus the
per-phone signaling ceiling is 5 Gbps raw, or 625,000,000 B/s, regardless of a
higher-rated cable. This ceiling is not an application-goodput measurement.

## 2. Measurement method

- Payload: 1 GiB (`1073741824` bytes) from `/dev/urandom`, so compression cannot
  explain the result.
- Transport: ADB sync with compression disabled (`adb push -Z`, `adb pull -Z`).
- Isolation: three solo repetitions per phone and direction; pull sink in `/dev/shm`.
- Concurrency: one simultaneous two-phone push and one simultaneous two-phone pull.
- Integrity: host and both phone copies matched SHA-256
  `ba7b41e067da12e9d554ad47f053d716ddeedf6a75b7b73257b7d7b886b02921`.
- Access: a separate non-disruptive ADB server on port 5038 avoided changing the
  other user's default ADB server.

ADB prints a value labeled `MB/s`, but its numeric convention is MiB/s. Both decimal
MB/s and MiB/s are shown below; canonical evidence is bytes, elapsed seconds, and
integer bytes/s.

## 3. Solo results

| Phone | Path | Direction | Elapsed samples (s) | Median (s) | Median B/s | Decimal MB/s | MiB/s |
|---|---|---|---|---:|---:|---:|---:|
| OP12 | host -> phone file | push | 4.742, 4.759, 4.731 | 4.742 | 226432270 | 226.4 | 215.9 |
| OP15 | host -> phone file | push | 3.789, 3.910, 3.912 | 3.910 | 274614277 | 274.6 | 261.9 |
| OP12 | phone file -> host | pull | 5.267, 5.436, 5.394 | 5.394 | 199062259 | 199.1 | 189.8 |
| OP15 | phone file -> host | pull | 4.515, 4.450, 4.595 | 4.515 | 237816572 | 237.8 | 226.8 |

The push path includes host file read, ADB framing, USB transfer, Android handling,
and phone file write/page-cache effects. The pull path includes phone file read,
ADB framing, USB transfer, and host delivery. Timed `adb push` does not prove a
durable `fsync`, so the profile names claim file staging, not durable UFS publish:

```text
adb_host_to_phone_file
adb_phone_file_to_host
```

Do not insert either value as a raw link rate while also charging a separate UFS
stage; that would double-count storage work.

## 4. Concurrent two-phone results

| Direction | OP12 stream | OP15 stream | Fleet wall (s) | Fleet bytes | Aggregate B/s | Decimal MB/s | MiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| host -> phone file | 207.2 MiB/s | 258.0 MiB/s | 5.002 | 2147483648 | 429325000 | 429.3 | 409.4 |
| phone file -> host | 197.8 MiB/s | 218.6 MiB/s | 5.216 | 2147483648 | 411710822 | 411.7 | 392.6 |

`Aggregate B/s` is total bytes divided by the parallel makespan. It is not the sum
of independently rounded ADB display rates. The small solo-to-concurrent change is
consistent with the phones using separate SuperSpeed root buses.

## 5. Provisioning time scale

The following is staged-transfer time only. It excludes SHA-256 verification,
materialization, backend preparation, warmup, and transfer/compute interference.

| Resident payload | OP12 at 215.9 MiB/s | OP15 at 261.9 MiB/s |
|---|---:|---:|
| 440 MiB | 2.0 s | 1.7 s |
| 900 MiB | 4.2 s | 3.4 s |
| 10 GiB | 47 s | 39 s |
| 24 GiB | 114 s | 94 s |

This is fast enough for predictive residency and background model turnover, but not
for request-critical weight fetching. Runtime dispatch remains READY-only; a miss
falls back to the server while the slow loop may initiate a later prefetch.

## 6. Planning values and open measurements

The measurements establish practical ADB-to-file staging rates and independent-link
topology. They do not establish native bulk transport capability. S9-V1 must keep
the measured staged profile separate from a decomposed profile:

```text
native H2D memory-to-memory transport
native D2H memory-to-memory transport
UFS write and read
durable fsync and atomic publish
SHA-256 verification
UFS-to-LPDDR materialization
backend preparation and warmup
transfer <-> HTP/GPU interference
```

Until those legs are measured, the staged profile may estimate cold provisioning
wall time only. A future USB 10 Gbps device profile may be swept at 800 and 1000
MiB/s as an explicitly unmeasured sensitivity study; it must not be reported as a
current OP12/OP15 result or as a capacity claim.
