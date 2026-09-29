"""Synchronized physical energy-window aggregation for runtime receipts."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Mapping, Sequence

from .._internal.runtime_capabilities import RuntimePhonePowerProfile
from .contracts import PhysicalAdapterError, RawEnergyMeasurement


@dataclass(frozen=True)
class PhonePowerEstimate:
    active_time_ns: int
    idle_time_ns: int
    active_power_mw: int
    idle_power_mw: int
    energy_uj: int
    evidence_kind: str
    estimation_version: str
    charging_state: str
    activity_interval_count: int


class PhoneActivityIntervalTracker:
    """Track the union of work performed by one physical phone."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, tuple[str, int]] = {}
        self._completed: list[tuple[int, int, str]] = []

    def begin(self, work_id: str, kind: str, started_ns: int) -> None:
        if (
            type(work_id) is not str
            or not work_id
            or not work_id.isascii()
            or type(kind) is not str
            or not kind
            or not kind.isascii()
            or type(started_ns) is not int
            or started_ns < 0
        ):
            raise PhysicalAdapterError("phone activity start is invalid")
        with self._lock:
            if work_id in self._active:
                raise PhysicalAdapterError("phone activity is duplicated")
            self._active[work_id] = (kind, started_ns)

    def finish(
        self, work_id: str, finished_ns: int, *, record: bool = True
    ) -> None:
        if type(finished_ns) is not int or finished_ns < 0:
            raise PhysicalAdapterError("phone activity finish is invalid")
        if type(record) is not bool:
            raise PhysicalAdapterError("phone activity record flag is invalid")
        with self._lock:
            started = self._active.pop(work_id, None)
            if started is None:
                raise PhysicalAdapterError("phone activity is absent")
            kind, started_ns = started
            if finished_ns < started_ns:
                raise PhysicalAdapterError(
                    "phone activity finishes before it starts"
                )
            if record and finished_ns > started_ns:
                self._completed.append((started_ns, finished_ns, kind))

    def record(
        self, work_id: str, kind: str, started_ns: int, finished_ns: int
    ) -> None:
        self.begin(work_id, kind, started_ns)
        self.finish(work_id, finished_ns)

    def overlaps(self, start_ns: int, end_ns: int) -> bool:
        with self._lock:
            return any(left < end_ns and right > start_ns
                       for left, right, _ in self._completed) or any(
                left < end_ns for _, left in self._active.values()
            )

    def estimate(
        self,
        start_ns: int,
        end_ns: int,
        profile: RuntimePhonePowerProfile,
        *,
        charging_state: str,
    ) -> PhonePowerEstimate:
        if (
            type(start_ns) is not int
            or type(end_ns) is not int
            or end_ns <= start_ns
            or not isinstance(profile, RuntimePhonePowerProfile)
            or charging_state not in {
                "charging", "mixed", "not_charging", "unknown"
            }
        ):
            raise PhysicalAdapterError("phone activity estimate is invalid")
        with self._lock:
            source = list(self._completed)
            source.extend(
                (started_ns, end_ns, kind)
                for kind, started_ns in self._active.values()
                if started_ns < end_ns
            )
        clipped = sorted(
            (max(start_ns, left), min(end_ns, right))
            for left, right, _kind in source
            if left < end_ns and right > start_ns
        )
        merged: list[list[int]] = []
        for left, right in clipped:
            if not merged or left > merged[-1][1]:
                merged.append([left, right])
            else:
                merged[-1][1] = max(merged[-1][1], right)
        active_ns = sum(right - left for left, right in merged)
        duration_ns = end_ns - start_ns
        if active_ns > duration_ns:
            raise PhysicalAdapterError("phone activity exceeds its boundary")
        idle_ns = duration_ns - active_ns
        energy_uj = (
            profile.active_power_mw * active_ns
            + profile.idle_power_mw * idle_ns
            + 500_000
        ) // 1_000_000
        return PhonePowerEstimate(
            active_time_ns=active_ns,
            idle_time_ns=idle_ns,
            active_power_mw=profile.active_power_mw,
            idle_power_mw=profile.idle_power_mw,
            energy_uj=energy_uj,
            evidence_kind=profile.evidence_kind,
            estimation_version=profile.estimation_version,
            charging_state=charging_state,
            activity_interval_count=len(merged),
        )


def integrate_milliwatt_samples(
    rows: Sequence[Mapping[str, int]],
    field: str,
    start_ns: int,
    end_ns: int,
) -> int:
    if type(start_ns) is not int or type(end_ns) is not int or end_ns <= start_ns:
        raise PhysicalAdapterError("power integration interval is invalid")
    points = sorted(
        (int(row["host_sample_t_ns"]), int(row[field])) for row in rows
    )
    if (
        len(points) < 2
        or points[0][0] > start_ns
        or points[-1][0] < end_ns
    ):
        raise PhysicalAdapterError("power sample coverage is incomplete")

    def interpolate(target_ns: int) -> float:
        for left, right in zip(points, points[1:]):
            if target_ns <= right[0]:
                if right[0] <= left[0]:
                    raise PhysicalAdapterError("power sample order is invalid")
                fraction = (target_ns - left[0]) / (right[0] - left[0])
                return left[1] + fraction * (right[1] - left[1])
        raise PhysicalAdapterError("power interpolation coverage is incomplete")

    bounded: list[tuple[int, float]] = [(start_ns, interpolate(start_ns))]
    bounded.extend(
        (sample_ns, float(power_mw))
        for sample_ns, power_mw in points
        if start_ns < sample_ns < end_ns
    )
    bounded.append((end_ns, interpolate(end_ns)))
    energy_uj = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2_000_000
        for left, right in zip(bounded, bounded[1:])
    )
    return max(0, round(energy_uj))


class PolledPhonePowerSampler:
    """Collect timestamp-aligned USB and battery power observations."""

    def __init__(
        self,
        probe: Callable[[], Mapping[str, int] | None],
        *,
        history_probe: Callable[
            [], Sequence[Mapping[str, int]]
        ] | None = None,
        interval_s: float = 0.2,
        coverage_timeout_s: float = 3.0,
        maximum_gap_s: float = 5.0,
        maximum_continuity_gap_s: float | None = None,
    ) -> None:
        if maximum_continuity_gap_s is None:
            maximum_continuity_gap_s = max(maximum_gap_s * 2, 60.0)
        if (
            not callable(probe)
            or (history_probe is not None and not callable(history_probe))
            or interval_s <= 0
            or coverage_timeout_s <= 0
            or maximum_gap_s < interval_s
            or maximum_continuity_gap_s < maximum_gap_s
        ):
            raise PhysicalAdapterError("phone power sampler setup is invalid")
        self._probe = probe
        self._history_probe = history_probe
        self._interval_s = interval_s
        self._coverage_timeout_s = coverage_timeout_s
        self._maximum_gap_ns = round(maximum_gap_s * 1_000_000_000)
        self._maximum_continuity_gap_ns = round(
            maximum_continuity_gap_s * 1_000_000_000
        )
        self.rows: list[dict[str, int]] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.last_probe_error: str | None = None
        self.events: list[dict[str, int | str]] = []
        self.reboot_boundaries_ns: list[int] = []
        self._bridged_gaps: set[tuple[int, int]] = set()
        self._dropped_clock_outliers: set[int] = set()
        self.thread = threading.Thread(
            target=self._run,
            name="runtime-phone-power-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    value = self._probe()
                except Exception as error:
                    self.last_probe_error = (
                        f"{type(error).__name__}: {error}"
                    )
                    with self.lock:
                        self.events.append({
                            "event": "PROBE_ERROR",
                            "host_sample_t_ns": time.monotonic_ns(),
                            "reason": self.last_probe_error,
                        })
                    self.stop_event.wait(self._interval_s)
                    continue
                if value is not None:
                    row = {key: int(item) for key, item in value.items()}
                    with self.lock:
                        if (
                            not self.rows
                            or row["host_sample_t_ns"]
                                > self.rows[-1]["host_sample_t_ns"]
                        ):
                            if (
                                self.rows
                                and row["phone_uptime_ns"]
                                    < self.rows[-1]["phone_uptime_ns"]
                            ):
                                self.reboot_boundaries_ns.append(
                                    row["host_sample_t_ns"]
                                )
                                self.events.append({
                                    "event": "PHONE_REBOOT",
                                    "host_sample_t_ns": (
                                        row["host_sample_t_ns"]
                                    ),
                                    "reason": "PHONE_UPTIME_DECREASED",
                                })
                            self.rows.append(row)
                            self.last_probe_error = None
                else:
                    with self.lock:
                        self.events.append({
                            "event": "PROBE_UNAVAILABLE",
                            "host_sample_t_ns": time.monotonic_ns(),
                            "reason": "NO_SAMPLE",
                        })
                self.stop_event.wait(self._interval_s)
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"

    def _merge_history(self) -> None:
        if self._history_probe is None:
            return
        history = tuple(
            {key: int(value) for key, value in row.items()}
            for row in self._history_probe()
        )
        if not history:
            return
        required = {
            "battery_discharge_power_mw",
            "phone_uptime_ns",
            "usb_input_power_mw",
        }
        if any(
            not required.issubset(row)
            or row["phone_uptime_ns"] <= 0
            for row in history
        ):
            raise PhysicalAdapterError(
                "phone-local power history is invalid"
            )
        if any(
            right["phone_uptime_ns"] <= left["phone_uptime_ns"]
            for left, right in zip(history, history[1:])
        ):
            raise PhysicalAdapterError(
                "phone-local power history crosses a reboot"
            )
        with self.lock:
            if not self.rows:
                return
            anchor = self.rows[-1]
            offset_ns = (
                anchor["host_sample_t_ns"] - anchor["phone_uptime_ns"]
            )
            by_uptime = {
                row["phone_uptime_ns"]: row for row in self.rows
            }
            for row in history:
                uptime_ns = row["phone_uptime_ns"]
                if uptime_ns in by_uptime:
                    continue
                aligned = dict(row)
                aligned["host_sample_t_ns"] = offset_ns + uptime_ns
                by_uptime[uptime_ns] = aligned
            self.rows[:] = sorted(
                by_uptime.values(),
                key=lambda row: row["host_sample_t_ns"],
            )
            self.events.append({
                "event": "PHONE_LOCAL_HISTORY_MERGED",
                "host_sample_t_ns": time.monotonic_ns(),
                "reason": str(len(history)),
            })

    def _drop_isolated_clock_outliers(
        self,
        rows: Sequence[dict[str, int]],
        reboot_boundaries: Sequence[int],
    ) -> list[dict[str, int]]:
        if len(rows) < 3:
            return list(rows)
        result = [rows[0]]
        for index in range(1, len(rows) - 1):
            left = rows[index - 1]
            middle = rows[index]
            right = rows[index + 1]
            left_host = left["host_sample_t_ns"]
            middle_host = middle["host_sample_t_ns"]
            right_host = right["host_sample_t_ns"]
            left_uptime = left["phone_uptime_ns"]
            middle_uptime = middle["phone_uptime_ns"]
            right_uptime = right["phone_uptime_ns"]
            direct_host_gap = right_host - left_host
            direct_uptime_gap = right_uptime - left_uptime
            neighbor_offsets_match = abs(
                (left_host - left_uptime)
                - (right_host - right_uptime)
            ) <= 1_000_000_000
            middle_offset = middle_host - middle_uptime
            neighbor_offset = (
                (left_host - left_uptime)
                + (right_host - right_uptime)
            ) // 2
            isolated_offset = (
                abs(middle_offset - neighbor_offset) > 1_000_000_000
            )
            continuous = (
                0 < direct_host_gap <= self._maximum_continuity_gap_ns
                and direct_uptime_gap > 0
                and abs(direct_host_gap - direct_uptime_gap)
                    <= 1_000_000_000
            )
            adjacent_gaps_are_exposed = (
                middle_host - left_host > self._maximum_gap_ns
                and right_host - middle_host > self._maximum_gap_ns
            )
            crosses_reboot = any(
                left_host <= boundary_ns <= right_host
                for boundary_ns in reboot_boundaries
            )
            if (
                left_host < middle_host < right_host
                and left_uptime < middle_uptime < right_uptime
                and neighbor_offsets_match
                and isolated_offset
                and continuous
                and adjacent_gaps_are_exposed
                and not crosses_reboot
            ):
                with self.lock:
                    if middle_host not in self._dropped_clock_outliers:
                        self._dropped_clock_outliers.add(middle_host)
                        self.events.append({
                            "event": (
                                "PHONE_POWER_CLOCK_OUTLIER_DROPPED"
                            ),
                            "host_sample_t_ns": middle_host,
                            "reason": str(
                                middle_offset - neighbor_offset
                            ),
                        })
                continue
            result.append(middle)
        result.append(rows[-1])
        return result

    def energy_between(self, start_ns: int, end_ns: int) -> dict[str, int]:
        self._merge_history()
        deadline = time.monotonic() + self._coverage_timeout_s
        while True:
            with self.lock:
                rows = list(self.rows)
            if (
                len(rows) >= 2
                and rows[0]["host_sample_t_ns"] <= start_ns
                and rows[-1]["host_sample_t_ns"] >= end_ns
            ):
                break
            if self.error is not None or time.monotonic() >= deadline:
                raise PhysicalAdapterError(
                    "phone power sample coverage is incomplete"
                )
            time.sleep(0.02)
        with self.lock:
            reboot_boundaries = tuple(self.reboot_boundaries_ns)
        if any(start_ns <= row <= end_ns for row in reboot_boundaries):
            raise PhysicalAdapterError(
                "phone power interval crosses a reboot"
            )
        rows = self._drop_isolated_clock_outliers(
            rows, reboot_boundaries
        )
        covered = tuple(
            row for row in rows
            if start_ns <= row["host_sample_t_ns"] <= end_ns
        )
        before = tuple(
            row for row in rows if row["host_sample_t_ns"] < start_ns
        )
        after = tuple(
            row for row in rows if row["host_sample_t_ns"] > end_ns
        )
        interval_rows = (
            ((before[-1],) if before else ())
            + covered
            + ((after[0],) if after else ())
        )
        for left, right in zip(interval_rows, interval_rows[1:]):
            host_gap_ns = (
                right["host_sample_t_ns"] - left["host_sample_t_ns"]
            )
            if host_gap_ns <= self._maximum_gap_ns:
                continue
            uptime_gap_ns = (
                right["phone_uptime_ns"] - left["phone_uptime_ns"]
            )
            if (
                host_gap_ns > self._maximum_continuity_gap_ns
                or uptime_gap_ns <= 0
                or abs(host_gap_ns - uptime_gap_ns) > 1_000_000_000
            ):
                raise PhysicalAdapterError(
                    "phone power sample coverage contains a gap"
                )
            gap = (
                left["host_sample_t_ns"], right["host_sample_t_ns"]
            )
            with self.lock:
                if gap not in self._bridged_gaps:
                    self._bridged_gaps.add(gap)
                    self.events.append({
                        "event": "PHONE_POWER_CONTINUITY_BRIDGE",
                        "host_sample_t_ns": right["host_sample_t_ns"],
                        "reason": (
                            str(host_gap_ns) + ":" + str(uptime_gap_ns)
                        ),
                    })
        return {
            "battery_discharge_uj": integrate_milliwatt_samples(
                rows, "battery_discharge_power_mw", start_ns, end_ns
            ),
            "sample_count": sum(
                start_ns <= row["host_sample_t_ns"] <= end_ns
                for row in rows
            ),
            "usb_input_uj": integrate_milliwatt_samples(
                rows, "usb_input_power_mw", start_ns, end_ns
            ),
        }

    def charging_state_between(self, start_ns: int, end_ns: int) -> str:
        """Return charging telemetry without using it as route energy."""
        if (
            type(start_ns) is not int
            or type(end_ns) is not int
            or end_ns <= start_ns
        ):
            raise PhysicalAdapterError("phone charging interval is invalid")
        with self.lock:
            rows = tuple(self.rows)
            reboot_boundaries = tuple(self.reboot_boundaries_ns)
        if any(start_ns <= value <= end_ns for value in reboot_boundaries):
            raise PhysicalAdapterError(
                "phone charging interval crosses a reboot"
            )
        relevant = tuple(
            row for row in rows
            if start_ns <= row["host_sample_t_ns"] <= end_ns
        )
        if not relevant:
            before = tuple(
                row for row in rows
                if row["host_sample_t_ns"] < start_ns
            )
            after = tuple(
                row for row in rows
                if row["host_sample_t_ns"] > end_ns
            )
            relevant = (
                ((before[-1],) if before else ())
                + ((after[0],) if after else ())
            )
        states = {
            bool(row.get("charging", row.get("usb_input_power_mw", 0) > 0))
            for row in relevant
        }
        if states == {True}:
            return "charging"
        if states == {False}:
            return "not_charging"
        return "mixed" if states else "unknown"

    def diagnostics(self) -> dict[str, object]:
        with self.lock:
            return {
                "events": [dict(row) for row in self.events],
                "last_probe_error": self.last_probe_error,
                "reboot_boundaries_ns": list(self.reboot_boundaries_ns),
                "rows": [dict(row) for row in self.rows],
                "schema": "heterogeneous-phone-power-diagnostics-v1",
            }

    def wait_until_ready(self, after_ns: int | None = None) -> int:
        """Wait for one fresh sample before opening a measurement window."""
        target_ns = time.monotonic_ns() if after_ns is None else after_ns
        if type(target_ns) is not int or target_ns < 0:
            raise PhysicalAdapterError(
                "phone power readiness timestamp is invalid"
            )
        deadline = time.monotonic() + self._coverage_timeout_s
        while True:
            with self.lock:
                latest_ns = (
                    None
                    if not self.rows
                    else self.rows[-1]["host_sample_t_ns"]
                )
            if latest_ns is not None and latest_ns >= target_ns:
                return latest_ns
            if self.error is not None or time.monotonic() >= deadline:
                raise PhysicalAdapterError(
                    "phone power sampler is not ready"
                )
            time.sleep(0.02)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(5)
        if self.thread.is_alive() or self.error is not None:
            raise PhysicalAdapterError("phone power sampler did not stop")


class RaplNvmlPhoneEnergyMeter:
    """Aggregate CPU package, GPU board, and whole-phone energy."""

    def __init__(
        self,
        server_rows: Callable[[], Sequence[Mapping[str, object]]],
        server_summary: Callable[
            [Sequence[Mapping[str, object]], int, int], Mapping[str, float]
        ],
        phone_sampler: PolledPhonePowerSampler,
        *,
        energy_boundary_id: str,
        attribution_kind: str = "diagnostic",
        coverage_timeout_s: float = 60.0,
        maximum_gap_s: float = 5.0,
        server_window_rows: Callable[
            [int, int], Sequence[Mapping[str, object]]
        ] | None = None,
        phone_power_profile: RuntimePhonePowerProfile | None = None,
        phone_activity: PhoneActivityIntervalTracker | None = None,
        helper_phone_power: Mapping[str, tuple[RuntimePhonePowerProfile, PhoneActivityIntervalTracker]] | None = None,
    ) -> None:
        if (
            not callable(server_rows)
            or not callable(server_summary)
            or (
                server_window_rows is not None
                and not callable(server_window_rows)
            )
        ):
            raise PhysicalAdapterError("server energy source is invalid")
        if not isinstance(phone_sampler, PolledPhonePowerSampler):
            raise PhysicalAdapterError("phone energy source is invalid")
        if coverage_timeout_s <= 0 or maximum_gap_s <= 0:
            raise PhysicalAdapterError("energy coverage timeout is invalid")
        if (
            type(energy_boundary_id) is not str
            or not energy_boundary_id
            or not energy_boundary_id.isascii()
        ):
            raise PhysicalAdapterError("energy boundary id is invalid")
        if attribution_kind not in {
            "diagnostic", "isolated", "matched_abba", "device_domain"
        }:
            raise PhysicalAdapterError(
                "energy attribution kind is invalid"
            )
        if phone_power_profile is not None and phone_activity is None:
            raise PhysicalAdapterError(
                "assumed phone energy requires a profile and activity"
            )
        self._server_rows = server_rows
        self._server_window_rows = server_window_rows
        self._server_summary = server_summary
        self._phone_sampler = phone_sampler
        self._coverage_timeout_s = coverage_timeout_s
        self._maximum_gap_ns = round(maximum_gap_s * 1_000_000_000)
        self._energy_boundary_id = energy_boundary_id
        self._attribution_kind = attribution_kind
        self._phone_power_profile = phone_power_profile
        self._phone_activity = phone_activity
        self._helper_phone_power = dict(helper_phone_power or {})
        domains = {"cpu-package", "gpu-board", "phone-system" if phone_power_profile is None
                   else phone_power_profile.domain_id}
        for device, (profile, activity) in self._helper_phone_power.items():
            if (profile.device_id != device or profile.domain_id in domains
                    or not isinstance(activity, PhoneActivityIntervalTracker)):
                raise PhysicalAdapterError("helper phone power identity is invalid")
            domains.add(profile.domain_id)
        self._receipt_context = None
        self._epoch_ns = 0

    def bind_scheduler(self, scheduler: object, epoch_ns: int) -> None:
        context = getattr(scheduler, "runtime_receipt_energy_context", None)
        if not callable(context):
            raise PhysicalAdapterError("energy attribution ledger is absent")
        self._receipt_context = context
        self._epoch_ns = epoch_ns

    def measure_receipt(self, command, start_ns: int, end_ns: int) -> RawEnergyMeasurement:
        return self.measure(start_ns, end_ns, receipt=command)

    def _receipt_attribution(self, receipt, start_ns: int, end_ns: int):
        if receipt is None:
            return "diagnostic", {"energy_attribution_reason": "ENERGY_CAMPAIGN_WINDOW"}
        if self._receipt_context is None or self._phone_activity is None:
            return "diagnostic", {"energy_attribution_reason": "ENERGY_LEDGER_UNAVAILABLE"}
        if self._phone_activity.overlaps(start_ns, end_ns) or any(
            activity.overlaps(start_ns, end_ns) for _, activity in self._helper_phone_power.values()
        ):
            return "diagnostic", {"energy_attribution_reason": "ENERGY_PHONE_ACTIVITY_OVERLAP"}
        participants = getattr(receipt, "participants", None)
        if participants is None:
            participants = (receipt.participant,)
        context = self._receipt_context(
            receipt.ticket_id, tuple(row.device_id for row in participants),
            (start_ns - self._epoch_ns) // 1000,
            (end_ns - self._epoch_ns) // 1000,
        )
        reason = context["reason"]
        return (
            "isolated" if reason == "ENERGY_ISOLATED_SERVER_RECEIPT" else "diagnostic",
            {
                "energy_attribution_reason": reason,
                "energy_attribution_ticket_id": receipt.ticket_id,
                "energy_attribution_overlapping_tickets": ",".join(
                    context.get("overlapping_ticket_ids", ())
                ) or "none",
            },
        )

    def record_phone_activity_duration(
        self,
        work_id: str,
        kind: str,
        start_ns: int,
        end_ns: int,
        active_us: int,
    ) -> None:
        """Record aggregate shared-phone work reported by the runtime."""
        if self._phone_activity is None:
            return
        if (
            type(active_us) is not int
            or active_us < 0
            or end_ns <= start_ns
        ):
            raise PhysicalAdapterError(
                "reported phone activity duration is invalid"
            )
        active_ns = min(end_ns - start_ns, active_us * 1_000)
        if active_ns:
            self._phone_activity.record(
                work_id, kind, start_ns, start_ns + active_ns
            )

    @staticmethod
    def _valid_server_rows(
        rows: Sequence[Mapping[str, object]],
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(
            row for row in rows
            if type(row.get("gpu")) is dict
            and type(row.get("rapl_package")) is dict
            and type(row["gpu"].get("sample_t_ns")) is int
            and type(row["rapl_package"].get("sample_t_ns")) is int
        )

    def record_helper_phone_window(self, work_id, start_ns, end_ns, device_ids):
        for device in device_ids:
            pair = self._helper_phone_power.get(device)
            if pair is not None and end_ns > start_ns:
                # Per-helper compute counters are not exposed; charge the full assisted window.
                pair[1].record(work_id, "assisted-window-upper-bound", start_ns, end_ns)

    @staticmethod
    def _covered(
        rows: Sequence[Mapping[str, object]], start_ns: int, end_ns: int
    ) -> bool:
        if len(rows) < 2:
            return False
        first = rows[0]
        last = rows[-1]
        first_gpu = first.get("gpu")
        last_gpu = last.get("gpu")
        first_rapl = first.get("rapl_package")
        last_rapl = last.get("rapl_package")
        return (
            type(first_gpu) is dict
            and type(last_gpu) is dict
            and type(first_rapl) is dict
            and type(last_rapl) is dict
            and first_gpu.get("sample_t_ns", start_ns + 1) <= start_ns
            and last_gpu.get("sample_t_ns", end_ns - 1) >= end_ns
            and first_rapl.get("sample_t_ns", start_ns + 1) <= start_ns
            and last_rapl.get("sample_t_ns", end_ns - 1) >= end_ns
        )

    def _require_contiguous_coverage(
        self,
        rows: Sequence[Mapping[str, object]],
        start_ns: int,
        end_ns: int,
    ) -> None:
        for name in ("gpu", "rapl_package"):
            points = sorted(int(row[name]["sample_t_ns"]) for row in rows)
            before = [sample_ns for sample_ns in points if sample_ns <= start_ns]
            after = [sample_ns for sample_ns in points if sample_ns >= end_ns]
            if not before or not after:
                raise PhysicalAdapterError(
                    "server energy sample coverage is incomplete"
                )
            bounded = (
                [before[-1]]
                + [
                    sample_ns for sample_ns in points
                    if start_ns < sample_ns < end_ns
                ]
                + [after[0]]
            )
            if any(
                right_ns - left_ns > self._maximum_gap_ns
                for left_ns, right_ns in zip(bounded, bounded[1:])
            ):
                raise PhysicalAdapterError(
                    "server energy sample coverage contains a gap"
                )

    def measure(self, start_ns: int, end_ns: int, *, receipt=None) -> RawEnergyMeasurement:
        deadline = time.monotonic() + self._coverage_timeout_s
        rows_source = (
            self._server_rows
            if self._server_window_rows is None
            else lambda: self._server_window_rows(start_ns, end_ns)
        )
        rows = self._valid_server_rows(tuple(rows_source()))
        while not self._covered(rows, start_ns, end_ns):
            if time.monotonic() >= deadline:
                raise PhysicalAdapterError(
                    "server energy sample coverage is incomplete"
                )
            time.sleep(0.02)
            rows = self._valid_server_rows(tuple(rows_source()))
        self._require_contiguous_coverage(rows, start_ns, end_ns)
        server = self._server_summary(rows, start_ns, end_ns)
        phone_estimate = None
        if self._phone_power_profile is None:
            phone = self._phone_sampler.energy_between(start_ns, end_ns)
            phone_energy_uj = (
                phone["battery_discharge_uj"] + phone["usb_input_uj"]
            )
            phone_evidence = "physical:phone-usb-plus-battery-power"
            estimation_metadata = {}
        else:
            assert self._phone_activity is not None
            phone_estimate = self._phone_activity.estimate(
                start_ns,
                end_ns,
                self._phone_power_profile,
                charging_state=self._phone_sampler.charging_state_between(
                    start_ns, end_ns
                ),
            )
            phone_energy_uj = phone_estimate.energy_uj
            phone_evidence = phone_estimate.evidence_kind
            estimation_metadata = {
                "phone_active_power_mw": phone_estimate.active_power_mw,
                "phone_active_time_ns": phone_estimate.active_time_ns,
                "phone_activity_interval_count": (
                    phone_estimate.activity_interval_count
                ),
                "phone_charging_state": phone_estimate.charging_state,
                "phone_energy_evidence": phone_estimate.evidence_kind,
                "phone_energy_uj": phone_estimate.energy_uj,
                "phone_estimation_version": (
                    phone_estimate.estimation_version
                ),
                "phone_idle_power_mw": phone_estimate.idle_power_mw,
                "phone_idle_time_ns": phone_estimate.idle_time_ns,
            }
        attribution_kind, attribution = self._receipt_attribution(receipt, start_ns, end_ns)
        estimation_metadata.update(attribution)
        helper_domains = {}
        helper_evidence = []
        for device, (profile, activity) in self._helper_phone_power.items():
            estimate = activity.estimate(start_ns, end_ns, profile, charging_state="unknown")
            helper_domains[profile.domain_id] = estimate.energy_uj
            helper_evidence.extend((estimate.evidence_kind, estimate.estimation_version))
            estimation_metadata.update({
                "helper:" + device + ":energy_uj": estimate.energy_uj,
                "helper:" + device + ":active_time_ns": estimate.active_time_ns,
                "helper:" + device + ":idle_time_ns": estimate.idle_time_ns,
                "helper:" + device + ":evidence": estimate.evidence_kind,
            })
        return RawEnergyMeasurement(
            energy_boundary_id=self._energy_boundary_id,
            fleet_energy_uj_by_domain={
                **helper_domains,
                "cpu-package": round(
                    float(server["cpu_package_energy_j"]) * 1_000_000
                ),
                "gpu-board": round(
                    float(server["gpu_board_energy_j"]) * 1_000_000
                ),
                (
                    "phone-system"
                    if self._phone_power_profile is None
                    else self._phone_power_profile.domain_id
                ): phone_energy_uj,
            },
            measurement_evidence_ids=tuple(dict.fromkeys((
                "physical:rapl-package-0",
                "physical:nvml-board-power",
                phone_evidence,
                *helper_evidence,
                *(
                    ()
                    if phone_estimate is None
                    else (phone_estimate.estimation_version,)
                ),
            ))),
            attribution_kind=attribution_kind,
            estimation_metadata=estimation_metadata,
        )

    def prepare(self) -> None:
        """Require fresh samples from the sources used for energy measurement."""
        target_ns = time.monotonic_ns()
        deadline = time.monotonic() + self._coverage_timeout_s
        while True:
            rows = self._valid_server_rows(tuple(self._server_rows()))
            if rows:
                latest = rows[-1]
                gpu = latest["gpu"]
                rapl = latest["rapl_package"]
                if (
                    int(gpu["sample_t_ns"]) >= target_ns
                    and int(rapl["sample_t_ns"]) >= target_ns
                ):
                    break
            if time.monotonic() >= deadline:
                raise PhysicalAdapterError(
                    "server energy sampler is not ready"
                )
            time.sleep(0.02)
        if self._phone_power_profile is None:
            self._phone_sampler.wait_until_ready(target_ns)
