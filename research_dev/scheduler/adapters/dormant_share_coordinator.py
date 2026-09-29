"""Rig-level admission for the decode-only FFN relocation.

One coordinator per physical rig owns a memory ledger over a fixed host budget. Every launched server is
booked with its resident footprint; a server launched with the dormant host share is additionally booked
with the largest share it may release (``ShareBinding``). Control acknowledgements carry the server's
release state (``dormant_release_generation``, ``dormant_released_bytes``, layers, columns); each new
generation is credited once through ``DecodeReleaseAccountant``. Before a prompt is sent to a server whose
share is released, ``before_prompt`` re-reserves the share, waiting while the room is taken (another
server's footprint, a growth reservation) and recording the hold. Physical eviction is never attempted
here: when the room does not return within the deadline the prompt fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Callable, Mapping

from .._internal.capacity import DeviceMemoryCapacity
from .._internal.decode_split_selection import DecodeReleaseAccountant, DecodeSplitSelectionError, ShareBinding
from .._internal.runtime_cost import RuntimeMemoryDemand
from .._internal.runtime_placement import RuntimePlacementSnapshot
from .._internal.runtime_resources import RuntimeHostShareReleaseProof, RuntimeMemoryLedger, RuntimeResourceError
from .contracts import PhysicalAdapterError


class DormantShareAdmissionError(PhysicalAdapterError):
    pass


@dataclass
class _ServerBooking:
    endpoint: str
    base_bytes: int
    workspace_bytes: int
    binding: ShareBinding | None
    credited_generation: int = 0
    holds: list[dict[str, object]] = field(default_factory=list)


class DormantShareCoordinator:
    def __init__(self, *, budget_bytes: int, safety_bytes: int = 256 * 1024**2, pool: str = "host-ram",
                 hold_timeout_s: float = 900.0, on_event: Callable[[Mapping[str, object]], None] | None = None) -> None:
        if type(budget_bytes) is not int or budget_bytes <= 0 or type(safety_bytes) is not int or safety_bytes < 0:
            raise DormantShareAdmissionError("dormant share budget is invalid")
        if safety_bytes >= budget_bytes:
            raise DormantShareAdmissionError("dormant share safety reserve exceeds the budget")
        if type(hold_timeout_s) not in (int, float) or hold_timeout_s <= 0:
            raise DormantShareAdmissionError("dormant share hold timeout is invalid")
        self._pool = pool
        self._budget = budget_bytes
        self._snapshot = RuntimePlacementSnapshot("dormant-share-budget", 0, 10**15,
                                                  {pool: DeviceMemoryCapacity(pool, budget_bytes, 0, safety_bytes)})
        self._ledger = RuntimeMemoryLedger()
        self._accountant = DecodeReleaseAccountant(self._ledger, host_pool=pool)
        self._servers: dict[str, _ServerBooking] = {}
        self._condition = threading.Condition()
        self._hold_timeout_s = float(hold_timeout_s)
        self._on_event = on_event
        self._events: list[dict[str, object]] = []

    # ---- bookkeeping ----------------------------------------------------------------------------------
    def _event(self, kind: str, **fields: object) -> None:
        row = {"kind": kind, "time_ns": time.monotonic_ns(),
               "ledger_reserved_bytes": self._ledger.snapshot()["by_resource_bytes"].get(self._pool, 0), **fields}
        self._events.append(row)
        if self._on_event is not None:
            self._on_event(row)

    def events(self) -> tuple[dict[str, object], ...]:
        with self._condition:
            return tuple(dict(row) for row in self._events)

    def reserved_bytes(self) -> int:
        return self._ledger.snapshot()["by_resource_bytes"].get(self._pool, 0)

    def headroom_bytes(self) -> int:
        return self._accountant.decode_phase_headroom_bytes(self._snapshot)

    def booking(self, endpoint: str) -> _ServerBooking | None:
        with self._condition:
            return self._servers.get(endpoint)

    def state(self, endpoint: str) -> str | None:
        return self._accountant.state(endpoint)

    def book_server(self, endpoint: str, *, base_bytes: int, workspace_bytes: int = 0,
                    binding: ShareBinding | None = None, strict: bool = True) -> None:
        """A server is READY: charge its resident footprint (minus the share) and book the share itself.
        With ``strict=False`` an over-subscribed budget is recorded and the server is booked with what fits
        (no share), instead of raising after the scheduler has already launched it."""
        if type(base_bytes) is not int or base_bytes <= 0 or type(workspace_bytes) is not int or workspace_bytes < 0:
            raise DormantShareAdmissionError("server booking bytes are invalid")
        if binding is not None and binding.endpoint != endpoint:
            raise DormantShareAdmissionError("share binding endpoint differs from the server")
        with self._condition:
            if endpoint in self._servers:
                raise DormantShareAdmissionError("server is already booked")
            checkpoint = self._ledger.checkpoint()
            try:
                demands = [RuntimeMemoryDemand(f"{endpoint}:base", self._pool, "resident-weights-and-anon", base_bytes, 0, "request")]
                if workspace_bytes:
                    demands.append(RuntimeMemoryDemand(f"{endpoint}:workspace", self._pool, "prefill-workspace", workspace_bytes, 0, "request"))
                self._ledger.reserve(f"server:{endpoint}", demands, self._snapshot)
                if binding is not None:
                    self._accountant.reserve_share(endpoint, binding, self._snapshot)
            except (RuntimeResourceError, DecodeSplitSelectionError) as error:
                self._ledger.restore(checkpoint)
                if strict:
                    self._event("server_booking_refused", endpoint=endpoint, error=repr(error))
                    raise DormantShareAdmissionError(f"host budget cannot hold this server: {error}") from error
                # the scheduler already launched this server: record the over-subscription and book what fits,
                # so later prompt gates see a ledger that is at least as full as reality allows
                fitting = max(0, self.headroom_bytes())
                self._event("server_booking_over_budget", endpoint=endpoint, base_bytes=base_bytes,
                            workspace_bytes=workspace_bytes, booked_base_bytes=fitting, error=repr(error))
                if fitting > 0:
                    self._ledger.reserve(f"server:{endpoint}", (RuntimeMemoryDemand(
                        f"{endpoint}:base", self._pool, "resident-weights-and-anon-partial", fitting, 0, "request"),), self._snapshot)
                self._servers[endpoint] = _ServerBooking(endpoint, fitting, 0, None)
                self._condition.notify_all()
                return
            self._servers[endpoint] = _ServerBooking(endpoint, base_bytes, workspace_bytes, binding)
            self._event("server_booked", endpoint=endpoint, base_bytes=base_bytes, workspace_bytes=workspace_bytes,
                        expected_release_bytes=None if binding is None else binding.expected_release_bytes)
            self._condition.notify_all()

    def forget_server(self, endpoint: str) -> None:
        with self._condition:
            booking = self._servers.pop(endpoint, None)
            if booking is None:
                return
            if booking.binding is not None:
                self._accountant.forget(endpoint)
            self._ledger.release_owner(f"server:{endpoint}")
            self._event("server_forgotten", endpoint=endpoint)
            self._condition.notify_all()

    def reserve_growth(self, owner_id: str, growth_bytes: int) -> None:
        """Decode-phase consumption by a tenant (another request's KV growth, a test consumer)."""
        with self._condition:
            try:
                self._accountant.reserve_decode_growth(owner_id, growth_bytes, self._snapshot)
            except RuntimeResourceError as error:
                self._event("growth_refused", owner=owner_id, growth_bytes=growth_bytes, error=repr(error))
                raise DormantShareAdmissionError(f"host budget cannot hold the growth: {error}") from error
            self._event("growth_reserved", owner=owner_id, growth_bytes=growth_bytes)

    def release_growth(self, owner_id: str) -> None:
        with self._condition:
            self._accountant.release_growth(owner_id)
            self._event("growth_released", owner=owner_id)
            self._condition.notify_all()

    # ---- runtime hooks --------------------------------------------------------------------------------
    def on_control_ack(self, endpoint: str, request_id: str, runtime_stats: Mapping[str, object] | None) -> int | None:
        """Credit a new release generation reported by the server's control acknowledgement."""
        if not isinstance(runtime_stats, Mapping):
            return None
        generation = runtime_stats.get("dormant_release_generation")
        released = runtime_stats.get("dormant_released_bytes")
        if type(generation) is not int or generation <= 0 or type(released) is not int or released <= 0:
            return None
        with self._condition:
            booking = self._servers.get(endpoint)
            if booking is None or booking.binding is None or generation <= booking.credited_generation:
                return None
            proof = RuntimeHostShareReleaseProof(
                phase="decode", layer_mask=int(runtime_stats.get("dormant_layer_mask", 0)),
                host_columns=int(runtime_stats.get("dormant_host_columns", 0)), released_bytes=released,
                ranges=0, elapsed_us=int(runtime_stats.get("dormant_release_elapsed_us", 0)))
            try:
                credited = self._accountant.enter_decode(endpoint, proof, self._snapshot, endpoint=endpoint,
                                                         release_generation=generation)
            except (DecodeSplitSelectionError, RuntimeResourceError) as error:
                # accounting refusal is recorded, never turned into a control failure for the running request
                self._event("release_credit_refused", endpoint=endpoint, request_id=request_id, generation=generation,
                            layer_mask=proof.layer_mask, host_columns=proof.host_columns, released_bytes=released, error=repr(error))
                return None
            booking.credited_generation = generation
            self._event("release_credited", endpoint=endpoint, request_id=request_id, generation=generation,
                        credited_bytes=credited, released_bytes=released, headroom_bytes=self.headroom_bytes())
            self._condition.notify_all()
            return credited

    def before_prompt(self, endpoint: str, request_id: str, *, deadline_s: float | None = None) -> dict[str, object] | None:
        """Wait until the server's share can be resident again, then re-reserve it. Returns the hold record
        (None when the server has no share or it was already resident)."""
        started = time.monotonic()
        limit = self._hold_timeout_s if deadline_s is None else float(deadline_s)
        with self._condition:
            booking = self._servers.get(endpoint)
            if booking is None or booking.binding is None:
                return None
            if self._accountant.state(endpoint) == "prefill-resident":
                return None
            attempts = 0
            while True:
                attempts += 1
                try:
                    self._accountant.restore_before_prompt(endpoint, self._snapshot)
                    break
                except RuntimeResourceError as error:
                    waited = time.monotonic() - started
                    if attempts == 1:
                        self._event("prompt_held", endpoint=endpoint, request_id=request_id, error=repr(error),
                                    headroom_bytes=self.headroom_bytes())
                    if waited >= limit:
                        self._event("prompt_hold_timeout", endpoint=endpoint, request_id=request_id, waited_s=waited)
                        raise DormantShareAdmissionError(
                            f"host budget did not return the released share within {limit:.0f} s") from error
                    self._condition.wait(timeout=min(1.0, limit - waited))
            hold = {"endpoint": endpoint, "request_id": request_id, "waited_s": time.monotonic() - started, "attempts": attempts}
            booking.holds.append(hold)
            self._event("prompt_admitted", **hold)
            return hold

    def to_json(self) -> dict[str, object]:
        with self._condition:
            return {"budget_bytes": self._budget, "pool": self._pool,
                    "servers": {ep: {"base_bytes": b.base_bytes, "workspace_bytes": b.workspace_bytes,
                                     "credited_generation": b.credited_generation, "holds": list(b.holds),
                                     "share": None if b.binding is None else {
                                         "layer_mask": b.binding.layer_mask, "host_columns": b.binding.host_columns,
                                         "expected_release_bytes": b.binding.expected_release_bytes,
                                         "state": self._accountant.state(ep)}}
                                for ep, b in sorted(self._servers.items())},
                    "accountant": self._accountant.to_json(), "events": list(self._events)}


__all__ = ["DormantShareAdmissionError", "DormantShareCoordinator"]
