#!/usr/bin/env python3
"""Shared-slot and serialized-client support for dense physical replay."""

from __future__ import annotations

import threading
from typing import Any


class SlotError(RuntimeError):
    pass


class SequenceSlotPool:
    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("slot capacity must be a positive integer")
        self.capacity = capacity
        self._free = list(range(capacity))
        self._leased: dict[int, int] = {}
        self._lock = threading.Lock()

    def try_acquire(self, request_id: int) -> int | None:
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id <= 0:
            raise ValueError("request_id must be a positive integer")
        with self._lock:
            if request_id in self._leased.values():
                raise SlotError("request already owns a slot")
            if not self._free:
                return None
            seq_id = self._free.pop(0)
            self._leased[seq_id] = request_id
            return seq_id

    def release(self, seq_id: int, request_id: int) -> None:
        with self._lock:
            if self._leased.get(seq_id) != request_id:
                raise SlotError("slot release ownership mismatch")
            del self._leased[seq_id]
            self._free.append(seq_id)
            self._free.sort()

    def available(self) -> int:
        with self._lock:
            return len(self._free)

    def leased(self) -> dict[int, int]:
        with self._lock:
            return dict(self._leased)


class SerializedStageClient:
    """Serialize request-response transactions on one StageNet socket."""

    def __init__(self, client: Any):
        self._client = client
        self._lock = threading.Lock()

    def hello(self):
        with self._lock:
            return self._client.hello()

    def batch(self, rows):
        with self._lock:
            return self._client.batch(rows)

    def remove(self, seq_id: int, request_id: int, route_epoch: int):
        with self._lock:
            return self._client.remove(seq_id, request_id, route_epoch)

    def status(self):
        with self._lock:
            return self._client.status()

    def drain(self):
        with self._lock:
            return self._client.drain()

    def stop(self) -> None:
        with self._lock:
            self._client.stop()

    def detach(self) -> None:
        with self._lock:
            self._client.detach()

    def close(self) -> None:
        with self._lock:
            self._client.close()
