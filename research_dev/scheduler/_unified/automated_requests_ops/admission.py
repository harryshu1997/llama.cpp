"""Static request-shape admission against the registered catalog.

A request whose prompt plus output cannot fit the preallocated context of
any coordinator serving its model (or of the mandatory desktop control) is
permanently unsupported by this rig: no calendar slot will ever exist for
it. Such shapes are rejected at submission with an exact terminal reason
instead of failing route generation and aborting the campaign. Temporary
resource contention is never judged here; it still queues through the
resource calendar.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..._internal.lifecycle import RequestShapeUnsupportedError
from ..._internal.policy import Request

REQUEST_EXCEEDS_CONTEXT_CAPACITY = "REQUEST_EXCEEDS_CONTEXT_CAPACITY"
REQUEST_EXCEEDS_DESKTOP_CONTROL_CONTEXT = (
    "REQUEST_EXCEEDS_DESKTOP_CONTROL_CONTEXT"
)


@dataclass(frozen=True)
class RequestShapeSupport:
    request_id: str
    model_id: str
    tokens: int
    supported: bool
    reason: str | None
    desktop_control_executor_id: str | None
    desktop_control_capacity_tokens: int | None
    capacity_tokens_by_executor: Mapping[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "capacity_tokens_by_executor": dict(
                sorted(self.capacity_tokens_by_executor.items())
            ),
            "desktop_control_capacity_tokens": (
                self.desktop_control_capacity_tokens
            ),
            "desktop_control_executor_id": self.desktop_control_executor_id,
            "model_id": self.model_id,
            "reason": self.reason,
            "request_id": self.request_id,
            "supported": self.supported,
            "tokens": self.tokens,
        }

    def details(self) -> dict[str, object]:
        return {
            "desktop_control_capacity_tokens": (
                self.desktop_control_capacity_tokens
            ),
            "desktop_control_executor_id": self.desktop_control_executor_id,
            "maximum_capacity_tokens": max(
                self.capacity_tokens_by_executor.values(), default=None
            ),
            "model_id": self.model_id,
            "tokens": self.tokens,
        }


def request_shape_support(
    controller, request: Request, model_id: str
) -> RequestShapeSupport:
    """Judge whether the request's token shape can ever be preallocated."""

    request.validate()
    manifest = controller.runtime_model_manifest(model_id)
    catalog = getattr(controller, "_runtime_capabilities", None)
    tokens = request.input_tokens + request.output_tokens
    capacities: dict[str, int] = {}
    if catalog is not None:
        for executor in catalog.composite_executors:
            parameters = executor.adapter_parameters
            if (
                executor.artifact_sha256 != manifest.artifact_sha256
                or parameters.get("request_memory_mode") != "preallocated"
            ):
                continue
            quantum = parameters.get("context_token_quantum")
            context_size = parameters.get("context_size")
            if type(quantum) is not int or quantum <= 0:
                continue
            resource = catalog.resources.get(
                str(parameters.get("context_resource_id"))
            )
            slots = (
                resource.capacity if resource is not None
                else (context_size // quantum
                      if type(context_size) is int else None)
            )
            if slots is None:
                continue
            capacities[executor.executor_id] = slots * quantum
    control = (
        None if catalog is None
        else catalog.desktop_control_by_artifact.get(manifest.artifact_sha256)
    )
    control_executor_id = None if control is None else control.executor_id
    control_capacity = (
        None if control_executor_id is None
        else capacities.get(control_executor_id)
    )
    reason = None
    # A request fits a coordinator when ceil(tokens / quantum) slots are
    # at most the resource capacity, i.e. tokens <= capacity * quantum.
    fits = {
        executor_id: tokens <= capacity
        for executor_id, capacity in capacities.items()
    }
    if capacities and not any(fits.values()):
        reason = REQUEST_EXCEEDS_CONTEXT_CAPACITY
    elif (
        control_capacity is not None
        and not fits.get(control_executor_id, True)
    ):
        reason = REQUEST_EXCEEDS_DESKTOP_CONTROL_CONTEXT
    return RequestShapeSupport(
        request_id=request.request_id,
        model_id=model_id,
        tokens=tokens,
        supported=reason is None,
        reason=reason,
        desktop_control_executor_id=control_executor_id,
        desktop_control_capacity_tokens=control_capacity,
        capacity_tokens_by_executor=MappingProxyType(dict(capacities)),
    )



def require_supported_request_shape(
    controller, request: Request, model_id: str
) -> RequestShapeSupport:
    support = request_shape_support(controller, request, model_id)
    if not support.supported:
        raise RequestShapeUnsupportedError(support.reason, support.details())
    return support
