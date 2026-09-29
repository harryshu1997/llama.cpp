"""Candidate construction: local, split, offload, decode split/offload, and route nodes.

Mixin of ``AutomatedRouteCompiler``; methods moved here verbatim.
"""

from __future__ import annotations

from dataclasses import replace
from ..model_manifest import ModelManifest, ModelRequestWork
from ..placement import (
    ComputeStep,
    ExecutionBranch,
    OperatorCandidate,
    OperatorNode,
    ResidentAllocation,
    TransferStep,
)
from ..runtime_capabilities import (
    desktop_control_placement_payload,
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeExecutorCapability,
)
from ..types import canonical_sha256
from .remote_resident import RouteRemoteResidentMixin
from .common import (
    RouteGenerationError,
    _ceil_div,
    _Pattern,
)


class RouteCandidateMixin:
    """Candidate construction: local, split, offload, decode split/offload, and route nodes."""

    @staticmethod
    def _fraction(value: int, fraction_ppm: int) -> tuple[int, int]:
        helper = _ceil_div(value * fraction_ppm, 1_000_000)
        helper = min(value, helper)
        return value - helper, helper

    def _residency_executor_id(
        self, pattern: _Pattern, device_id: str
    ) -> str:
        if pattern.coordinator_executor_id is not None:
            return pattern.coordinator_executor_id
        return self.catalog.executor_by_device[device_id].executor_id

    def _matching_residency(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        device_id: str,
    ):
        residency = snapshot.residency_for(
            manifest.model_id, manifest.artifact_sha256, device_id
        )
        if residency is None:
            return None
        expected = self._residency_executor_id(pattern, device_id)
        if residency.executor_id not in {None, expected}:
            return None
        return residency

    def _local_candidate(
        self,
        operator_id: str,
        kind: str,
        device_id: str,
        work,
        tensors,
        request_work: ModelRequestWork,
    ) -> OperatorCandidate:
        capability = self.catalog.executor_by_device[device_id]
        kernel_profile_id, selector = capability.kernel_profile_for(
            kind,
            input_tokens=request_work.input_tokens,
            output_tokens=request_work.output_tokens,
            compute_ops=work.compute_ops,
            memory_bytes=work.memory_bytes,
        )
        kernel = self.catalog.placement_profile.kernels[kernel_profile_id]
        measured = (
            capability.maturity == "QUALIFIED"
            and kernel.status == "measured"
            and (selector is None or selector.maturity == "QUALIFIED")
        )
        evidence = set(capability.evidence_ids)
        if selector is not None:
            evidence.update(selector.evidence_ids)
        allocations = tuple(
            ResidentAllocation(
                allocation_id=f"weight:{tensor.tensor_id}:{device_id}",
                device_id=device_id,
                bytes=tensor.nbytes,
            )
            for tensor in tensors
        )
        return OperatorCandidate(
            candidate_id=f"local:{operator_id}:{device_id}",
            operator_id=operator_id,
            input_device=device_id,
            output_device=device_id,
            branches=(ExecutionBranch(
                branch_id="local",
                steps=(ComputeStep(
                    step_id=f"compute:{operator_id}:{device_id}",
                    kernel_profile_id=kernel_profile_id,
                    invocations=1,
                    compute_ops=work.compute_ops,
                    memory_bytes=work.memory_bytes,
                ),),
            ),),
            resident_allocations=allocations,
            workspace_bytes={device_id: (
                work.workspace_bytes
                + capability.workspace_bytes_per_token
            )},
            status="measured" if measured else "estimated",
            placement_verified=True,
            evidence_ids=tuple(sorted(evidence)),
        )

    @staticmethod
    def _request_phase(
        request_work: ModelRequestWork, phase: str
    ):
        row = next(
            (value for value in request_work.phases if value.phase == phase),
            None,
        )
        if row is None:
            raise RouteGenerationError(
                "phase-aware placement requires request phase work"
            )
        return row

    def _decode_split_candidate(
        self,
        operator_id: str,
        kind: str,
        base_id: str,
        helper_id: str,
        fraction_ppm: int,
        axis: str,
        tensors,
        request_work: ModelRequestWork,
        full_width_output: bool,
    ) -> OperatorCandidate:
        prefill = self._request_phase(request_work, "prefill")
        decode = self._request_phase(request_work, "decode")
        prefill_work = prefill.by_operator_id[operator_id]
        decode_work = decode.by_operator_id[operator_id]
        base = self.catalog.executor_by_device[base_id]
        helper = self.catalog.executor_by_device[helper_id]
        base_ops, helper_ops = self._fraction(
            decode_work.compute_ops, fraction_ppm
        )
        base_memory, helper_memory = self._fraction(
            decode_work.memory_bytes, fraction_ppm
        )
        prefill_profile_id, prefill_selector = base.kernel_profile_for(
            kind,
            input_tokens=request_work.input_tokens,
            output_tokens=1,
            compute_ops=prefill_work.compute_ops,
            memory_bytes=prefill_work.memory_bytes,
        )
        base_profile_id, base_selector = base.kernel_profile_for(
            kind,
            input_tokens=1,
            output_tokens=request_work.output_tokens,
            compute_ops=base_ops,
            memory_bytes=base_memory,
        )
        helper_profile_id, helper_selector = helper.kernel_profile_for(
            kind,
            input_tokens=1,
            output_tokens=request_work.output_tokens,
            compute_ops=helper_ops,
            memory_bytes=helper_memory,
        )
        input_bytes = max(
            1, decode_work.activation_bytes // decode.tokens
        )
        output_bytes = max(1, decode_work.output_bytes // decode.tokens)
        if not full_width_output:
            output_bytes = max(
                1, self._fraction(output_bytes, fraction_ppm)[1]
            )
        allocations = []
        for tensor in tensors:
            allocations.append(ResidentAllocation(
                allocation_id=f"weight:{tensor.tensor_id}:{base_id}",
                device_id=base_id,
                bytes=tensor.nbytes,
            ))
            helper_bytes = self._fraction(
                tensor.nbytes, fraction_ppm
            )[1]
            if helper_bytes:
                allocations.append(ResidentAllocation(
                    allocation_id=f"weight:{tensor.tensor_id}:{helper_id}",
                    device_id=helper_id,
                    bytes=helper_bytes,
                ))
        selectors = (
            (base, prefill_profile_id, prefill_selector),
            (base, base_profile_id, base_selector),
            (helper, helper_profile_id, helper_selector),
        )
        measured = all(
            capability.maturity == "QUALIFIED"
            and self.catalog.placement_profile.kernels[
                profile_id
            ].status == "measured"
            and (selector is None or selector.maturity == "QUALIFIED")
            for capability, profile_id, selector in selectors
        )
        evidence = set(base.evidence_ids + helper.evidence_ids)
        for _capability, _profile_id, selector in selectors:
            if selector is not None:
                evidence.update(selector.evidence_ids)
        return OperatorCandidate(
            candidate_id=(
                f"split:{operator_id}:{base_id}+{helper_id}:"
                f"{axis}:{fraction_ppm}:decode"
            ),
            operator_id=operator_id,
            input_device=base_id,
            output_device=base_id,
            branches=(
                ExecutionBranch(
                    branch_id="base-decode",
                    steps=(ComputeStep(
                        step_id=f"decode-compute:{operator_id}:{base_id}",
                        kernel_profile_id=base_profile_id,
                        invocations=1,
                        compute_ops=base_ops,
                        memory_bytes=base_memory,
                    ),),
                ),
                ExecutionBranch(
                    branch_id="helper-decode",
                    steps=(
                        TransferStep(
                            step_id=(
                                f"decode-input:{operator_id}:{helper_id}"
                            ),
                            source_device=base_id,
                            target_device=helper_id,
                            bytes=input_bytes,
                            invocations=request_work.output_tokens,
                        ),
                        ComputeStep(
                            step_id=(
                                f"decode-compute:{operator_id}:{helper_id}"
                            ),
                            kernel_profile_id=helper_profile_id,
                            invocations=1,
                            compute_ops=helper_ops,
                            memory_bytes=helper_memory,
                        ),
                        TransferStep(
                            step_id=(
                                f"decode-output:{operator_id}:{helper_id}"
                            ),
                            source_device=helper_id,
                            target_device=base_id,
                            bytes=output_bytes,
                            invocations=request_work.output_tokens,
                        ),
                    ),
                ),
            ),
            tail_steps=(ComputeStep(
                step_id=f"prefill-compute:{operator_id}:{base_id}",
                kernel_profile_id=prefill_profile_id,
                invocations=1,
                compute_ops=prefill_work.compute_ops,
                memory_bytes=prefill_work.memory_bytes,
            ),),
            resident_allocations=tuple(allocations),
            workspace_bytes={
                base_id: max(
                    prefill_work.workspace_bytes,
                    decode_work.workspace_bytes // decode.tokens,
                ),
                helper_id: max(
                    1, decode_work.workspace_bytes // decode.tokens
                ),
            },
            status="measured" if measured else "estimated",
            placement_verified=True,
            evidence_ids=tuple(sorted(evidence)),
            split_axis=axis,
            split_amount=fraction_ppm,
            split_total=1_000_000,
        )

    def _decode_offload_candidate(
        self,
        operator_id: str,
        kind: str,
        base_id: str,
        helper_id: str,
        tensors,
        request_work: ModelRequestWork,
    ) -> OperatorCandidate:
        prefill = self._request_phase(request_work, "prefill")
        decode = self._request_phase(request_work, "decode")
        prefill_work = prefill.by_operator_id[operator_id]
        decode_work = decode.by_operator_id[operator_id]
        base = self.catalog.executor_by_device[base_id]
        helper = self.catalog.executor_by_device[helper_id]
        prefill_profile_id, prefill_selector = base.kernel_profile_for(
            kind,
            input_tokens=request_work.input_tokens,
            output_tokens=1,
            compute_ops=prefill_work.compute_ops,
            memory_bytes=prefill_work.memory_bytes,
        )
        helper_profile_id, helper_selector = helper.kernel_profile_for(
            kind,
            input_tokens=1,
            output_tokens=request_work.output_tokens,
            compute_ops=decode_work.compute_ops,
            memory_bytes=decode_work.memory_bytes,
        )
        selectors = (
            (base, prefill_profile_id, prefill_selector),
            (helper, helper_profile_id, helper_selector),
        )
        measured = all(
            capability.maturity == "QUALIFIED"
            and self.catalog.placement_profile.kernels[
                profile_id
            ].status == "measured"
            and (selector is None or selector.maturity == "QUALIFIED")
            for capability, profile_id, selector in selectors
        )
        evidence = set(base.evidence_ids + helper.evidence_ids)
        for _capability, _profile_id, selector in selectors:
            if selector is not None:
                evidence.update(selector.evidence_ids)
        return OperatorCandidate(
            candidate_id=(
                f"offload:{operator_id}:{base_id}+{helper_id}:decode"
            ),
            operator_id=operator_id,
            input_device=base_id,
            output_device=base_id,
            branches=(ExecutionBranch(
                branch_id="helper-decode",
                steps=(
                    TransferStep(
                        step_id=f"decode-input:{operator_id}:{helper_id}",
                        source_device=base_id,
                        target_device=helper_id,
                        bytes=max(
                            1,
                            decode_work.activation_bytes // decode.tokens,
                        ),
                        invocations=request_work.output_tokens,
                    ),
                    ComputeStep(
                        step_id=(
                            f"decode-compute:{operator_id}:{helper_id}"
                        ),
                        kernel_profile_id=helper_profile_id,
                        invocations=1,
                        compute_ops=decode_work.compute_ops,
                        memory_bytes=decode_work.memory_bytes,
                    ),
                    TransferStep(
                        step_id=f"decode-output:{operator_id}:{helper_id}",
                        source_device=helper_id,
                        target_device=base_id,
                        bytes=max(
                            1, decode_work.output_bytes // decode.tokens
                        ),
                        invocations=request_work.output_tokens,
                    ),
                ),
            ),),
            tail_steps=(ComputeStep(
                step_id=f"prefill-compute:{operator_id}:{base_id}",
                kernel_profile_id=prefill_profile_id,
                invocations=1,
                compute_ops=prefill_work.compute_ops,
                memory_bytes=prefill_work.memory_bytes,
            ),),
            resident_allocations=tuple(
                ResidentAllocation(
                    allocation_id=f"weight:{tensor.tensor_id}:{device_id}",
                    device_id=device_id,
                    bytes=tensor.nbytes,
                )
                for tensor in tensors
                for device_id in (base_id, helper_id)
            ),
            workspace_bytes={
                base_id: prefill_work.workspace_bytes,
                helper_id: max(
                    1, decode_work.workspace_bytes // decode.tokens
                ),
            },
            status="measured" if measured else "estimated",
            placement_verified=True,
            evidence_ids=tuple(sorted(evidence)),
        )

    def _split_candidate(
        self,
        operator_id: str,
        kind: str,
        base_id: str,
        helper_id: str,
        fraction_ppm: int,
        axis: str,
        work,
        tensors,
        request_work: ModelRequestWork,
        full_width_output: bool,
        maximum_transfer_tokens: int | None,
    ) -> OperatorCandidate:
        base = self.catalog.executor_by_device[base_id]
        helper = self.catalog.executor_by_device[helper_id]
        base_ops, helper_ops = self._fraction(work.compute_ops, fraction_ppm)
        base_memory, helper_memory = self._fraction(
            work.memory_bytes, fraction_ppm
        )
        base_profile_id, base_selector = base.kernel_profile_for(
            kind,
            input_tokens=request_work.input_tokens,
            output_tokens=request_work.output_tokens,
            compute_ops=base_ops,
            memory_bytes=base_memory,
        )
        helper_profile_id, helper_selector = helper.kernel_profile_for(
            kind,
            input_tokens=request_work.input_tokens,
            output_tokens=request_work.output_tokens,
            compute_ops=helper_ops,
            memory_bytes=helper_memory,
        )
        allocations = []
        for tensor in tensors:
            base_bytes, helper_bytes = self._fraction(
                tensor.nbytes, fraction_ppm
            )
            if base_bytes:
                allocations.append(ResidentAllocation(
                    allocation_id=f"weight:{tensor.tensor_id}:{base_id}",
                    device_id=base_id,
                    bytes=base_bytes,
                ))
            if helper_bytes:
                allocations.append(ResidentAllocation(
                    allocation_id=f"weight:{tensor.tensor_id}:{helper_id}",
                    device_id=helper_id,
                    bytes=helper_bytes,
                ))
        total_tokens = (
            request_work.input_tokens + request_work.output_tokens
        )
        if (
            work.activation_bytes % total_tokens
            or work.output_bytes % total_tokens
        ):
            raise RouteGenerationError(
                "operator activation work is not token exact"
            )
        input_bytes_per_token = work.activation_bytes // total_tokens
        output_bytes_per_token = work.output_bytes // total_tokens
        if not full_width_output:
            output_bytes_per_token = max(
                1,
                self._fraction(output_bytes_per_token, fraction_ppm)[1],
            )
        prefill_tokens = request_work.input_tokens
        prefill_invocations = 1
        if maximum_transfer_tokens is not None:
            if maximum_transfer_tokens <= 0:
                raise RouteGenerationError(
                    "maximum transfer tokens must be positive"
                )
            prefill_tokens = min(prefill_tokens, maximum_transfer_tokens)
            prefill_invocations = _ceil_div(
                request_work.input_tokens, maximum_transfer_tokens
            )
        prefill_input_bytes = max(1, input_bytes_per_token * prefill_tokens)
        prefill_output_bytes = max(1, output_bytes_per_token * prefill_tokens)
        decode_input_bytes = max(1, input_bytes_per_token)
        decode_output_bytes = max(1, output_bytes_per_token)
        measured = (
            base.maturity == "QUALIFIED"
            and helper.maturity == "QUALIFIED"
            and self.catalog.placement_profile.kernels[
                base_profile_id
            ].status == "measured"
            and self.catalog.placement_profile.kernels[
                helper_profile_id
            ].status == "measured"
            and (
                base_selector is None
                or base_selector.maturity == "QUALIFIED"
            )
            and (
                helper_selector is None
                or helper_selector.maturity == "QUALIFIED"
            )
        )
        evidence = set(base.evidence_ids + helper.evidence_ids)
        for selector in (base_selector, helper_selector):
            if selector is not None:
                evidence.update(selector.evidence_ids)
        return OperatorCandidate(
            candidate_id=(
                f"split:{operator_id}:{base_id}+{helper_id}:{axis}:{fraction_ppm}"
            ),
            operator_id=operator_id,
            input_device=base_id,
            output_device=base_id,
            branches=(
                ExecutionBranch(
                    branch_id="base",
                    steps=(ComputeStep(
                        step_id=f"compute:{operator_id}:{base_id}",
                        kernel_profile_id=base_profile_id,
                        invocations=1,
                        compute_ops=base_ops,
                        memory_bytes=base_memory,
                    ),),
                ),
                ExecutionBranch(
                    branch_id="helper",
                    steps=(
                        TransferStep(
                            step_id=f"prefill-input:{operator_id}:{helper_id}",
                            source_device=base_id,
                            target_device=helper_id,
                            bytes=prefill_input_bytes,
                            invocations=prefill_invocations,
                        ),
                        TransferStep(
                            step_id=f"decode-input:{operator_id}:{helper_id}",
                            source_device=base_id,
                            target_device=helper_id,
                            bytes=decode_input_bytes,
                            invocations=request_work.output_tokens,
                        ),
                        ComputeStep(
                            step_id=f"compute:{operator_id}:{helper_id}",
                            kernel_profile_id=helper_profile_id,
                            invocations=1,
                            compute_ops=helper_ops,
                            memory_bytes=helper_memory,
                        ),
                        TransferStep(
                            step_id=f"prefill-output:{operator_id}:{helper_id}",
                            source_device=helper_id,
                            target_device=base_id,
                            bytes=prefill_output_bytes,
                            invocations=prefill_invocations,
                        ),
                        TransferStep(
                            step_id=f"decode-output:{operator_id}:{helper_id}",
                            source_device=helper_id,
                            target_device=base_id,
                            bytes=decode_output_bytes,
                            invocations=request_work.output_tokens,
                        ),
                    ),
                ),
            ),
            resident_allocations=tuple(allocations),
            workspace_bytes={
                base_id: work.workspace_bytes,
                helper_id: work.workspace_bytes,
            },
            status="measured" if measured else "estimated",
            placement_verified=True,
            evidence_ids=tuple(sorted(evidence)),
            split_axis=axis,
            split_amount=fraction_ppm,
            split_total=1_000_000,
        )

    def _offload_candidate(
        self,
        operator_id: str,
        kind: str,
        base_id: str,
        helper_id: str,
        work,
        tensors,
        request_work: ModelRequestWork,
        maximum_transfer_tokens: int | None,
    ) -> OperatorCandidate:
        helper = self.catalog.executor_by_device[helper_id]
        helper_profile_id, helper_selector = helper.kernel_profile_for(
            kind,
            input_tokens=request_work.input_tokens,
            output_tokens=request_work.output_tokens,
            compute_ops=work.compute_ops,
            memory_bytes=work.memory_bytes,
        )
        total_tokens = request_work.input_tokens + request_work.output_tokens
        if (
            work.activation_bytes % total_tokens
            or work.output_bytes % total_tokens
        ):
            raise RouteGenerationError(
                "operator activation work is not token exact"
            )
        input_bytes_per_token = work.activation_bytes // total_tokens
        output_bytes_per_token = work.output_bytes // total_tokens
        prefill_tokens = request_work.input_tokens
        prefill_invocations = 1
        if maximum_transfer_tokens is not None:
            if maximum_transfer_tokens <= 0:
                raise RouteGenerationError(
                    "maximum transfer tokens must be positive"
                )
            prefill_tokens = min(prefill_tokens, maximum_transfer_tokens)
            prefill_invocations = _ceil_div(
                request_work.input_tokens, maximum_transfer_tokens
            )
        measured = (
            helper.maturity == "QUALIFIED"
            and self.catalog.placement_profile.kernels[
                helper_profile_id
            ].status == "measured"
            and (
                helper_selector is None
                or helper_selector.maturity == "QUALIFIED"
            )
        )
        evidence = set(helper.evidence_ids)
        if helper_selector is not None:
            evidence.update(helper_selector.evidence_ids)
        return OperatorCandidate(
            candidate_id=(
                f"offload:{operator_id}:{base_id}+{helper_id}"
            ),
            operator_id=operator_id,
            input_device=base_id,
            output_device=base_id,
            branches=(ExecutionBranch(
                branch_id="helper",
                steps=(
                    TransferStep(
                        step_id=f"prefill-input:{operator_id}:{helper_id}",
                        source_device=base_id,
                        target_device=helper_id,
                        bytes=max(
                            1, input_bytes_per_token * prefill_tokens
                        ),
                        invocations=prefill_invocations,
                    ),
                    TransferStep(
                        step_id=f"decode-input:{operator_id}:{helper_id}",
                        source_device=base_id,
                        target_device=helper_id,
                        bytes=max(1, input_bytes_per_token),
                        invocations=request_work.output_tokens,
                    ),
                    ComputeStep(
                        step_id=f"compute:{operator_id}:{helper_id}",
                        kernel_profile_id=helper_profile_id,
                        invocations=1,
                        compute_ops=work.compute_ops,
                        memory_bytes=work.memory_bytes,
                    ),
                    TransferStep(
                        step_id=f"prefill-output:{operator_id}:{helper_id}",
                        source_device=helper_id,
                        target_device=base_id,
                        bytes=max(
                            1, output_bytes_per_token * prefill_tokens
                        ),
                        invocations=prefill_invocations,
                    ),
                    TransferStep(
                        step_id=f"decode-output:{operator_id}:{helper_id}",
                        source_device=helper_id,
                        target_device=base_id,
                        bytes=max(1, output_bytes_per_token),
                        invocations=request_work.output_tokens,
                    ),
                ),
            ),),
            resident_allocations=tuple(
                ResidentAllocation(
                    allocation_id=f"weight:{tensor.tensor_id}:{helper_id}",
                    device_id=helper_id,
                    bytes=tensor.nbytes,
                )
                for tensor in tensors
            ),
            workspace_bytes={helper_id: work.workspace_bytes},
            status="measured" if measured else "estimated",
            placement_verified=True,
            evidence_ids=tuple(sorted(evidence)),
        )

    def _nodes(
        self,
        manifest: ModelManifest,
        work: ModelRequestWork,
        pattern: _Pattern,
    ) -> tuple[OperatorNode, ...]:
        work_by_id = work.by_operator_id
        tensors = manifest.tensor_by_id
        rows = []
        coordinator = self._coordinator(pattern)
        full_width_output = (
            isinstance(coordinator, RuntimeCompositeExecutorCapability)
            and coordinator.adapter_parameters.get(
                "split_output_width"
            ) == "full"
        )
        maximum_transfer_tokens = (
            coordinator.adapter_parameters.get("ubatch_size")
            if isinstance(coordinator, RuntimeCompositeExecutorCapability)
            else None
        )
        if maximum_transfer_tokens is not None and type(
            maximum_transfer_tokens
        ) is not int:
            raise RouteGenerationError(
                "composite ubatch size must be an integer"
            )
        baseline_by_id = {}
        if (
            isinstance(coordinator, RuntimeCompositeExecutorCapability)
            and coordinator.route_family == "operator_offload"
            and coordinator.baseline_executor_id is not None
        ):
            baseline = self.catalog.composite_executor_by_id[
                coordinator.baseline_executor_id
            ]
            baseline_by_id = {
                row.operator_id: row.primary_device_id
                for row in baseline.operator_placements
            }
        for operator in manifest.operators:
            primary, helper, fraction = pattern.assignments[operator.operator_id]
            operator_work = work_by_id[operator.operator_id]
            operator_tensors = tuple(tensors[value] for value in operator.tensor_ids)
            if (
                helper is None
                and baseline_by_id
                and primary == coordinator.helper_device_id
            ):
                if pattern.assistance_phase == "decode":
                    candidate = self._decode_offload_candidate(
                        operator.operator_id,
                        operator.kind,
                        baseline_by_id[operator.operator_id],
                        primary,
                        operator_tensors,
                        work,
                    )
                else:
                    candidate = self._offload_candidate(
                        operator.operator_id,
                        operator.kind,
                        baseline_by_id[operator.operator_id],
                        primary,
                        operator_work,
                        operator_tensors,
                        work,
                        maximum_transfer_tokens,
                    )
            elif helper is None:
                candidate = self._local_candidate(
                    operator.operator_id,
                    operator.kind,
                    primary,
                    operator_work,
                    operator_tensors,
                    work,
                )
            else:
                if pattern.assistance_phase == "decode":
                    candidate = self._decode_split_candidate(
                        operator.operator_id,
                        operator.kind,
                        primary,
                        helper,
                        fraction,
                        pattern.split_axis,
                        operator_tensors,
                        work,
                        full_width_output,
                    )
                else:
                    candidate = self._split_candidate(
                        operator.operator_id,
                        operator.kind,
                        primary,
                        helper,
                        fraction,
                        pattern.split_axis,
                        operator_work,
                        operator_tensors,
                        work,
                        full_width_output,
                        maximum_transfer_tokens,
                    )
            rows.append(OperatorNode(
                operator_id=operator.operator_id,
                layer_id=operator.layer_id,
                input_bytes=operator_work.activation_bytes,
                output_bytes=operator_work.output_bytes,
                candidates=(candidate,),
            ))
        if (
            pattern.route_family == "whole_model"
            and isinstance(coordinator, RuntimeExecutorCapability)
            and coordinator.adapter_parameters.get("request_io_protocol")
                == "token-ids-v1"
        ):
            token_bytes = coordinator.adapter_parameters.get(
                "token_id_bytes", 4
            )
            if type(token_bytes) is not int or token_bytes <= 0:
                raise RouteGenerationError(
                    "whole-model token width is invalid"
                )
            rows[0] = replace(
                rows[0],
                input_bytes=max(1, work.input_tokens * token_bytes),
            )
            rows[-1] = replace(
                rows[-1],
                output_bytes=max(1, work.output_tokens * token_bytes),
            )
        return tuple(rows)

    def _desktop_placement_sha256(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
    ) -> str:
        if set(pattern.desktop_assignments) != {
            row.operator_id for row in manifest.operators
        }:
            raise RouteGenerationError(
                "desktop placement does not cover the model graph"
            )
        coordinator = self._coordinator(pattern)
        cuda_graph_mode = (
            coordinator.adapter_parameters.get("cuda_graph_mode", "default")
            if coordinator is not None else "default"
        )
        remote_resident = RouteRemoteResidentMixin._remote_resident_group(
            self, manifest, pattern, coordinator
        )
        key = (
            manifest.artifact_sha256,
            tuple(sorted(pattern.desktop_assignments.items())),
            cuda_graph_mode,
            None if remote_resident is None else remote_resident.geometry_sha256,
        )
        cached = self._desktop_placement_hash_cache.get(key)
        if cached is not None:
            return cached
        result = canonical_sha256(desktop_control_placement_payload(
            manifest.artifact_sha256,
            [
                RuntimeCompositeOperatorPlacement(
                    operator_id=operator.operator_id,
                    primary_device_id=pattern.desktop_assignments[
                        operator.operator_id
                    ],
                    helper_device_id=None,
                    split_axis="none",
                    split_fraction_ppm=0,
                )
                for operator in sorted(
                    manifest.operators,
                    key=lambda row: row.operator_id,
                )
            ],
            cuda_graph_mode,
            remote_resident,
        ))
        if len(self._desktop_placement_hash_cache) >= 4_096:
            self._desktop_placement_hash_cache.pop(next(iter(
                self._desktop_placement_hash_cache
            )))
        self._desktop_placement_hash_cache[key] = result
        return result
