"""Materialize the Gemma split table through the unified scheduler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research_dev.scheduler import (
    OperatorSplitPolicy,
    ParallelSplitBalance,
    ParallelSplitShapeMeasurement,
    balance_parallel_split,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = HERE / "FIXED_POLICY_SHAPE_CALIBRATION_ABBA_V1.json"
QUALIFIED_TABLE = "1:6144,16:6144,512:0"
VARIANTS = ("qualified", "shape-balanced")


class ShapePolicyError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ShapePolicyError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_calibration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(
        type(value) is dict
        and value.get("schema")
            == "s42-full-fp16-gemma-fixed-shape-calibration-v1"
        and value.get("status") == "PASS",
        "shape calibration identity",
    )
    runs = value.get("source_runs")
    require(
        type(runs) is list
        and len(runs) == 2
        and [row.get("repeat_index") for row in runs] == [1, 2]
        and {row.get("plan_sha256") for row in runs}
            == {
                "a55806a14c5808e442c31831b7afc6a68f7f195813343176c28b0f5f0641cc3b"
            }
        and all(
            type(row.get(name)) is str and len(row[name]) == 64
            for row in runs
            for name in ("result_sha256", "server_log_sha256")
        ),
        "shape calibration source receipts",
    )
    return value


def _qualified_policy() -> OperatorSplitPolicy:
    return OperatorSplitPolicy.from_table(
        policy_id="gemma-f16-fixed-6144-m1-m16-v1",
        operator_family="gemma4-dense-ffn-geglu",
        layer_ids=range(23),
        n_embd=3840,
        eligible_columns=6144,
        max_tokens=512,
        column_quantum=1024,
        alternate_columns=(),
        io_type="f16",
        weight_layout="view_safe_dense_ffn",
        table=QUALIFIED_TABLE,
    )


def materialize_gemma_policy(
    variant: str,
    calibration_path: Path = DEFAULT_CALIBRATION,
) -> tuple[OperatorSplitPolicy, ParallelSplitBalance | None, str | None]:
    require(variant in VARIANTS, "shape policy variant")
    if variant == "qualified":
        return _qualified_policy(), None, None

    value = read_calibration(calibration_path)
    rows = value.get("measurements")
    require(type(rows) is list and rows, "shape calibration measurements")
    source_hashes = tuple(
        "sha256:" + row[name]
        for row in value["source_runs"]
        for name in ("result_sha256", "server_log_sha256")
    )
    measurements = tuple(
        ParallelSplitShapeMeasurement(
            tokens=row.get("tokens"),
            calls=row.get("calls"),
            total_columns=row.get("total_columns"),
            phone_columns=row.get("phone_columns"),
            phone_rpc_us=row.get("phone_rpc_us"),
            phone_compute_us=row.get("phone_compute_us"),
            host_us=row.get("host_us"),
            evidence_ids=source_hashes,
        )
        for row in rows
    )
    balance = balance_parallel_split(
        measurements,
        candidate_columns=value.get("candidate_columns"),
        alternate_columns=value.get("alternate_columns", ()),
        physical_max_tokens=value.get("physical_max_tokens"),
        policy_max_tokens=value.get("policy_max_tokens"),
        resident_phone_columns=value.get("resident_phone_columns"),
        column_quantum=value.get("column_quantum"),
        minimum_saving_ppm=value.get("minimum_saving_ppm"),
    )
    policy = OperatorSplitPolicy.from_table(
        policy_id="gemma-f16-shape-balanced-shadow-v1",
        operator_family="gemma4-dense-ffn-geglu",
        layer_ids=range(23),
        n_embd=3840,
        eligible_columns=value["resident_phone_columns"],
        max_tokens=value["policy_max_tokens"],
        column_quantum=value["column_quantum"],
        alternate_columns=tuple(value.get("alternate_columns", ())),
        io_type="f16",
        weight_layout="view_safe_dense_ffn",
        table=balance.table,
    )
    return policy, balance, file_sha256(calibration_path)
