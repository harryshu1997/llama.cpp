#!/usr/bin/env python3
"""Strict evidence-bundle validator and deterministic binder (S10-V0-R-E1).

Implements EVIDENCE_CONTRACT.md. Research-only. It performs schema-shape,
semantic, cross-record, hash, and eligibility validation and emits stable E_*
diagnostics. It contains no optimality-search code and imports no oracle.

Everything fails closed. There is no default value, no best-effort path, and no
warning-and-continue. A quantity that is not proven by a PASS record bound to a
hashed artifact cannot reach the solver.

The single most important structural rule: required bindings are computed FROM
THE INSTANCE (see required_targets), never from the binding list. Deleting a
binding entry is therefore E_BINDING_MISSING, not a skipped check.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
from collections import defaultdict

import canon

MIN_PROCESSES = 3
MIN_SAMPLES = 30
MIN_POWER_SAMPLES = 100
MAX_RECORDS = 256
MAX_BINDINGS = 1024
MAX_STRING = 256
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "fixtures" / "evidence"
SCHEMA_WORKER = pathlib.Path(__file__).with_name("schema_gate.py")
SYSTEM_PYTHON = pathlib.Path("/usr/bin/python3")

INSTRUMENT_SCOPES = {
    "SYNTHETIC": frozenset({"GPU_BOARD", "SERVER_WALL", "TOTAL_WALL"}),
    "NVML": frozenset({"GPU_BOARD"}),
    "EXTERNAL_SERVER_METER": frozenset({"SERVER_WALL"}),
    "EXTERNAL_TOTAL_METER": frozenset({"TOTAL_WALL"}),
    "COMPONENT_RAIL_METER": frozenset({"GPU_BOARD", "SERVER_WALL"}),
}

RECORD_SECTIONS = (
    ("artifacts", None),
    ("correctness", "CorrectnessCertificate"),
    ("routes", "RouteProfile"),
    ("boundaries", "BoundaryProfile"),
    ("thermal", "ThermalInterferenceProfile"),
    ("power", "PowerProfile"),
)


class EvidenceError(ValueError):
    """Raised for a fatal, non-recoverable evidence failure."""


def _fail(failures, code, message):
    failures.append(f"{code}: {message}")


def _check_schema(kind, document, failures):
    """Validate one live document with the checked-in draft-2020 schema.

    The ambient Python has no jsonschema package. The worker runs under the
    system interpreter that provides it. Any worker, schema, or protocol error
    is a refusal; there is no hand-written-validation fallback.
    """
    try:
        payload = json.dumps({"kind": kind, "document": document},
                             ensure_ascii=True, allow_nan=False)
        result = subprocess.run(
            [str(SYSTEM_PYTHON), "-I", str(SCHEMA_WORKER)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, TypeError, ValueError, RecursionError,
            subprocess.SubprocessError) as exc:
        _fail(failures, "E_SCHEMA_ENGINE",
              f"cannot run the schema gate for {kind}: {type(exc).__name__}: {exc}")
        return False
    try:
        reply = json.loads(result.stdout)
    except (TypeError, ValueError, RecursionError) as exc:
        _fail(failures, "E_SCHEMA_ENGINE",
              f"schema gate for {kind} returned invalid JSON: {exc}")
        return False
    if result.returncode != 0 or "engine_error" in reply:
        detail = reply.get("engine_error", result.stderr.strip() or
                           f"exit {result.returncode}")
        _fail(failures, "E_SCHEMA_ENGINE", f"schema gate for {kind} failed: {detail}")
        return False
    errors = reply.get("errors")
    if not isinstance(errors, list):
        _fail(failures, "E_SCHEMA_ENGINE",
              f"schema gate for {kind} returned no errors array")
        return False
    for error in errors:
        if not isinstance(error, dict):
            _fail(failures, "E_SCHEMA_ENGINE",
                  f"schema gate for {kind} returned a malformed diagnostic")
            return False
        _fail(failures, "E_SCHEMA",
              f"{kind}{error.get('path', '/')}: {error.get('message', 'invalid')}")
    return not errors


def _ident_ok(value):
    return (isinstance(value, str) and 0 < len(value) <= MAX_STRING
            and value.isascii())


def _no_dot(value):
    """Target paths are dot-separated, so an id containing a dot is ambiguous.

    An ambiguous target is a fail-open surface: two different instance fields
    could parse to the same target string, letting one binding stand in for
    another. Reject the id instead of guessing.
    """
    return "." not in value


# ---------------------------------------------------------------------------
# structural bundle validation
# ---------------------------------------------------------------------------

def validate_bundle_shape(bundle, failures):
    if not isinstance(bundle, dict):
        _fail(failures, "E_SCHEMA", "bundle is not an object")
        return False
    expected = {"schema_version", "bundle_id", "provenance",
                "evaluation_timestamp_utc", "artifacts",
                "correctness", "routes", "boundaries", "thermal", "power",
                "bundle_sha256"}
    if set(bundle) != expected:
        missing = sorted(expected - set(bundle))
        extra = sorted(set(bundle) - expected)
        _fail(failures, "E_SCHEMA",
              f"bundle fields wrong (missing={missing}, unknown={extra})")
        return False
    if bundle["schema_version"] != 3 or type(bundle["schema_version"]) is not int:
        _fail(failures, "E_VERSION",
              f"unsupported bundle schema_version {bundle['schema_version']!r}")
        return False
    if bundle["provenance"] not in ("MEASURED", "MECHANICS_ONLY"):
        _fail(failures, "E_SCHEMA",
              f"unknown bundle provenance {bundle['provenance']!r}")
        return False
    if not _ident_ok(bundle["bundle_id"]):
        _fail(failures, "E_SCHEMA", "invalid bundle_id")
        return False
    for section, _kind in RECORD_SECTIONS:
        if not isinstance(bundle[section], list):
            _fail(failures, "E_SCHEMA", f"{section} is not an array")
            return False
        if len(bundle[section]) > MAX_RECORDS:
            _fail(failures, "E_SCHEMA",
                  f"{section} exceeds MAX_RECORDS={MAX_RECORDS}")
            return False
    return True


def _check_ids_unique(bundle, failures):
    seen = {}
    for artifact in bundle["artifacts"]:
        aid = artifact.get("artifact_id")
        if not _ident_ok(aid):
            _fail(failures, "E_SCHEMA", f"invalid artifact_id {aid!r}")
            continue
        if aid in seen:
            _fail(failures, "E_DUPLICATE_ID", f"artifact_id repeated: {aid}")
        seen[aid] = artifact
    records = {}
    for section, kind in RECORD_SECTIONS:
        if kind is None:
            continue
        for record in bundle[section]:
            rid = record.get("record_id")
            if not _ident_ok(rid):
                _fail(failures, "E_SCHEMA", f"invalid record_id {rid!r}")
                continue
            if record.get("kind") != kind:
                _fail(failures, "E_SCHEMA",
                      f"record {rid} in {section} has kind {record.get('kind')!r}")
                continue
            if record.get("record_version") != 3:
                _fail(failures, "E_VERSION",
                      f"record {rid} has unsupported record_version "
                      f"{record.get('record_version')!r}")
                continue
            if rid in records:
                _fail(failures, "E_DUPLICATE_ID", f"record_id repeated: {rid}")
            records[rid] = record
    return seen, records


def _check_hashes(bundle, records, failures):
    for rid, record in sorted(records.items()):
        if "record_sha256" not in record:
            _fail(failures, "E_SCHEMA", f"record {rid} has no record_sha256")
            continue
        try:
            computed = canon.record_digest(record)
        except (TypeError, ValueError) as exc:
            _fail(failures, "E_SCHEMA", f"record {rid} is not canonicalisable: {exc}")
            continue
        if computed != record["record_sha256"]:
            _fail(failures, "E_HASH",
                  f"record {rid} digest mismatch: declared "
                  f"{record['record_sha256']}, computed {computed}")
    try:
        computed = canon.bundle_digest(bundle)
    except (TypeError, ValueError) as exc:
        _fail(failures, "E_SCHEMA", f"bundle is not canonicalisable: {exc}")
        return
    if computed != bundle["bundle_sha256"]:
        _fail(failures, "E_HASH",
              f"bundle digest mismatch: declared {bundle['bundle_sha256']}, "
              f"computed {computed}")


def _check_artifact_files(artifacts, artifact_root, failures):
    """Resolve and hash every artifact under one explicit trusted root."""
    try:
        root = pathlib.Path(artifact_root).resolve(strict=True)
    except (OSError, TypeError) as exc:
        _fail(failures, "E_ARTIFACT_PATH",
              f"artifact root {artifact_root!r} is not usable: {exc}")
        return
    if not root.is_dir():
        _fail(failures, "E_ARTIFACT_PATH", f"artifact root {root} is not a directory")
        return
    for aid, artifact in sorted(artifacts.items()):
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str):
            _fail(failures, "E_ARTIFACT_PATH",
                  f"artifact {aid} has no relative path")
            continue
        candidate = pathlib.PurePosixPath(raw_path)
        if candidate.is_absolute() or not candidate.parts or \
                any(part in ("", ".", "..") for part in candidate.parts):
            _fail(failures, "E_ARTIFACT_PATH",
                  f"artifact {aid} path {raw_path!r} is not a normalized relative path")
            continue
        path = root.joinpath(*candidate.parts)
        try:
            # A symlink can redirect a digest check outside the trusted evidence
            # tree after review. Reject every symlink component, not just the leaf.
            current = root
            for part in candidate.parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError(f"symlink component {current}")
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except FileNotFoundError:
            _fail(failures, "E_ARTIFACT_MISSING",
                  f"artifact {aid} does not exist at {path}")
            continue
        except (OSError, ValueError) as exc:
            _fail(failures, "E_ARTIFACT_PATH",
                  f"artifact {aid} path {raw_path!r} is unsafe: {exc}")
            continue
        if not resolved.is_file():
            _fail(failures, "E_ARTIFACT_TYPE",
                  f"artifact {aid} path {resolved} is not a regular file")
            continue
        try:
            actual = canon.file_sha256(resolved)
        except OSError as exc:
            _fail(failures, "E_ARTIFACT_MISSING",
                  f"artifact {aid} cannot be read: {exc}")
            continue
        if actual != artifact.get("sha256"):
            _fail(failures, "E_ARTIFACT_HASH",
                  f"artifact {aid} digest mismatch: declared {artifact.get('sha256')}, "
                  f"computed {actual}")


def _timestamp_us(value):
    """Parse the contract's UTC timestamp without float epoch conversion."""
    parsed = datetime.datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    delta = parsed - datetime.datetime(1970, 1, 1)
    return (delta.days * 86400 + delta.seconds) * 1000000


def _check_artifact_validity(bundle, artifacts, failures):
    """Require every artifact to cover the bundle's signed evaluation epoch."""
    value = bundle.get("evaluation_timestamp_utc")
    try:
        evaluation_us = _timestamp_us(value)
    except (TypeError, ValueError) as exc:
        _fail(failures, "E_STALE",
              f"bundle evaluation_timestamp_utc {value!r} is invalid: {exc}")
        return
    for aid, artifact in sorted(artifacts.items()):
        timestamp = artifact.get("timestamp_utc")
        validity = artifact.get("validity_us")
        try:
            timestamp_us = _timestamp_us(timestamp)
        except (TypeError, ValueError) as exc:
            _fail(failures, "E_STALE",
                  f"artifact {aid} timestamp_utc {timestamp!r} is invalid: {exc}")
            continue
        if not canon.is_int(validity) or validity <= 0:
            _fail(failures, "E_STALE",
                  f"artifact {aid} has no positive validity_us")
            continue
        if timestamp_us > evaluation_us:
            _fail(failures, "E_STALE",
                  f"artifact {aid} timestamp {timestamp} is after bundle evaluation "
                  f"epoch {value}")
        elif evaluation_us > timestamp_us + validity:
            _fail(failures, "E_STALE",
                  f"artifact {aid} expired before bundle evaluation epoch {value}")


def _check_artifact_links(bundle, artifacts, records, failures):
    for rid, record in sorted(records.items()):
        listed = record.get("artifacts")
        if not isinstance(listed, list) or not listed:
            _fail(failures, "E_EMPTY_ARTIFACTS",
                  f"record {rid} binds no hashed artifact")
            continue
        for aid in listed:
            if aid not in artifacts:
                _fail(failures, "E_MISSING_ARTIFACT",
                      f"record {rid} names artifact {aid} which is not in the bundle")

        if record.get("kind") == "CorrectnessCertificate":
            proof_aid = (record.get("no_fallback_proof") or {}).get("artifact_id")
            if proof_aid not in artifacts:
                _fail(failures, "E_MISSING_ARTIFACT",
                      f"correctness {rid} names no-fallback artifact {proof_aid!r} "
                      f"which is not in the bundle")
            if proof_aid not in listed:
                _fail(failures, "E_FALLBACK",
                      f"correctness {rid} no-fallback artifact {proof_aid!r} is not "
                      f"included in that certificate's artifact set")
            reference_aid = (record.get("reference_route") or {}).get("artifact_id")
            if reference_aid not in artifacts:
                _fail(failures, "E_MISSING_ARTIFACT",
                      f"correctness {rid} names reference artifact "
                      f"{reference_aid!r} which is not in the bundle")
            if reference_aid not in listed:
                _fail(failures, "E_CORRECTNESS",
                      f"correctness {rid} reference artifact {reference_aid!r} is not "
                      f"included in that certificate's artifact set")
            if proof_aid == reference_aid:
                _fail(failures, "E_FALLBACK",
                      f"correctness {rid} uses its CPU reference artifact as the "
                      f"target-route no-fallback proof")
            proof_artifact = artifacts.get(proof_aid)
            reference_artifact = artifacts.get(reference_aid)
            if proof_artifact is not None and reference_artifact is not None and (
                    proof_artifact.get("path") == reference_artifact.get("path") or
                    proof_artifact.get("sha256") == reference_artifact.get("sha256")):
                _fail(failures, "E_FALLBACK",
                      f"correctness {rid} aliases its target-route no-fallback proof "
                      f"and CPU reference through the same path or payload; distinct "
                      f"artifact IDs do not make one artifact two independent proofs")
            if proof_artifact is not None and (
                    proof_artifact.get("device_id") != record.get("device_id") or
                    proof_artifact.get("backend_build") != record.get("backend_build")):
                _fail(failures, "E_FALLBACK",
                      f"correctness {rid} no-fallback artifact {proof_aid} was not "
                      f"produced by its target device/backend")


def _check_identity_against_artifacts(record, artifacts, failures):
    """A record may not claim a device or build its own artifacts do not show."""
    rid = record["record_id"]
    revisions = set()
    reference_aid = None
    reference = {}
    if record.get("kind") == "CorrectnessCertificate":
        reference = record.get("reference_route") or {}
        reference_aid = reference.get("artifact_id")
    for aid in record.get("artifacts", []):
        artifact = artifacts.get(aid)
        if artifact is None:
            continue
        revisions.add(artifact.get("source_revision"))
        expected_device = reference.get("device_id") if aid == reference_aid \
            else record.get("device_id")
        expected_build = reference.get("backend_build") if aid == reference_aid \
            else record.get("backend_build")
        if expected_device is not None and artifact.get("device_id") != expected_device:
            _fail(failures, "E_STALE",
                  f"record {rid} claims device {expected_device} but artifact "
                  f"{aid} was produced on {artifact.get('device_id')}")
        if (expected_build is not None
                and artifact.get("backend_build") != expected_build):
            _fail(failures, "E_STALE",
                  f"record {rid} claims backend build {expected_build} but "
                  f"artifact {aid} was produced by {artifact.get('backend_build')}")
    if len(revisions) > 1:
        _fail(failures, "E_STALE",
              f"record {rid} mixes source revisions {sorted(revisions)}")


def _check_eligibility(bundle, artifacts, records, failures):
    for rid, record in sorted(records.items()):
        status = record.get("status")
        if status not in ("PASS", "FAIL", "UNKNOWN", "INELIGIBLE"):
            _fail(failures, "E_SCHEMA", f"record {rid} has invalid status {status!r}")
            continue
        if not _ident_ok(record.get("reason_code")):
            _fail(failures, "E_SCHEMA", f"record {rid} has no stable reason_code")
        _check_identity_against_artifacts(record, artifacts, failures)
        if record.get("validity_us") == 0:
            _fail(failures, "E_STALE", f"record {rid} has an expired validity window")

        kind = record["kind"]
        if kind == "CorrectnessCertificate":
            _check_correctness(record, failures)
        elif kind == "RouteProfile":
            _check_route(record, records, artifacts, failures)
        elif kind == "BoundaryProfile":
            _check_boundary(record, failures)
        elif kind == "ThermalInterferenceProfile":
            _check_thermal(record, failures)
        elif kind == "PowerProfile":
            _check_power(record, failures)


def _check_correctness(record, failures):
    rid = record["record_id"]
    _check_shape_envelope(record.get("shape_envelope"), rid, failures)
    if record["status"] != "PASS":
        return
    if record.get("verdict") != "PASS":
        _fail(failures, "E_CORRECTNESS",
              f"correctness {rid} is status PASS but verdict {record.get('verdict')!r}")
    observed = record.get("observed_ppm")
    threshold = record.get("threshold_ppm")
    # Guard idiom: fail closed when the type is wrong. Written the other way
    # round ("if is_int(a) and is_int(b) and bad") a non-integer would SKIP the
    # comparison entirely and bind.
    if not canon.is_int(observed) or not canon.is_int(threshold):
        _fail(failures, "E_CORRECTNESS",
              f"correctness {rid} has non-integer ppm fields "
              f"(observed={observed!r}, threshold={threshold!r})")
    elif record.get("metric") == "EXACT" and (observed != 0 or threshold != 0):
        _fail(failures, "E_CORRECTNESS",
              f"correctness {rid} uses metric EXACT but observed={observed} ppm "
              f"and threshold={threshold} ppm; exact equality requires both to be zero")
    elif observed > threshold:
        _fail(failures, "E_CORRECTNESS",
              f"correctness {rid} is status PASS but observed {observed} ppm "
              f"exceeds threshold {threshold} ppm")
    proof = record.get("no_fallback_proof") or {}
    if proof.get("method") == "NONE":
        _fail(failures, "E_FALLBACK",
              f"correctness {rid} claims PASS with no no-fallback proof method")
    observed_fallback = proof.get("fallback_ops_observed")
    if not canon.is_int(observed_fallback):
        _fail(failures, "E_FALLBACK",
              f"correctness {rid} has a non-integer fallback count "
              f"{observed_fallback!r}")
    elif observed_fallback > 0:
        _fail(failures, "E_FALLBACK",
              f"correctness {rid} observed {observed_fallback} fallback ops; a route "
              f"that silently fell back to CPU is not an accelerated route")
    reference = record.get("reference_route") or {}
    if reference.get("device_id") == record.get("device_id") and \
       reference.get("kernel_id") == record.get("kernel_id"):
        _fail(failures, "E_CORRECTNESS",
              f"correctness {rid} compares a route against itself")
    if reference.get("reference_kind") != "CPU_FP32":
        _fail(failures, "E_CORRECTNESS",
              f"correctness {rid} does not use the CPU_FP32 reference kind")


def _check_shape_envelope(envelope, rid, failures):
    if not isinstance(envelope, dict):
        _fail(failures, "E_ENVELOPE", f"record {rid} has no shape envelope")
        return
    values = tuple(envelope.get(field) for field in
                   ("tokens_min", "tokens_max", "kv_min", "kv_max"))
    if not all(canon.is_int(value) for value in values):
        _fail(failures, "E_ENVELOPE",
              f"record {rid} shape envelope contains non-integer bounds")
        return
    tokens_min, tokens_max, kv_min, kv_max = values
    if tokens_min > tokens_max or kv_min > kv_max:
        _fail(failures, "E_ENVELOPE", f"record {rid} shape envelope is inverted")


def _envelope_contains(outer, inner):
    fields = ("tokens_min", "tokens_max", "kv_min", "kv_max")
    if not isinstance(outer, dict) or not isinstance(inner, dict) or \
            not all(canon.is_int(outer.get(field)) and canon.is_int(inner.get(field))
                    for field in fields):
        return False
    return (outer["tokens_min"] <= inner["tokens_min"] and
            outer["tokens_max"] >= inner["tokens_max"] and
            outer["kv_min"] <= inner["kv_min"] and
            outer["kv_max"] >= inner["kv_max"])


def _artifact_field_values(record, artifacts, field):
    return {artifacts[aid].get(field) for aid in record.get("artifacts", [])
            if aid in artifacts}


def _check_artifact_revision_coherence(route, related, relation, artifacts, failures):
    """Require linked records to describe one source and build revision set."""
    for field in ("source_revision", "build_revision"):
        route_values = _artifact_field_values(route, artifacts, field)
        related_values = _artifact_field_values(related, artifacts, field)
        if route_values != related_values:
            _fail(failures, "E_STALE",
                  f"route {route['record_id']} and {relation} "
                  f"{related['record_id']} come from different {field} sets "
                  f"{sorted(route_values)} and {sorted(related_values)}")


def _check_route(record, records, artifacts, failures):
    rid = record["record_id"]
    _check_shape_envelope(record.get("shape_envelope"), rid, failures)
    latency = record.get("latency") or {}
    p50, p95, mx = latency.get("p50_us"), latency.get("p95_us"), latency.get("max_us")
    if all(canon.is_int(v) for v in (p50, p95, mx)):
        if not (p50 <= p95 <= mx):
            _fail(failures, "E_SCHEMA",
                  f"route {rid} latency is not monotone: p50={p50} p95={p95} max={mx}")
    else:
        _fail(failures, "E_SCHEMA", f"route {rid} latency fields are not integers")
    if record["status"] != "PASS":
        return
    if not canon.is_int(record.get("process_count")) or \
            record["process_count"] < MIN_PROCESSES:
        _fail(failures, "E_SAMPLES",
              f"route {rid} has process_count {record.get('process_count')!r} "
              f"below MIN_PROCESSES={MIN_PROCESSES}")
    if not canon.is_int(record.get("sample_count")) or \
            record["sample_count"] < MIN_SAMPLES:
        _fail(failures, "E_SAMPLES",
              f"route {rid} has sample_count {record.get('sample_count')!r} "
              f"below MIN_SAMPLES={MIN_SAMPLES}")

    cert = records.get(record.get("correctness_id"))
    if cert is None:
        _fail(failures, "E_CORRECTNESS",
              f"route {rid} names correctness {record.get('correctness_id')!r} "
              f"which is not in the bundle")
    elif cert.get("kind") != "CorrectnessCertificate":
        _fail(failures, "E_CORRECTNESS",
              f"route {rid} names {cert['record_id']} which is not a correctness record")
    else:
        if cert.get("status") != "PASS":
            _fail(failures, "E_CORRECTNESS",
                  f"route {rid} is PASS but its correctness {cert['record_id']} is "
                  f"{cert.get('status')}")
        if cert.get("route_kind") != "ACCELERATED":
            _fail(failures, "E_CORRECTNESS",
                  f"route {rid} uses correctness {cert['record_id']} with "
                  f"route_kind={cert.get('route_kind')!r}; schedulable routes require "
                  f"an ACCELERATED target-vs-reference comparison")
        for field in ("island_id", "graph_id", "model_id", "weight_set_id",
                      "device_id", "backend_build", "kernel_id"):
            if cert.get(field) != record.get(field):
                _fail(failures, "E_IDENTITY",
                      f"route {rid} {field}={record.get(field)!r} does not match its "
                      f"correctness {cert['record_id']} {field}={cert.get(field)!r}; a "
                      f"correctness point for one configuration is not evidence for "
                      f"another")
        if not _envelope_contains(cert.get("shape_envelope"),
                                  record.get("shape_envelope")):
            _fail(failures, "E_ENVELOPE",
                  f"route {rid} shape envelope is not covered by correctness "
                  f"{cert['record_id']}; correctness for one shape is not evidence "
                  f"for another")
        _check_artifact_revision_coherence(
            record, cert, "correctness", artifacts, failures)

    thermal = records.get(record.get("thermal_id"))
    if thermal is None:
        _fail(failures, "E_THERMAL",
              f"route {rid} names thermal {record.get('thermal_id')!r} which is not "
              f"in the bundle")
    elif thermal.get("kind") != "ThermalInterferenceProfile":
        _fail(failures, "E_THERMAL",
              f"route {rid} names {thermal['record_id']} which is not a thermal record")
    else:
        if thermal.get("status") != "PASS":
            _fail(failures, "E_THERMAL",
                  f"route {rid} is PASS but its thermal {thermal['record_id']} is "
                  f"{thermal.get('status')}")
        if thermal.get("device_id") != record.get("device_id"):
            _fail(failures, "E_THERMAL",
                  f"route {rid} on {record.get('device_id')} binds a thermal profile "
                  f"measured on {thermal.get('device_id')}")
        if thermal.get("backend_build") != record.get("backend_build"):
            _fail(failures, "E_THERMAL",
                  f"route {rid} uses backend build {record.get('backend_build')} but "
                  f"its thermal profile was measured with "
                  f"{thermal.get('backend_build')}")
        _check_artifact_revision_coherence(
            record, thermal, "thermal profile", artifacts, failures)

    boundary = records.get(record.get("boundary_id"))
    if boundary is None:
        _fail(failures, "E_IDENTITY",
              f"route {rid} names boundary {record.get('boundary_id')!r} which is "
              f"not in the bundle")
    elif boundary.get("kind") != "BoundaryProfile":
        _fail(failures, "E_IDENTITY",
              f"route {rid} boundary_id names a non-boundary record")
    else:
        if boundary.get("status") != "PASS":
            _fail(failures, "E_STATUS",
                  f"route {rid} binds boundary {boundary['record_id']} with status "
                  f"{boundary.get('status')}")
        for field in ("island_id", "graph_id", "model_id", "weight_set_id",
                      "device_id", "backend_build"):
            if boundary.get(field) != record.get(field):
                _fail(failures, "E_IDENTITY",
                      f"route {rid} {field}={record.get(field)!r} does not match its "
                      f"boundary {boundary['record_id']} {field}="
                      f"{boundary.get(field)!r}")
        _check_artifact_revision_coherence(
            record, boundary, "boundary", artifacts, failures)
        boundary_wall = boundary.get("boundary_wall_us")
        if canon.is_int(boundary_wall) and canon.is_int(p95) and p95 < boundary_wall:
            _fail(failures, "E_INCOHERENT",
                  f"route {rid} p95_us={p95} is shorter than its pinned boundary "
                  f"wall time {boundary_wall} us; END_TO_END latency cannot omit "
                  f"boundary work")
        if record.get("h2d_bytes") != boundary.get("h2d_bytes") or \
                record.get("d2h_bytes") != boundary.get("d2h_bytes"):
            _fail(failures, "E_INCOHERENT",
                  f"route {rid} declares h2d/d2h bytes "
                  f"({record.get('h2d_bytes')},{record.get('d2h_bytes')}) but "
                  f"boundary {boundary['record_id']} declares h2d/d2h bytes "
                  f"({boundary.get('h2d_bytes')},"
                  f"{boundary.get('d2h_bytes')})")


def _check_boundary(record, failures):
    rid = record["record_id"]
    if record["status"] != "PASS":
        return
    if not canon.is_int(record.get("process_count")) or \
            record["process_count"] < MIN_PROCESSES:
        _fail(failures, "E_SAMPLES",
              f"boundary {rid} has process_count {record.get('process_count')!r} "
              f"below MIN_PROCESSES={MIN_PROCESSES}")
    if not canon.is_int(record.get("sample_count")) or \
            record["sample_count"] < MIN_SAMPLES:
        _fail(failures, "E_SAMPLES",
              f"boundary {rid} has sample_count {record.get('sample_count')!r} "
              f"below MIN_SAMPLES={MIN_SAMPLES}")
    if not canon.is_int(record.get("energy_nj")):
        _fail(failures, "E_SCHEMA",
              f"boundary {rid} has a non-integer energy_nj {record.get('energy_nj')!r}")
    elif record["energy_nj"] > 0 and record.get("energy_scope") == "NONE":
        _fail(failures, "E_SCOPE",
              f"boundary {rid} reports {record['energy_nj']} nJ with no declared "
              f"energy scope; energy without a boundary is not a measurement")
    stages = tuple(record.get(field) for field in
                   ("transfer_us", "verification_us", "materialization_us",
                    "prepare_us", "warmup_us"))
    wall = record.get("boundary_wall_us")
    if not canon.is_int(wall) or not all(canon.is_int(stage) for stage in stages):
        _fail(failures, "E_SCHEMA",
              f"boundary {rid} has non-integer stage or wall timing")
    elif wall < max(stages):
        _fail(failures, "E_INCOHERENT",
              f"boundary {rid} wall time {wall} us is shorter than one of its "
              f"measured stages {stages}")
    if canon.is_int(record.get("energy_nj")) and record["energy_nj"] > 0 and wall == 0:
        _fail(failures, "E_INCOHERENT",
              f"boundary {rid} reports positive incremental energy over a zero-time "
              f"measurement window")
    transported = record.get("h2d_bytes", 0) or record.get("d2h_bytes", 0)
    if record.get("transport_domain") != "NONE" and transported and \
            (record.get("transfer_us") == 0 or wall == 0):
        _fail(failures, "E_INCOHERENT",
              f"boundary {rid} transports nonzero bytes over "
              f"{record.get('transport_domain')} but reports zero transfer/wall time")
    if record.get("transport_domain") == "NONE" and any(
            record.get(field) != 0
            for field in ("h2d_bytes", "d2h_bytes", "transfer_us")):
        _fail(failures, "E_INCOHERENT",
              f"boundary {rid} uses transport_domain NONE but h2d_bytes, d2h_bytes, "
              f"and transfer_us are not all zero")


def _check_thermal(record, failures):
    rid = record["record_id"]
    envelope = record.get("envelope") or {}
    lo, hi = envelope.get("temp_min_c"), envelope.get("temp_max_c")
    if not canon.is_int(lo) or not canon.is_int(hi):
        _fail(failures, "E_SCHEMA",
              f"thermal {rid} has non-integer envelope bounds ({lo!r}, {hi!r})")
    elif lo > hi:
        _fail(failures, "E_SCHEMA", f"thermal {rid} envelope is inverted")
    if not canon.is_int(record.get("duration_us")) or record["duration_us"] <= 0:
        _fail(failures, "E_THERMAL",
              f"thermal {rid} has no positive observation duration")
    if record["status"] != "PASS":
        return
    if record.get("thermal_state") == "UNKNOWN":
        _fail(failures, "E_THERMAL",
              f"thermal {rid} is PASS with an UNKNOWN thermal state")
    if not canon.is_int(record.get("validity_us")) or record["validity_us"] <= 0:
        _fail(failures, "E_STALE",
              f"thermal {rid} has no positive validity window")


def _check_power(record, failures):
    rid = record["record_id"]
    idle, active = record.get("idle_mw"), record.get("active_mw")
    if not canon.is_int(idle) or not canon.is_int(active):
        _fail(failures, "E_SCHEMA",
              f"power {rid} has non-integer power fields "
              f"(idle={idle!r}, active={active!r})")
    elif active < idle:
        _fail(failures, "E_SCHEMA",
              f"power {rid} active {active} mW is below idle {idle} mW")
    included = record.get("included_rails") or []
    excluded = record.get("excluded_rails") or []
    overlap = sorted(set(included) & set(excluded))
    if overlap:
        _fail(failures, "E_DOUBLE_COUNT",
              f"power {rid} lists rails {overlap} as both included and excluded")

    scope = record.get("scope")
    instrument_kind = record.get("instrument_kind")
    allowed_scopes = INSTRUMENT_SCOPES.get(instrument_kind, frozenset())
    if scope not in allowed_scopes:
        _fail(failures, "E_SCOPE",
              f"power {rid} instrument kind {instrument_kind!r} cannot observe scope "
              f"{scope}; instrument names do not upgrade measurement capability")

    if record["status"] != "PASS":
        return
    if not canon.is_int(record.get("sample_count")) or \
            record["sample_count"] < MIN_POWER_SAMPLES:
        _fail(failures, "E_SAMPLES",
              f"power {rid} has sample_count {record.get('sample_count')!r} below "
              f"MIN_POWER_SAMPLES={MIN_POWER_SAMPLES}")
    if not canon.is_int(record.get("sample_rate_hz")) or record["sample_rate_hz"] <= 0:
        _fail(failures, "E_SAMPLES", f"power {rid} has no positive sample rate")
    if record.get("synchronization") == "NONE" and scope == "TOTAL_WALL":
        _fail(failures, "E_SCOPE",
              f"power {rid} claims TOTAL_WALL with no clock synchronization; an "
              f"unsynchronized total boundary cannot be matched to a work window")
    if not _ident_ok(record.get("instrument")):
        _fail(failures, "E_SCHEMA", f"power {rid} declares no instrument")


def validate_bundle(bundle, artifact_root=DEFAULT_ARTIFACT_ROOT):
    """Return a sorted list of stable E_* diagnostics. Empty means valid."""
    failures = []
    if not validate_bundle_shape(bundle, failures):
        return failures
    # The single type gate. Every number in the bundle must be a true, in-range
    # integer BEFORE any comparison runs, because a float or bool compares equal
    # to an integer and would sail through every eligibility check below.
    try:
        canon.check_integers(bundle, "bundle")
    except ValueError as exc:
        _fail(failures, "E_SCHEMA", str(exc))
        return failures
    for section, kind in RECORD_SECTIONS:
        if kind is None:
            continue
        for record in bundle.get(section, []):
            if isinstance(record, dict) and record.get("record_version") != 3:
                _fail(failures, "E_VERSION",
                      f"record {record.get('record_id')!r} has unsupported "
                      f"record_version {record.get('record_version')!r}")
    if failures:
        return failures
    if not _check_schema("bundle", bundle, failures):
        return failures
    artifacts, records = _check_ids_unique(bundle, failures)
    _check_hashes(bundle, records, failures)
    _check_artifact_validity(bundle, artifacts, failures)
    _check_artifact_files(artifacts, artifact_root, failures)
    _check_artifact_links(bundle, artifacts, records, failures)
    _check_eligibility(bundle, artifacts, records, failures)
    return failures


# ---------------------------------------------------------------------------
# required bindings, derived FROM THE INSTANCE
# ---------------------------------------------------------------------------

def required_targets(inst):
    """Every evidence-derived field of the instance, with its declared value.

    Computed from the instance alone. The binding list never influences which
    targets are required, so an omitted binding is a missing binding.
    """
    targets = {}
    targets["activation_mem_bound_bytes"] = inst["activation_mem_bound_bytes"]
    for field in ("p8_mw", "p0_mw", "wake_us", "idle_entry_us", "transition_nj"):
        targets[f"server_power.{field}"] = inst["server_power"][field]
    for name, device in inst["devices"].items():
        targets[f"devices.{name}.active_mw"] = device["active_mw"]
    for node in inst["nodes"]:
        nid = node["id"]
        targets[f"nodes.{nid}.output_bytes"] = node["output_bytes"]
        for device, route in node["routes"].items():
            targets[f"nodes.{nid}.routes.{device}.duration_us"] = route["duration_us"]
            targets[f"nodes.{nid}.routes.{device}.extra_energy_nj"] = \
                route["extra_energy_nj"]
    for key, profile in inst["batch_profiles"].items():
        for size, duration in profile.items():
            targets[f"batch_profiles.{key}.{size}"] = duration
    return targets


# target -> (record kind, record field path)
def _expected_binding(target, inst):
    parts = target.split(".")
    if target == "activation_mem_bound_bytes":
        return "RouteProfile", "memory_bytes", "SERVER"
    if parts[0] == "server_power":
        field = {"p8_mw": "idle_mw", "p0_mw": "active_mw", "wake_us": "wake_us",
                 "idle_entry_us": "idle_entry_us",
                 "transition_nj": "transition_nj"}.get(parts[1])
        return "PowerProfile", field, "SERVER"
    if parts[0] == "devices":
        return "PowerProfile", "active_mw", parts[1]
    if parts[0] == "nodes" and parts[-1] == "output_bytes":
        return "BoundaryProfile", "output_bytes", None
    if parts[0] == "nodes" and parts[-1] == "duration_us":
        return "RouteProfile", "latency.p95_us", parts[3]
    if parts[0] == "nodes" and parts[-1] == "extra_energy_nj":
        # A BoundaryProfile is keyed by direction and transport, not by a device,
        # so there is no device_id to cross-check here.
        return "BoundaryProfile", "energy_nj", None
    if parts[0] == "batch_profiles":
        return "RouteProfile", "latency.p95_us", "SERVER"
    return None, None, None


def _resolve_field(record, field_path):
    """Return the field, or canon.MISSING. Never None-for-absent.

    None-for-absent would let a binding carrying `value: null` satisfy
    `record[field] == value` by None == None.
    """
    current = record
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return canon.MISSING
        current = current[part]
    return current


def validate_instance_shape(inst, failures):
    if not isinstance(inst, dict):
        _fail(failures, "E_SCHEMA", "instance is not an object")
        return False
    if inst.get("schema_version") != 3:
        _fail(failures, "E_VERSION",
              f"unsupported instance schema_version {inst.get('schema_version')!r}")
        return False
    evidence = inst.get("evidence")
    if not isinstance(evidence, dict):
        _fail(failures, "E_SCHEMA", "instance has no evidence block")
        return False
    if set(evidence) != {"schema_version", "scope", "bundle_sha256", "bindings"}:
        _fail(failures, "E_SCHEMA", "evidence block fields do not match v3")
        return False
    if evidence.get("schema_version") != 3:
        _fail(failures, "E_VERSION",
              f"unsupported evidence schema_version {evidence.get('schema_version')!r}")
        return False
    if evidence.get("scope") not in ("MECHANICS_ONLY", "MEASURED"):
        _fail(failures, "E_SCHEMA", f"unknown evidence scope {evidence.get('scope')!r}")
        return False
    if not isinstance(evidence.get("bindings"), list):
        _fail(failures, "E_SCHEMA", "evidence.bindings is not an array")
        return False
    if len(evidence["bindings"]) > MAX_BINDINGS:
        _fail(failures, "E_SCHEMA", f"more than MAX_BINDINGS={MAX_BINDINGS} bindings")
        return False
    # Same single type gate as the bundle: no floats, no booleans, no overflow.
    try:
        canon.check_integers(inst, "instance")
    except ValueError as exc:
        _fail(failures, "E_SCHEMA", str(exc))
        return False
    if not _check_schema("instance", inst, failures):
        return False

    node_ids = [node.get("id") for node in inst.get("nodes", [])]
    duplicate_node_ids = sorted({node_id for node_id in node_ids
                                 if node_ids.count(node_id) > 1})
    if duplicate_node_ids:
        _fail(failures, "E_DUPLICATE_ID",
              f"node ids are repeated: {duplicate_node_ids}")
        return False

    # Dot-free identifiers keep target paths unambiguous.
    for name in inst.get("devices", {}):
        if not _no_dot(name):
            _fail(failures, "E_SCHEMA",
                  f"device name {name!r} contains a dot and makes its binding target "
                  f"ambiguous")
            return False
    for node in inst.get("nodes", []):
        if not _no_dot(node.get("id", "")):
            _fail(failures, "E_SCHEMA",
                  f"node id {node.get('id')!r} contains a dot and makes its binding "
                  f"target ambiguous")
            return False
        for device in node.get("routes", {}):
            if not _no_dot(device):
                _fail(failures, "E_SCHEMA",
                      f"route device {device!r} contains a dot")
                return False
    for key in inst.get("batch_profiles", {}):
        if not _no_dot(key):
            _fail(failures, "E_SCHEMA", f"batch key {key!r} contains a dot")
            return False
    return True


def validate_binding(inst, bundle, failures):
    evidence = inst["evidence"]
    if evidence["bundle_sha256"] != bundle.get("bundle_sha256"):
        _fail(failures, "E_HASH",
              f"instance binds bundle {evidence['bundle_sha256']} but was given "
              f"bundle {bundle.get('bundle_sha256')}")
        return

    records = {}
    for section, kind in RECORD_SECTIONS:
        if kind is None:
            continue
        for record in bundle[section]:
            if _ident_ok(record.get("record_id")):
                records[record["record_id"]] = record
    node_by_id = {node["id"]: node for node in inst["nodes"]}

    required = required_targets(inst)
    bound = {}
    for binding in evidence["bindings"]:
        target = binding.get("target")
        if target in bound:
            _fail(failures, "E_SCHEMA", f"target {target} is bound twice")
            continue
        bound[target] = binding

    for target in sorted(set(required) - set(bound)):
        _fail(failures, "E_BINDING_MISSING",
              f"instance field {target} = {required[target]} is evidence-derived but "
              f"has no binding")
    for target in sorted(set(bound) - set(required)):
        _fail(failures, "E_BINDING_EXTRA",
              f"binding names {target}, which is not an evidence-derived field of "
              f"this instance")

    for target in sorted(set(required) & set(bound)):
        binding = bound[target]
        declared = required[target]
        record = records.get(binding.get("record_id"))
        if record is None:
            _fail(failures, "E_MISSING_RECORD",
                  f"binding for {target} names record "
                  f"{binding.get('record_id')!r} which is not in the bundle")
            continue
        if binding.get("record_sha256") != record.get("record_sha256"):
            _fail(failures, "E_HASH",
                  f"binding for {target} pins record digest "
                  f"{binding.get('record_sha256')} but record "
                  f"{record['record_id']} carries {record.get('record_sha256')}")
            continue

        kind, field, device = _expected_binding(target, inst)
        if kind is None:
            _fail(failures, "E_BINDING_EXTRA", f"target {target} is not recognised")
            continue
        if record.get("kind") != kind:
            _fail(failures, "E_IDENTITY",
                  f"{target} must bind a {kind} but record {record['record_id']} is a "
                  f"{record.get('kind')}")
            continue
        if binding.get("field") != field:
            _fail(failures, "E_BINDING_VALUE",
                  f"{target} must bind field {field} but the binding names "
                  f"{binding.get('field')!r}")
            continue

        # "Unknown phone energy is never zero" gets its own diagnostic, and it is
        # checked before the generic status gate so the message names the real
        # mistake rather than a bare status complaint.
        if record.get("status") == "UNKNOWN" and binding.get("value") == 0:
            _fail(failures, "E_UNKNOWN_AS_ZERO",
                  f"{target} is bound to UNKNOWN record {record['record_id']} "
                  f"(reason {record.get('reason_code')!r}) and encoded as 0; an "
                  f"unmeasured quantity is UNKNOWN, never zero")
            continue
        if record.get("status") != "PASS":
            _fail(failures, "E_STATUS",
                  f"{target} binds record {record['record_id']} with status "
                  f"{record.get('status')} (reason {record.get('reason_code')!r}); "
                  f"only PASS records are bindable")
            continue

        actual = _resolve_field(record, field)
        if actual is canon.MISSING:
            _fail(failures, "E_BINDING_VALUE",
                  f"{target} binds field {field} but record {record['record_id']} has "
                  f"no such field; an absent field evidences nothing")
            continue
        if not canon.is_int(binding.get("value")):
            _fail(failures, "E_BINDING_VALUE",
                  f"{target} has a non-integer bound value {binding.get('value')!r}")
            continue
        if actual != binding.get("value"):
            _fail(failures, "E_BINDING_VALUE",
                  f"{target} binds value {binding.get('value')!r} but record "
                  f"{record['record_id']}.{field} is {actual!r}")
            continue
        if declared != binding.get("value"):
            _fail(failures, "E_BINDING_VALUE",
                  f"{target} is {declared!r} in the instance but the binding claims "
                  f"{binding.get('value')!r}")
            continue

        if device is not None and record.get("device_id") != device:
            _fail(failures, "E_IDENTITY",
                  f"{target} concerns device {device} but record "
                  f"{record['record_id']} was measured on "
                  f"{record.get('device_id')!r}")
            continue

        _check_target_identity(target, inst, node_by_id, record, failures)

    _check_record_coherence(inst, records, bound, failures)
    _check_scope_consistency(inst, bundle, records, bound, failures)


PHONE_DIRECTIONS = frozenset({"HOST_TO_PHONE", "PHONE_TO_HOST"})
SERVER_DIRECTIONS = frozenset({"INTRA_HOST", "HOST_TO_GPU", "GPU_TO_HOST"})
# The activation bound may only bind a record explicitly designated as a capacity
# probe. Without this, ANY PASS route's memory_bytes would do -- and this is the
# one evidence-derived field where a LARGER value makes a schedule more feasible.
CAPACITY_LAYER_CLASS = "CAPACITY_PROBE"


def _check_target_identity(target, inst, node_by_id, record, failures):
    """A record may not be rebound to another model, island, graph, or shape."""
    parts = target.split(".")
    if target == "activation_mem_bound_bytes":
        if record.get("device_id") != "SERVER":
            _fail(failures, "E_IDENTITY",
                  f"activation_mem_bound_bytes binds route {record['record_id']} "
                  f"measured on {record.get('device_id')!r}; the activation bound is "
                  f"the SERVER resource limit and requires a SERVER-owned capacity "
                  f"probe")
        if record.get("layer_class") != CAPACITY_LAYER_CLASS:
            _fail(failures, "E_IDENTITY",
                  f"activation_mem_bound_bytes binds route {record['record_id']} with "
                  f"layer_class={record.get('layer_class')!r}; the activation bound may "
                  f"only bind a designated {CAPACITY_LAYER_CLASS} record, not an "
                  f"arbitrary route's footprint")
        return
    if parts[0] == "nodes":
        node = node_by_id.get(parts[1])
        if node is None:
            return
        if record["kind"] == "BoundaryProfile":
            for field in ("model_id", "weight_set_id", "graph_id", "island_id"):
                if record.get(field) != node.get(field):
                    _fail(failures, "E_IDENTITY",
                          f"{target} binds a boundary measured with {field}="
                          f"{record.get(field)!r} to a node with {field}="
                          f"{node.get(field)!r}")
            if parts[-1] == "extra_energy_nj":
                device = parts[3]
                if record.get("device_id") != device:
                    _fail(failures, "E_IDENTITY",
                          f"{target} is the transfer energy to {device} but binds a "
                          f"boundary measured against {record.get('device_id')!r}")
                kind = inst["devices"].get(device, {}).get("kind")
                allowed = PHONE_DIRECTIONS if kind == "phone" else SERVER_DIRECTIONS
                if record.get("direction") not in allowed:
                    _fail(failures, "E_IDENTITY",
                          f"{target} routes to a {kind} device but binds a boundary "
                          f"measured in direction {record.get('direction')!r}; a "
                          f"transfer across one link is not evidence for another")
        if record["kind"] == "RouteProfile":
            for field in ("model_id", "weight_set_id", "graph_id", "island_id"):
                if record.get(field) != node.get(field):
                    _fail(failures, "E_IDENTITY",
                          f"{target} binds a route measured with {field}="
                          f"{record.get(field)!r} to a node with {field}="
                          f"{node.get(field)!r}")
            envelope = record.get("shape_envelope") or {}
            tokens = node.get("tokens")
            kv_tokens = node.get("kv_tokens")
            token_lo, token_hi = envelope.get("tokens_min"), envelope.get("tokens_max")
            kv_lo, kv_hi = envelope.get("kv_min"), envelope.get("kv_max")
            if not all(canon.is_int(value) for value in
                       (tokens, kv_tokens, token_lo, token_hi, kv_lo, kv_hi)):
                _fail(failures, "E_ENVELOPE",
                      f"{target} has an incomplete token/KV shape")
            else:
                if not (token_lo <= tokens <= token_hi):
                    _fail(failures, "E_ENVELOPE",
                          f"{target} has tokens={tokens} outside the record envelope "
                          f"[{token_lo},{token_hi}]")
                if not (kv_lo <= kv_tokens <= kv_hi):
                    _fail(failures, "E_ENVELOPE",
                          f"{target} has kv_tokens={kv_tokens} outside the record "
                          f"envelope [{kv_lo},{kv_hi}]")
            if record.get("batch_size") != tokens:
                _fail(failures, "E_ENVELOPE",
                      f"{target} has tokens={tokens} but binds a route measured at "
                      f"batch_size={record.get('batch_size')!r}")
    if parts[0] == "batch_profiles":
        key, size = parts[1], parts[2]
        if not size.isdigit():
            _fail(failures, "E_SCHEMA",
                  f"batch profile size {size!r} in {target} is not a decimal integer")
            return
        if record.get("batch_size") != int(size):
            _fail(failures, "E_ENVELOPE",
                  f"{target} is the duration at batch size {size} but binds a route "
                  f"measured at batch_size={record.get('batch_size')!r}")
        envelope = record.get("shape_envelope") or {}
        token_count = int(size)
        if not (canon.is_int(envelope.get("tokens_min")) and
                canon.is_int(envelope.get("tokens_max")) and
                envelope["tokens_min"] <= token_count <= envelope["tokens_max"]):
            _fail(failures, "E_ENVELOPE",
                  f"{target} has aggregate tokens={token_count} outside the route "
                  f"shape envelope")
        members = [node for node in inst["nodes"] if node.get("batch_key") == key]
        for node in members:
            for field in ("model_id", "weight_set_id", "graph_id", "island_id"):
                if record.get(field) != node.get(field):
                    _fail(failures, "E_IDENTITY",
                          f"{target} binds a route with {field}={record.get(field)!r} "
                          f"but batch member {node['id']} has {field}="
                          f"{node.get(field)!r}")
            kv_tokens = node.get("kv_tokens")
            if not (canon.is_int(kv_tokens) and
                    canon.is_int(envelope.get("kv_min")) and
                    canon.is_int(envelope.get("kv_max")) and
                    envelope["kv_min"] <= kv_tokens <= envelope["kv_max"]):
                _fail(failures, "E_ENVELOPE",
                      f"{target} uses member {node['id']} with kv_tokens="
                      f"{kv_tokens!r} outside the route shape envelope")


def _check_record_coherence(inst, records, bound, failures):
    """One physical thing, one record.

    The contract froze `duration_us := latency.p95_us` because a selectable
    statistic lets a binder pick whichever number suits it. Selectable RECORDS are
    the same fail-open surface one level up: with per-field freedom a binder can
    assemble a machine that no record in the bundle describes -- taking the wake
    ramp from one profile and the transition energy from another, or a COLD route
    for one node and a STEADY route for another in the same schedule.
    """
    # 1. All power fields of one device come from ONE PowerProfile.
    power_by_device = defaultdict(dict)
    for target, binding in sorted(bound.items()):
        record = records.get(binding.get("record_id"))
        if record is None or record.get("kind") != "PowerProfile":
            continue
        _kind, _field, device = _expected_binding(target, inst)
        if device is None:
            continue
        power_by_device[device][target] = binding.get("record_id")
    for device, targets in sorted(power_by_device.items()):
        chosen = sorted(set(targets.values()))
        if len(chosen) > 1:
            detail = ", ".join(f"{t} -> {r}" for t, r in sorted(targets.items()))
            _fail(failures, "E_INCOHERENT",
                  f"device {device} draws its power fields from {len(chosen)} different "
                  f"records {chosen}: {detail}. One device is one measurement; mixing "
                  f"fields describes a machine that no record in the bundle describes")

    # 2. All routes bound for one device share ONE thermal condition and ONE build.
    thermal_by_device = defaultdict(dict)
    build_by_device = defaultdict(dict)
    for target, binding in sorted(bound.items()):
        record = records.get(binding.get("record_id"))
        if record is None or record.get("kind") != "RouteProfile":
            continue
        device = record.get("device_id")
        thermal_by_device[device][target] = record.get("thermal_id")
        build_by_device[device][target] = record.get("backend_build")
    for device, targets in sorted(thermal_by_device.items()):
        chosen = sorted({t for t in targets.values() if t is not None})
        if len(chosen) > 1:
            detail = ", ".join(f"{t} -> {r}" for t, r in sorted(targets.items()))
            _fail(failures, "E_THERMAL",
                  f"device {device} binds routes measured under {len(chosen)} different "
                  f"thermal conditions {chosen}: {detail}. A device is in one thermal "
                  f"state during one schedule, so cherry-picking a cold route for one "
                  f"node and a steady route for another is not a schedule that can run")
    for device, targets in sorted(build_by_device.items()):
        chosen = sorted({b for b in targets.values() if b is not None})
        if len(chosen) > 1:
            _fail(failures, "E_STALE",
                  f"device {device} binds routes from {len(chosen)} different backend "
                  f"builds {chosen}; no single binary produces that schedule")

    # 3. A RouteProfile pins its exact BoundaryProfile. Duration and transfer
    # energy cannot select unrelated records with different contention or byte
    # geometry. output_bytes must agree with every route's pinned boundary.
    for node in inst["nodes"]:
        nid = node["id"]
        route_boundary_ids = set()
        for device in sorted(node["routes"]):
            duration_target = f"nodes.{nid}.routes.{device}.duration_us"
            energy_target = f"nodes.{nid}.routes.{device}.extra_energy_nj"
            route_binding = bound.get(duration_target) or {}
            route_record = records.get(route_binding.get("record_id"))
            if route_record is None or route_record.get("kind") != "RouteProfile":
                continue
            boundary_id = route_record.get("boundary_id")
            route_boundary_ids.add(boundary_id)
            energy_binding = bound.get(energy_target) or {}
            if energy_binding.get("record_id") != boundary_id:
                _fail(failures, "E_INCOHERENT",
                      f"{duration_target} pins boundary {boundary_id!r}, but "
                      f"{energy_target} selects {energy_binding.get('record_id')!r}; "
                      f"latency and boundary cost must describe one measured route")
            boundary_record = records.get(boundary_id)
            if boundary_record is not None and \
                    boundary_record.get("output_bytes") != node.get("output_bytes"):
                _fail(failures, "E_INCOHERENT",
                      f"route {route_record['record_id']} pins boundary {boundary_id} "
                      f"with output_bytes={boundary_record.get('output_bytes')}, "
                      f"but node {nid} declares output_bytes={node.get('output_bytes')}")
        output_target = f"nodes.{nid}.output_bytes"
        output_binding = bound.get(output_target) or {}
        if route_boundary_ids and output_binding.get("record_id") not in route_boundary_ids:
            _fail(failures, "E_INCOHERENT",
                  f"{output_target} selects boundary {output_binding.get('record_id')!r}, "
                  f"which is not pinned by any route of node {nid}: "
                  f"{sorted(route_boundary_ids)}")

    # 4. A batch profile can time any legal SERVER subset with the same key and
    # aggregate token count. One profile is bindable only when all such subsets
    # have the same aggregate output geometry. The frozen solver has no separate
    # batch-boundary energy term, so that boundary must also carry zero energy.
    for target, binding in sorted(bound.items()):
        parts = target.split(".")
        if len(parts) != 3 or parts[0] != "batch_profiles" or \
                not parts[2].isdigit():
            continue
        key = parts[1]
        token_count = int(parts[2])
        route_record = records.get(binding.get("record_id"))
        if route_record is None or route_record.get("kind") != "RouteProfile":
            continue
        boundary_id = route_record.get("boundary_id")
        boundary_record = records.get(boundary_id)
        if boundary_record is None or boundary_record.get("kind") != "BoundaryProfile":
            continue
        members = [node for node in inst["nodes"]
                   if node.get("batch_key") == key and "SERVER" in node.get("routes", {})]
        output_sums = set()
        for mask in range(1, 1 << len(members)):
            subset = [members[index] for index in range(len(members))
                      if mask & (1 << index)]
            if sum(node["tokens"] for node in subset) != token_count:
                continue
            subset_ids = {node["id"] for node in subset}
            if any(subset_ids.intersection(node["predecessors"]) for node in subset):
                continue
            identities = {(node["model_id"], node["weight_set_id"])
                          for node in subset}
            if len(identities) != 1:
                continue
            output_sums.add(sum(node["output_bytes"] for node in subset))
        if not output_sums:
            _fail(failures, "E_ENVELOPE",
                  f"{target} has no legal SERVER batch with aggregate tokens="
                  f"{token_count}")
        elif len(output_sums) != 1:
            _fail(failures, "E_INCOHERENT",
                  f"{target} can describe legal batches with different aggregate "
                  f"output sizes {sorted(output_sums)}; split them into distinct "
                  f"batch keys")
        elif boundary_record.get("output_bytes") != next(iter(output_sums)):
            _fail(failures, "E_INCOHERENT",
                  f"{target} pins boundary {boundary_id} with output_bytes="
                  f"{boundary_record.get('output_bytes')}, but its legal batch "
                  f"geometry produces {next(iter(output_sums))} bytes")
        if boundary_record.get("direction") not in SERVER_DIRECTIONS:
            _fail(failures, "E_IDENTITY",
                  f"{target} is a SERVER batch but pins boundary {boundary_id} "
                  f"with direction {boundary_record.get('direction')!r}")
        if boundary_record.get("energy_nj") != 0:
            _fail(failures, "E_INCOHERENT",
                  f"{target} pins boundary {boundary_id} with energy_nj="
                  f"{boundary_record.get('energy_nj')}; the frozen solver has no "
                  f"batch-boundary energy term")


def _check_scope_consistency(inst, bundle, records, bound, failures):
    scope = inst["evidence"]["scope"]
    if scope == "MEASURED":
        _fail(failures, "E_SCOPE",
              "E1 does not authorize physical energy claims: its solver uses "
              "additive per-device power, while the available GPU_BOARD, "
              "SERVER_WALL, and TOTAL_WALL records do not provide a complete "
              "typed additive accounting model. Use a later matched comparison")
    if scope == "MEASURED" and bundle.get("provenance") != "MEASURED":
        _fail(failures, "E_PROVENANCE",
              f"instance claims MEASURED evidence but the bundle is "
              f"{bundle.get('provenance')}; a synthetic bundle cannot support a "
              f"physical claim")
    artifacts = {a.get("artifact_id"): a for a in bundle["artifacts"]}
    if scope == "MEASURED":
        for binding in bound.values():
            record = records.get(binding.get("record_id"))
            if record is None:
                continue
            for aid in record.get("artifacts", []):
                artifact = artifacts.get(aid)
                if artifact is None:
                    continue
                if artifact.get("provenance") != "MEASURED":
                    _fail(failures, "E_PROVENANCE",
                          f"MEASURED instance binds record {record['record_id']} "
                          f"backed by {artifact.get('provenance')} artifact {aid}")
            if record.get("kind") == "PowerProfile" and \
                    record.get("instrument_kind") == "SYNTHETIC":
                _fail(failures, "E_PROVENANCE",
                      f"MEASURED instance binds synthetic instrument record "
                      f"{record['record_id']}")

    # The solver adds one power term per device. TOTAL_WALL is one whole-system
    # timeline and is therefore non-additive: binding it once per device counts
    # the same wall rail repeatedly. A system-energy claim needs a separate
    # matched control/treatment comparison record, which E1 does not yet define.
    bound_power_ids = set()
    bound_boundary_ids = set()
    for target, binding in sorted(bound.items()):
        record = records.get(binding.get("record_id"))
        if record is None:
            continue
        if record.get("kind") == "PowerProfile":
            bound_power_ids.add(record["record_id"])
        elif record.get("kind") == "BoundaryProfile" and \
                target.endswith("extra_energy_nj"):
            bound_boundary_ids.add(record["record_id"])
    for rid in sorted(bound_power_ids):
        record = records[rid]
        if record.get("scope") in ("SERVER_WALL", "TOTAL_WALL") and scope == "MEASURED":
            _fail(failures, "E_SCOPE",
                  f"bound power record {rid} is aggregate scope "
                  f"{record.get('scope')}, which cannot be used as an additive "
                  f"per-device solver term")
        expected_model = "SERVER_PSTATE" if record.get("device_id") == "SERVER" \
            else "ACTIVE_ONLY"
        if record.get("accounting_model") != expected_model:
            _fail(failures, "E_INCOHERENT",
                  f"power record {rid} uses accounting model "
                  f"{record.get('accounting_model')!r}; device "
                  f"{record.get('device_id')} requires {expected_model}")
        if expected_model == "ACTIVE_ONLY" and any(
                record.get(field) != 0 for field in
                ("idle_mw", "wake_us", "idle_entry_us", "transition_nj")):
            _fail(failures, "E_INCOHERENT",
                  f"ACTIVE_ONLY power record {rid} carries omitted idle/transition "
                  f"costs; the v2 solver has nowhere to account for them")
        if record.get("uncertainty_mw") != 0:
            _fail(failures, "E_INCOHERENT",
                  f"bound power record {rid} has uncertainty_mw="
                  f"{record.get('uncertainty_mw')}; the exact solver has no "
                  f"uncertainty term")
    for rid in sorted(bound_boundary_ids):
        if records[rid].get("energy_scope") == "TOTAL_WALL":
            _fail(failures, "E_SCOPE",
                  f"bound boundary record {rid} is TOTAL_WALL, which cannot be "
                  f"added to per-device energy")

    # Additive component records must name globally disjoint rails. Repeated
    # bindings to the same record are deduplicated before this check.
    rail_owner = {}
    for rid in sorted(bound_power_ids):
        record = records[rid]
        for rail in record.get("included_rails") or []:
            previous = rail_owner.get(rail)
            if previous is not None and previous != rid:
                _fail(failures, "E_DOUBLE_COUNT",
                      f"bound power records {previous} and {rid} both include rail "
                      f"{rail}; additive power terms must have disjoint rail sets")
            rail_owner[rail] = rid

    # A MEASURED instance may not take an energy term from a record that names no
    # boundary. extra_energy_nj is a real energy term (it flows into
    # energy.total_nj), so an unmeasured zero there is the same lie as an unknown
    # phone power encoded as zero -- just through a different door.
    if scope == "MEASURED":
        for target, binding in sorted(bound.items()):
            if not target.endswith("extra_energy_nj"):
                continue
            record = records.get(binding.get("record_id"))
            if record is None or record.get("kind") != "BoundaryProfile":
                continue
            if record.get("energy_scope") == "NONE":
                _fail(failures, "E_UNKNOWN_AS_ZERO",
                      f"{target} takes transfer energy from boundary "
                      f"{record['record_id']}, which declares energy_scope NONE; energy "
                      f"measured at no boundary is unknown, not {binding.get('value')}")

    # USB VBUS double counting (contract rule 4). Rail names are per-device, so a
    # phone's own CPU rail is NOT the server's CPU rail and a bare name collision
    # means nothing. The real hazard is specific: when phones are powered through
    # the metered server wall, the phone's energy is ALREADY inside the server
    # number, and adding a separate phone profile counts it twice.
    #
    # Scan EVERY bound server power record, not just devices.SERVER.active_mw:
    # server_power.p0_mw is the field the server energy term actually multiplies,
    # and a twin record could otherwise hide the rail. (Coherence now forces these
    # to be one record, but this stays independent of that.)
    server_meters_phone_supply = False
    server_record = None
    for target, binding in sorted(bound.items()):
        _kind, _field, device = _expected_binding(target, inst)
        if device != "SERVER":
            continue
        record = records.get(binding.get("record_id"))
        if record is None or record.get("kind") != "PowerProfile":
            continue
        if record.get("scope") in ("SERVER_WALL", "TOTAL_WALL") and \
                "SERVER/USB_VBUS" in set(record.get("included_rails") or []):
            server_meters_phone_supply = True
            server_record = record
    if server_meters_phone_supply:
        for target, binding in sorted(bound.items()):
            if not target.startswith("devices.") or not target.endswith(".active_mw"):
                continue
            device = target.split(".")[1]
            if device == "SERVER" or \
                    inst["devices"].get(device, {}).get("kind") != "phone":
                continue
            if binding.get("value"):
                _fail(failures, "E_DOUBLE_COUNT",
                      f"the server boundary record {server_record['record_id']} "
                      f"includes rail SERVER/USB_VBUS, so phone {device} is already "
                      f"inside "
                      f"the metered server wall; binding {device} a further "
                      f"{binding['value']} mW counts that energy twice")


def validate(inst, bundle, artifact_root=DEFAULT_ARTIFACT_ROOT):
    """Full validation: bundle, instance shape, binding, and scope. Fails closed."""
    failures = list(validate_bundle(bundle, artifact_root))
    if not validate_instance_shape(inst, failures):
        return failures
    validate_binding(inst, bundle, failures)
    return failures


def validate_safe(inst, bundle, artifact_root=DEFAULT_ARTIFACT_ROOT):
    """validate(), but an internal error becomes a failure, never a traceback.

    A crash is not a rejection: a caller that catches only EvidenceError would see
    an unhandled exception, and an unhandled path is where fail-open bugs hide.
    """
    try:
        return validate(inst, bundle, artifact_root)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError,
            RecursionError) as exc:
        return [f"E_SCHEMA: validator could not process the input: "
                f"{type(exc).__name__}: {exc}"]


def validate_certificate(cert):
    """Validate a live certificate before any checker field is indexed."""
    failures = []
    if not isinstance(cert, dict):
        return ["E_SCHEMA: certificate is not an object"]
    if cert.get("schema_version") != 3:
        return [f"E_VERSION: certificate schema_version "
                f"{cert.get('schema_version')!r} is not 3"]
    try:
        canon.check_integers(cert, "certificate",
                             frozenset({"certificate.search.complete"}))
    except (TypeError, ValueError, RecursionError) as exc:
        _fail(failures, "E_SCHEMA", str(exc))
        return failures
    _check_schema("certificate", cert, failures)
    return failures


def binding_digest(inst):
    return canon.digest(inst["evidence"]["bindings"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="validate an S10-V0-R-E1 bundle")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--instance")
    parser.add_argument("--artifact-root",
                        help="trusted root for relative artifact paths; defaults to "
                             "the bundle directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        bundle = canon.load_strict(args.bundle)
        inst = canon.load_strict(args.instance) if args.instance else None
    except (OSError, ValueError, UnicodeDecodeError, RecursionError) as exc:
        print(f"EVIDENCE_FAIL: cannot load input: {exc}", file=sys.stderr)
        return 2

    try:
        artifact_root = pathlib.Path(args.artifact_root) if args.artifact_root \
            else pathlib.Path(args.bundle).resolve().parent
        failures = validate_safe(inst, bundle, artifact_root) if inst is not None \
            else validate_bundle(bundle, artifact_root)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError,
            RecursionError) as exc:
        print(f"EVIDENCE_FAIL: validation error: {exc}", file=sys.stderr)
        return 2

    if failures:
        for failure in failures:
            print(f"EVIDENCE_FAIL: {failure}", file=sys.stderr)
        return 1
    if not args.quiet:
        import json as _json
        out = {"bundle_id": bundle["bundle_id"],
               "bundle_sha256": bundle["bundle_sha256"],
               "provenance": bundle["provenance"],
               "records": sum(len(bundle[s]) for s, k in RECORD_SECTIONS if k),
               "valid": True}
        if inst is not None:
            out["instance_id"] = inst["instance_id"]
            out["evidence_scope"] = inst["evidence"]["scope"]
            out["binding_sha256"] = binding_digest(inst)
            out["bound_targets"] = len(required_targets(inst))
        print(_json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
