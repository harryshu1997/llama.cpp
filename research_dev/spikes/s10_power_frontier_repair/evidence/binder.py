#!/usr/bin/env python3
"""Deterministic evidence binder and the v3 evidence-bound oracle/checker path.

The v3 instance is a v2 mechanics core plus an evidence binding block. To reuse
the PROVEN temporal solver without touching it, the binder projects a validated
v3 instance to its v2 mechanics core IN MEMORY ONLY.

The projection is never the signed identity. The v3 certificate binds:

  - instance_sha256   = SHA-256 of the FULL v3 instance (bindings included), and
  - evidence.bundle_sha256 / evidence.binding_sha256.

So the bundle digest and every derived field stay checker-visible, and swapping
in a different v2 projection cannot validate: the projection is recomputed from
the v3 instance the certificate names, and any edit changes instance_sha256.

The oracle and checker consume the same validated binding through this module,
but share no optimality-search code: the oracle searches via oracle/exact.py and
the checker proves optimality via the independently written checker/reference.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "oracle"), str(ROOT / "checker"), str(ROOT / "evidence")]

import boundary  # noqa: E402
import canon  # noqa: E402
import validator  # noqa: E402

# v2 mechanics node fields. v3 adds graph_id and island_id, which are evidence
# identity, not mechanics, and are therefore projected out.
V2_NODE_FIELDS = ("id", "request_id", "model_id", "weight_set_id", "predecessors",
                  "release_us", "routes", "batch_key", "tokens", "output_bytes")


class EvidenceError(ValueError):
    """A binding or bundle failure. Always fatal; never downgraded to a warning."""


def project_v2(inst):
    """Deterministically project a v3 instance onto its v2 mechanics core.

    In-memory only. Callers must never sign or persist this as the instance
    identity; the v3 instance is the identity.
    """
    if inst.get("schema_version") != 3:
        raise EvidenceError("only a v3 instance can be projected to v2")
    core = {
        "schema_version": 2,
        "instance_id": inst["instance_id"],
        "horizon_us": inst["horizon_us"],
        "activation_mem_bound_bytes": inst["activation_mem_bound_bytes"],
        "server_power": dict(inst["server_power"]),
        "devices": {name: dict(device) for name, device in inst["devices"].items()},
        "batch_profiles": {key: dict(profile)
                           for key, profile in inst["batch_profiles"].items()},
        "requests": [dict(request) for request in inst["requests"]],
        "nodes": [{field: node[field] for field in V2_NODE_FIELDS}
                  for node in inst["nodes"]],
        "evidence": {"scope": "MECHANICS_ONLY"},
    }
    return core


def _records_of(bundle):
    records = {}
    for section, kind in validator.RECORD_SECTIONS:
        if kind is None:
            continue
        for record in bundle[section]:
            rid = record.get("record_id")
            if isinstance(rid, str) and rid:
                records[rid] = record
    return records


def bind(inst, bundle, artifact_root=validator.DEFAULT_ARTIFACT_ROOT):
    """Validate a v3 instance against its bundle. Raise on ANY failure.

    Returns (v2_core, binding_sha256, energy_claim). There is no partial success
    and no best-effort mode: an unproven number never reaches the solver.
    """
    # validate_safe, not validate: an internal error must become a refusal, not a
    # traceback out of the binder. A caller catching EvidenceError would otherwise
    # see an unhandled exception, and an unhandled path is not a rejection.
    failures = validator.validate_safe(inst, bundle, artifact_root)
    if failures:
        raise EvidenceError("evidence binding failed:\n  " + "\n  ".join(failures))

    records = _records_of(bundle)
    bound = {binding["target"]: binding for binding in inst["evidence"]["bindings"]}
    claim, reasons = boundary.classify_energy_claim(inst, bundle, records, bound)
    core = project_v2(inst)

    # Defence in depth: re-derive the projected values straight from the bound
    # records and require equality. validate() already proved this, but the
    # projection is the thing the solver actually reads, so it is re-checked
    # against the evidence rather than trusted.
    projected = validator.required_targets(core)
    declared = validator.required_targets(inst)
    if projected != declared:
        raise EvidenceError(
            "projection changed an evidence-derived field: "
            f"{sorted(k for k in set(projected) | set(declared) if projected.get(k) != declared.get(k))}")
    for target, value in sorted(projected.items()):
        binding = bound.get(target)
        actual = binding.get("value") if binding else None
        if actual != value:
            raise EvidenceError(
                f"projected field {target}={value!r} is not the bound value {actual!r}")
    return core, validator.binding_digest(inst), claim, reasons


def load_bound(instance_path, bundle_path, artifact_root=None):
    inst = canon.load_strict(instance_path)
    bundle = canon.load_strict(bundle_path)
    root = artifact_root or pathlib.Path(bundle_path).resolve().parent
    core, binding_sha, claim, reasons = bind(inst, bundle, root)
    return inst, bundle, core, binding_sha, claim, reasons


def solve_bound(inst, bundle, max_states=2_000_000,
                artifact_root=validator.DEFAULT_ARTIFACT_ROOT):
    """Solve a v3 evidence-bound instance and emit a v3 certificate."""
    import exact

    core, binding_sha, claim, _reasons = bind(inst, bundle, artifact_root)
    cert = exact.solve(core, max_states)
    # The v2 projection's own digest is discarded on purpose. The signed identity
    # is the full v3 instance, so a swapped projection cannot validate.
    out = {
        "schema_version": 3,
        "instance_id": inst["instance_id"],
        "instance_sha256": canon.digest(inst),
        "evidence": {
            "scope": inst["evidence"]["scope"],
            "bundle_sha256": bundle["bundle_sha256"],
            "binding_sha256": binding_sha,
            "energy_claim": claim,
        },
        "actions": cert["actions"],
        "request_outcomes": cert["request_outcomes"],
        "activation_peak_bytes": cert["activation_peak_bytes"],
        "energy": cert["energy"],
        "objective": cert["objective"],
        "search": {"complete": True},
    }
    out["certificate_sha256"] = canon.digest(out)
    cert_failures = validator.validate_certificate(out)
    if cert_failures:
        raise EvidenceError("generated certificate violates the v3 schema:\n  " +
                            "\n  ".join(cert_failures))
    return out


def check_bound(inst, bundle, cert, feasibility_only=False,
                artifact_root=validator.DEFAULT_ARTIFACT_ROOT):
    """Re-validate the binding and re-check a v3 certificate. Returns failures."""
    failures = validator.validate_certificate(cert)
    if failures:
        return failures

    import checker

    try:
        core, binding_sha, claim, _reasons = bind(inst, bundle, artifact_root)
    except EvidenceError as exc:
        return [f"E_BINDING_VALUE: {exc}"]

    # Bind the certificate to the FULL v3 instance, not to the projection.
    expected_instance_sha = canon.digest(inst)
    if cert.get("instance_sha256") != expected_instance_sha:
        failures.append(
            f"E_HASH: certificate names instance {cert.get('instance_sha256')} but the "
            f"given v3 instance hashes to {expected_instance_sha}; a certificate for a "
            f"different (for example projected or re-bound) instance is not valid here")
        return failures

    evidence = cert.get("evidence") or {}
    if evidence.get("bundle_sha256") != bundle["bundle_sha256"]:
        failures.append(
            f"E_HASH: certificate names bundle {evidence.get('bundle_sha256')} but the "
            f"given bundle hashes to {bundle['bundle_sha256']}")
    if evidence.get("binding_sha256") != binding_sha:
        failures.append(
            f"E_HASH: certificate names binding {evidence.get('binding_sha256')} but "
            f"the instance binding hashes to {binding_sha}")
    if evidence.get("scope") != inst["evidence"]["scope"]:
        failures.append(
            f"E_SCHEMA: certificate scope {evidence.get('scope')!r} differs from "
            f"instance scope {inst['evidence']['scope']!r}")
    if evidence.get("energy_claim") != claim:
        failures.append(
            f"E_SCOPE: certificate claims {evidence.get('energy_claim')!r} but the "
            f"bound power evidence supports at most {claim!r}")
    if failures:
        return failures

    # Re-verify the certificate's own digest before trusting any field of it.
    body = {key: value for key, value in cert.items() if key != "certificate_sha256"}
    if canon.digest(body) != cert.get("certificate_sha256"):
        return [f"E_HASH: certificate_sha256 does not cover the certificate body"]

    # Mechanics: hand the projection and a v2-shaped certificate to the frozen,
    # independently verified checker. Optimality comes from checker/reference.py.
    v2_cert = {
        "schema_version": 2,
        "instance_id": cert["instance_id"],
        "instance_sha256": canon.digest(core),
        "actions": cert["actions"],
        "request_outcomes": cert["request_outcomes"],
        "activation_peak_bytes": cert["activation_peak_bytes"],
        "energy": cert["energy"],
        "objective": cert["objective"],
        "search": {"complete": True},
    }
    v2_cert["certificate_sha256"] = canon.digest(v2_cert)
    failures.extend(checker.check(core, v2_cert, require_complete=not feasibility_only))
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="bind, solve, or check an evidence-bound v3 instance")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--certificate")
    parser.add_argument("--artifact-root",
                        help="trusted root for relative artifact paths; defaults to "
                             "the bundle directory")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--feasibility-only", action="store_true")
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        inst = canon.load_strict(args.instance)
        bundle = canon.load_strict(args.bundle)
    except (OSError, ValueError, UnicodeDecodeError, RecursionError) as exc:
        print(f"BIND_FAIL: cannot load input: {exc}", file=sys.stderr)
        return 2

    try:
        artifact_root = pathlib.Path(args.artifact_root) if args.artifact_root \
            else pathlib.Path(args.bundle).resolve().parent
        if args.solve:
            cert = solve_bound(inst, bundle, args.max_states, artifact_root)
            output = json.dumps(cert, indent=2, sort_keys=True) + "\n"
            if args.out:
                with open(args.out, "w", encoding="ascii") as handle:
                    handle.write(output)
            if not args.quiet:
                print(json.dumps({"instance_id": cert["instance_id"],
                                  "objective": cert["objective"],
                                  "evidence_scope": cert["evidence"]["scope"],
                                  "energy_claim": cert["evidence"]["energy_claim"],
                                  "certificate_sha256": cert["certificate_sha256"]},
                                 sort_keys=True))
            return 0
        if args.certificate:
            cert = canon.load_strict(args.certificate)
            failures = check_bound(inst, bundle, cert, args.feasibility_only,
                                   artifact_root)
            if failures:
                for failure in failures:
                    print(f"BIND_FAIL: {failure}", file=sys.stderr)
                return 1
            if not args.quiet:
                print(json.dumps({"instance_id": cert["instance_id"],
                                  "objective": cert["objective"],
                                  "energy_claim": cert["evidence"]["energy_claim"],
                                  "mode": "feasibility_only" if args.feasibility_only
                                          else "exact_certificate",
                                  "optimality_verified": not args.feasibility_only},
                                 sort_keys=True))
            return 0
        core, binding_sha, claim, reasons = load_bound(
            args.instance, args.bundle, artifact_root)[2:]
        if not args.quiet:
            print(json.dumps({"instance_id": inst["instance_id"],
                              "evidence_scope": inst["evidence"]["scope"],
                              "binding_sha256": binding_sha,
                              "energy_claim": claim,
                              "reasons": reasons,
                              "bound_targets": len(validator.required_targets(inst))},
                             sort_keys=True))
        return 0
    except EvidenceError as exc:
        print(f"BIND_FAIL: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, TypeError, KeyError,
            json.JSONDecodeError, RecursionError) as exc:
        print(f"BIND_FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
