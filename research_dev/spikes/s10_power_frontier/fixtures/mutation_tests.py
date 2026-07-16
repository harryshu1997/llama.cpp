#!/usr/bin/env python3
"""S10-V0 CP3 adversarial mutation fixtures. Take a VALID oracle certificate and
apply each PLAN-listed illegal mutation; the INDEPENDENT checker must reject every
one (nonzero). Each mutated body has its certificate_sha256 RECOMPUTED so the
checker catches the SUBSTANTIVE physics/constraint violation, not merely a hash
mismatch (two dedicated integrity mutations cover the hash path). Exits nonzero if
any mutation slips through (fail-closed self-test).
"""
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SP, "oracle"))
sys.path.insert(0, os.path.join(SP, "checker"))
import model_data as M      # noqa: E402
import oracle as ORACLE     # noqa: E402
import checker as CHK       # noqa: E402


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha_hex(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def reseal(cert):
    body = {k: cert[k] for k in cert if k != "certificate_sha256"}
    cert["certificate_sha256"] = sha_hex(body)
    return cert


def base_cert_and_inst():
    inst = M.load_instance(os.path.join(SP, "fixtures", "frozen", "primary_favorable.json"))
    placement, batch_split, sim, _, _ = ORACLE.solve(inst)
    cert = ORACLE.build_certificate(inst, placement, batch_split, sim, "C4_oracle")
    assert not CHK.check(inst, cert), "base cert must be valid"
    return inst, cert


def find_phone_req(cert):
    for s in cert["schedule"]:
        if s["mid_device"] != "SERVER":
            return s["request"]
    return None


def find_server_req(cert):
    for s in cert["schedule"]:
        if s["mid_device"] == "SERVER":
            return s["request"]
    return None


MUTATIONS = []


def mut(name):
    def deco(fn):
        MUTATIONS.append((name, fn))
        return fn
    return deco


@mut("removed_dependency_suf_before_mid")
def m1(inst, cert):
    rid = cert["schedule"][0]["request"]
    for s in cert["schedule"]:
        if s["request"] == rid:
            s["suf_start"] = s["mid_finish"] - 1
            s["suf_finish"] = s["suf_start"] + inst["suf_us"]
    return inst, reseal(cert)


@mut("execution_before_ready_mid_before_pre")
def m2(inst, cert):
    rid = find_phone_req(cert) or cert["schedule"][0]["request"]
    for s in cert["schedule"]:
        if s["request"] == rid:
            s["mid_start"] = s["pre_finish"] - 1
    return inst, reseal(cert)


@mut("omitted_d2h_transfer_time")
def m3(inst, cert):
    rid = find_phone_req(cert)
    if rid is None:
        # if no phone offload, shrink a server batch latency instead (omit compute)
        cert["server_batches"][0]["latency_us"] -= 1
        cert["server_batches"][0]["finish"] -= 1
        return inst, reseal(cert)
    for s in cert["schedule"]:
        if s["request"] == rid:
            s["mid_finish"] = s["mid_start"] + 1   # claim near-instant phone island (skipped transfer)
    return inst, reseal(cert)


@mut("phone_energy_double_counted")
def m4(inst, cert):
    cert["energy_phone_nj"] = cert["energy_phone_nj"] * 2 + 1
    cert["energy_nj"] = cert["energy_server_nj"] + cert["energy_phone_nj"]
    return inst, reseal(cert)


@mut("hbm_credit_in_mirrored_mode")
def m5(inst, cert):
    cert["hbm"] = {"mode": {inst["weight_sets"][0]: "mirrored"},
                   "relief_bytes": inst["atlas_provenance"]["weight_bytes_f16"]}
    return inst, reseal(cert)


@mut("free_instant_server_state_transition")
def m6(inst, cert):
    cert["energy_server_nj"] = cert["energy_server_nj"] // 2   # claim a free lower-power transition
    cert["energy_nj"] = cert["energy_server_nj"] + cert["energy_phone_nj"]
    return inst, reseal(cert)


@mut("server_claim_after_latest_start")
def m7(inst, cert):
    # shift a server batch far past its deadline but keep the outcome MET
    b = cert["server_batches"][0]
    late = inst["horizon_us"]  # start at horizon -> suffix misses deadline
    b["start"] = late
    b["finish"] = late + b["latency_us"]
    for s in cert["schedule"]:
        if s["request"] in b["members"]:
            s["mid_start"] = b["start"]
            s["mid_finish"] = b["finish"]
            s["suf_start"] = b["finish"]
            s["suf_finish"] = b["finish"] + inst["suf_us"]
            # keep the (now false) MET claim
    return inst, reseal(cert)


@mut("activation_memory_overflow")
def m8(inst, cert):
    inst2 = copy.deepcopy(inst)
    inst2["activation_mem_bound_bytes"] = 100   # tiny bound; islands (240 KB) overflow
    cert["instance_sha256"] = sha_hex(inst2)
    return inst2, reseal(cert)


@mut("duplicate_completion")
def m9(inst, cert):
    cert["schedule"].append(copy.deepcopy(cert["schedule"][0]))
    return inst, reseal(cert)


@mut("unfinished_work_omitted_at_horizon")
def m10(inst, cert):
    cert["schedule"] = cert["schedule"][:-1]   # drop a request (no terminal outcome)
    return inst, reseal(cert)


@mut("timeout_relabeled_met")
def m11(inst, cert):
    s = cert["schedule"][0]
    s["suf_finish"] = inst["horizon_us"] + 10_000   # past horizon
    s["suf_start"] = s["suf_finish"] - inst["suf_us"]
    s["outcome"] = "MET"                              # false claim
    return inst, reseal(cert)


@mut("integrity_body_tamper_without_reseal")
def m12(inst, cert):
    cert["energy_nj"] += 1     # do NOT reseal -> certificate_sha256 mismatch
    return inst, cert


@mut("integrity_instance_binding_tamper")
def m13(inst, cert):
    cert["instance_sha256"] = "0" * 64   # wrong instance binding
    return inst, reseal(cert)


def main():
    inst0, cert0 = base_cert_and_inst()
    npass = 0
    nfail = 0
    for name, fn in MUTATIONS:
        inst, cert = fn(copy.deepcopy(inst0), copy.deepcopy(cert0))
        fails = CHK.check(inst, cert)
        caught = len(fails) > 0
        status = "CAUGHT" if caught else "SLIPPED"
        print(f"  [{status}] {name}: {fails[0] if fails else 'checker returned VALID (BUG)'}")
        if caught:
            npass += 1
        else:
            nfail += 1
    print(f"\nmutation self-test: {npass}/{len(MUTATIONS)} caught, {nfail} slipped")
    return 0 if nfail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
