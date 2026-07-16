#!/usr/bin/env python3
"""Generate deterministic schema and bundle fixtures for S9 bundle version 5."""
import copy
import json
import os

import make_fixtures_v2 as F2
import make_fixtures_v4 as F4
import s9lib


HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures", "v5")
GHOST = F2.h("v5:ghost")


def digest(kind, record):
    field, builder = s9lib.DIGEST_BUILDERS_V5[kind]
    record[field] = builder(record)


def refresh_chain_v5(records):
    for key in ("ALLOC1", "ALLOC2"):
        if key in records:
            digest("canonical_allocation", records[key])
    for key in ("PI1", "PI2", "PIgpu"):
        if key in records:
            digest("prepared_image", records[key])
    for key in ("CC1", "CC2"):
        if key in records:
            digest("correctness_certificate", records[key])
    if "ISL1" in records:
        records["ISL1"]["required_prepared_image_digests"] = [records["PI1"]["derived_image_digest"]]
        records["ISL1"]["correctness_digest"] = records["CC1"]["correctness_digest"]
        digest("island_executable", records["ISL1"])
    if "ISL2" in records:
        records["ISL2"]["required_prepared_image_digests"] = [
            records["PI1"]["derived_image_digest"], records["PI2"]["derived_image_digest"]]
        records["ISL2"]["correctness_digest"] = records["CC2"]["correctness_digest"]
        digest("island_executable", records["ISL2"])
    for rck, cck, pik in (("RC1", "CC1", "PI1"), ("RC2", "CC2", "PI2")):
        if rck in records:
            records[rck]["prepared_image_digest"] = records[pik]["derived_image_digest"]
            records[rck]["correctness_digest"] = records[cck]["correctness_digest"]
            digest("ready_certificate", records[rck])
    for rlk, rck in (("RL1", "RC1"), ("RL2", "RC2")):
        if rlk in records:
            records[rlk]["ready_certificate_digest"] = records[rck]["ready_certificate_digest"]
            digest("residency_lease", records[rlk])
    if "SL1" in records:
        digest("state_lease", records["SL1"])
    for key in ("TT1", "TT2"):
        if key in records:
            digest("transfer_ticket", records[key])
    return records


def build_records_v5():
    records = copy.deepcopy(F4.build_records_v4())
    for record in records.values():
        record["schema_version"] = 5
    refresh_chain_v5(records)
    return records


CORE_KEYS = ["MM", "SEGa", "WS1", "ALLOC1", "PI1", "CC1", "ISL1", "RC1",
             "RL1", "SL1", "DD1", "TT1", "TT2", "TFbulk", "TFexec", "DI"]


def bundle(bundle_id, records, version=5):
    return {"schema_version": version, "kind": "bundle", "bundle_version": version,
            "bundle_id": bundle_id, "records": copy.deepcopy(records)}


def core(records):
    return [records[key] for key in CORE_KEYS]


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(s9lib.canonical(obj) + "\n")


def mutate_rl_horizon(records, refresh):
    records["RL1"]["horizon"]["end_us"] += 1000000


def mutate_rc_free_ram_without_digest(records, refresh):
    records["RC1"]["free_ram_after_bytes"] -= 1


def mutate_rc_weight_digest(records, refresh):
    records["RC1"]["weight_set_digest"] = GHOST
    refresh(records)


def mutate_rl_model(records, refresh):
    records["RL1"]["model_id"] = "foreign-model"
    refresh(records)


def mutate_island_range(records, refresh):
    records["ISL1"]["layer_range"] = {"start": 1, "end": 2, "n_layer_total": 2}
    refresh(records)


def mutate_ws_model(records, refresh):
    records["WS1"]["model_id"] = "foreign-model"


def mutate_min_soc(records, refresh):
    records["MM"]["required_backends"][0]["min_soc"] = "op15"
    for key in ("DI", "ALLOC1", "PI1", "RC1"):
        records[key]["soc"] = "op12"
    refresh(records)


def mutate_future_di(records, refresh):
    records["DI"]["receiver_ts_us"] = records["DD1"]["decision_ts_us"] + 1


def mutate_future_rc(records, refresh):
    records["RC1"]["issued_receiver_ts_us"] = records["DD1"]["decision_ts_us"] + 1
    refresh(records)


def mutate_lease_before_ready(records, refresh):
    records["RC1"]["issued_receiver_ts_us"] = records["RL1"]["horizon"]["start_us"] + 1
    refresh(records)


def mutate_ticket_model(records, refresh):
    records["TT1"]["model_id"] = "foreign-model"
    refresh(records)


def mutate_ticket_device(records, refresh):
    records["TT1"]["device_id"] = "op12"
    records["TFbulk"]["device_id"] = "op12"
    refresh(records)


def mutate_ticket_weight_set(records, refresh):
    records["TT1"]["weight_set_id"] = "ws2"
    refresh(records)


def mutate_ticket_missing_segment(records, refresh):
    records["TT1"]["segment_id"] = "missing-segment"
    records["TFbulk"]["bulk_binding"]["segment_id"] = "missing-segment"
    refresh(records)


def mutate_ticket_unlisted_segment(records, refresh):
    records["EXTRA"] = copy.deepcopy(records["SEGa"])
    records["EXTRA"]["segment_id"] = "seg-unlisted"
    records["TT1"]["segment_id"] = "seg-unlisted"
    records["TFbulk"]["bulk_binding"]["segment_id"] = "seg-unlisted"
    refresh(records)


def mutate_ticket_range(records, refresh):
    records["TT1"]["byte_range"]["length"] = 1
    refresh(records)


def mutate_frame_outside_ticket(records, refresh):
    records["TFbulk"]["bulk_binding"]["ticket_id"] = "tt2"


def mutate_frame_payload_length(records, refresh):
    records["TFbulk"]["payload_bytes"] = 1


def mutate_duplicate_execute(records, refresh):
    records["EXTRA"] = copy.deepcopy(records["TFexec"])


def mutate_duplicate_bulk(records, refresh):
    records["EXTRA"] = copy.deepcopy(records["TFbulk"])


def mutate_island_total(records, refresh):
    records["ISL1"]["layer_range"]["n_layer_total"] = 99
    refresh(records)


def mutate_island_io(records, refresh):
    records["ISL1"]["io_schema"]["max_input_bytes"] = \
        records["CC1"]["shape_envelope"]["max_input_bytes"] + 1
    refresh(records)


def mutate_duplicate_requirements(records, refresh):
    for field in ("required_weight_set_ids", "required_weight_set_digests",
                  "required_prepared_image_ids", "required_prepared_image_digests"):
        records["ISL1"][field].append(records["ISL1"][field][0])
    records["DD1"]["required_tuples"].append(copy.deepcopy(records["DD1"]["required_tuples"][0]))
    records["DD1"]["satisfied_tuples"].append(copy.deepcopy(records["DD1"]["satisfied_tuples"][0]))
    if records["ISL1"]["schema_version"] == 4:
        records["ISL1"]["island_digest"] = s9lib.island_digest_v4(records["ISL1"])
    else:
        digest("island_executable", records["ISL1"])


def mutate_rc_physical_partition(records, refresh):
    records["RC1"]["physical_bytes"]["canonical"] = 0
    records["RC1"]["physical_bytes"]["derived"] = 400
    refresh(records)


def mutate_rc_false_free(records, refresh):
    records["RC1"]["free_ram_after_bytes"] -= 1
    refresh(records)


def mutate_duplicate_backend_identity(records, refresh):
    duplicate = copy.deepcopy(records["MM"]["required_backends"][0])
    duplicate["min_soc"] = "op15"
    records["MM"]["required_backends"].append(duplicate)


CASES = [
    ("rl_horizon_digest.json", mutate_rl_horizon, ["E_DIGEST_MISMATCH"]),
    ("rc_free_ram_digest.json", mutate_rc_free_ram_without_digest,
     ["E_DIGEST_MISMATCH", "E_LEDGER_DERIVED"]),
    ("rc_weight_digest_foreign.json", mutate_rc_weight_digest, ["E_CHAIN_BROKEN"]),
    ("rl_model_foreign.json", mutate_rl_model, ["E_CHAIN_BROKEN"]),
    ("island_range_uncovered.json", mutate_island_range, ["E_IDENTITY_MISMATCH"]),
    ("weight_set_foreign_model.json", mutate_ws_model, ["E_FRAME_BINDING", "E_IDENTITY_MISMATCH"]),
    ("min_soc_violation.json", mutate_min_soc, ["E_DEVICE_INELIGIBLE"]),
    ("device_snapshot_from_future.json", mutate_future_di, ["E_SNAPSHOT_CAUSAL"]),
    ("ready_certificate_from_future.json", mutate_future_rc, ["E_SNAPSHOT_CAUSAL"]),
    ("lease_predates_ready.json", mutate_lease_before_ready, ["E_SNAPSHOT_CAUSAL"]),
    ("ticket_foreign_model.json", mutate_ticket_model, ["E_FRAME_BINDING"]),
    ("ticket_unknown_device.json", mutate_ticket_device, ["E_FRAME_BINDING"]),
    ("ticket_weight_set_mismatch.json", mutate_ticket_weight_set, ["E_FRAME_BINDING"]),
    ("ticket_missing_segment.json", mutate_ticket_missing_segment, ["E_FRAME_BINDING"]),
    ("ticket_unlisted_segment.json", mutate_ticket_unlisted_segment, ["E_FRAME_BINDING"]),
    ("ticket_range_mismatch.json", mutate_ticket_range, ["E_FRAME_BINDING"]),
    ("frame_chunk_outside_ticket.json", mutate_frame_outside_ticket, ["E_FRAME_BINDING"]),
    ("frame_payload_length_mismatch.json", mutate_frame_payload_length, ["E_FRAME_BINDING"]),
    ("duplicate_execute.json", mutate_duplicate_execute, ["E_FRAME_DUPLICATE"]),
    ("duplicate_bulk.json", mutate_duplicate_bulk, ["E_FRAME_DUPLICATE"]),
    ("island_total_mismatch.json", mutate_island_total, ["E_IDENTITY_MISMATCH"]),
    ("island_io_exceeds_certificate.json", mutate_island_io, ["E_IDENTITY_MISMATCH"]),
    ("duplicate_requirements.json", mutate_duplicate_requirements, ["E_SCHEMA"]),
    ("rc_false_physical_partition.json", mutate_rc_physical_partition, ["E_LEDGER_DERIVED"]),
    ("rc_false_free_ram.json", mutate_rc_false_free, ["E_LEDGER_DERIVED"]),
    ("duplicate_backend_identity.json", mutate_duplicate_backend_identity, ["E_IDENTITY_MISMATCH"]),
]


def emit_schema_fixtures(records):
    valid = []
    invalid = []

    def add_valid(name, kind, record):
        write(os.path.join(FX, "valid", name), record)
        valid.append({"file": f"valid/{name}", "schema": f"schemas/v5/{s9lib.V5_KINDS[kind]}",
                      "expect": "valid"})

    def add_invalid(name, kind, record):
        write(os.path.join(FX, "invalid", name), record)
        invalid.append({"file": f"invalid/{name}", "schema": f"schemas/v5/{s9lib.V5_KINDS[kind]}",
                        "expect": "invalid"})

    entries = [
        ("weight_segment.json", "weight_segment", "SEGa"),
        ("weight_set.json", "weight_set", "WS1"),
        ("model_manifest.json", "model_manifest", "MM"),
        ("canonical_allocation.json", "canonical_allocation", "ALLOC1"),
        ("prepared_image_htp.json", "prepared_image", "PI1"),
        ("prepared_image_gpu.json", "prepared_image", "PIgpu"),
        ("correctness_certificate.json", "correctness_certificate", "CC1"),
        ("island_executable.json", "island_executable", "ISL1"),
        ("ready_certificate.json", "ready_certificate", "RC1"),
        ("residency_lease.json", "residency_lease", "RL1"),
        ("state_lease.json", "state_lease", "SL1"),
        ("dispatch_decision.json", "dispatch_decision", "DD1"),
        ("transfer_ticket.json", "transfer_ticket", "TT1"),
        ("transport_frame_bulk.json", "transport_frame", "TFbulk"),
        ("transport_frame_exec.json", "transport_frame", "TFexec"),
        ("device_inventory.json", "device_inventory", "DI"),
    ]
    for name, kind, key in entries:
        add_valid(name, kind, records[key])

    record = copy.deepcopy(records["DD1"])
    del record["required_tuples"][0]["weight_set_id"]
    add_invalid("dispatch_tuple_missing_weight_set.json", "dispatch_decision", record)
    record = copy.deepcopy(records["RL1"])
    del record["horizon"]
    add_invalid("residency_lease_missing_horizon.json", "residency_lease", record)
    record = copy.deepcopy(records["MM"])
    del record["required_backends"][0]["min_soc"]
    add_invalid("manifest_backend_missing_min_soc.json", "model_manifest", record)
    record = copy.deepcopy(records["TFbulk"])
    record["payload_sha256"] = None
    add_invalid("bulk_frame_missing_payload_hash.json", "transport_frame", record)
    record = copy.deepcopy(records["TT1"])
    record["byte_range"]["length"] = 0
    add_invalid("transfer_ticket_zero_length.json", "transfer_ticket", record)
    record = copy.deepcopy(records["RC1"])
    record["issued_receiver_ts_us"] = -1
    add_invalid("ready_certificate_negative_timestamp.json", "ready_certificate", record)
    record = copy.deepcopy(records["WS1"])
    record["segments"].append(copy.deepcopy(record["segments"][0]))
    add_invalid("weight_set_duplicate_segment.json", "weight_set", record)
    record = copy.deepcopy(records["MM"])
    record["required_backends"].append(copy.deepcopy(record["required_backends"][0]))
    add_invalid("manifest_duplicate_backend.json", "model_manifest", record)

    index = valid + invalid
    with open(os.path.join(FX, "index.json"), "w") as f:
        f.write(json.dumps(index, indent=2) + "\n")
    return len(valid), len(invalid)


def emit_bundle_fixtures(records):
    valid_dir = os.path.join(FX, "bundles", "valid")
    invalid_dir = os.path.join(FX, "bundles", "invalid")
    write(os.path.join(valid_dir, "dispatchable.json"), bundle("b5-dispatchable", core(records)))
    index = [{"file": "valid/dispatchable.json", "expect": "valid", "expected_codes": []}]
    for name, mutator, codes in CASES:
        mutated = copy.deepcopy(records)
        mutator(mutated, refresh_chain_v5)
        ordered = core(mutated)
        if "EXTRA" in mutated:
            ordered.append(mutated["EXTRA"])
        write(os.path.join(invalid_dir, name), bundle("b5-" + name[:-5], ordered))
        index.append({"file": f"invalid/{name}", "expect": "invalid", "expected_codes": codes})

    malformed = copy.deepcopy(records)
    del malformed["DD1"]["required_tuples"][0]["weight_set_id"]
    name = "schema_invalid_tuple.json"
    write(os.path.join(invalid_dir, name), bundle("b5-schema-invalid", core(malformed)))
    index.append({"file": f"invalid/{name}", "expect": "invalid", "expected_codes": ["E_SCHEMA"]})

    with open(os.path.join(FX, "bundles", "index.json"), "w") as f:
        f.write(json.dumps(index, indent=2) + "\n")
    return 1, len(index) - 1


def main():
    records = build_records_v5()
    schema_counts = emit_schema_fixtures(records)
    bundle_counts = emit_bundle_fixtures(records)
    print(f"v5 schema fixtures: {schema_counts[0]} valid + {schema_counts[1]} invalid")
    print(f"v5 bundle fixtures: {bundle_counts[0]} valid + {bundle_counts[1]} invalid")


if __name__ == "__main__":
    main()
