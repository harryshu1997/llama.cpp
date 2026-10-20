#!/usr/bin/env python3
"""Materialize a measured-placement arm (WS11): arm inputs whose helper layer ownership is a placement plan.

Layer ownership is bound when a model's server launches (``S41_SERVER_FFN_HELPER<k>_LAYER_MASK``) and comes
from the shard indexes: the primary phone's ``phone_ffn_shard_index_path`` (its sessions' masks) and each
co-helper's ``helper_phone_ffn_shards`` index (one full-width record) plus its evidence bundle, whose worker
``layer_mask`` must equal that record. This tool takes a base arm (campaign/rig/models/evidence JSON), a plan
(``layer_placement plan`` output, or one plan of a report) and candidate indexes / evidence bundles, and
writes a new arm in which every planned owner gets exactly its planned layers -- or refuses, naming why:

* the primary phone's planned mask must equal the union of the chosen index's records, its parent must be the
  model artifact, every record must fit the per-session resident limit;
* a co-helper's planned mask must equal its index record's mask AND its evidence worker's ``layer_mask``,
  with the evidence's artifact, shard sha256 and PASS status consistent (the full pinned-hash verification
  still runs at preflight on the rig: ``helper_phone_evidence.load_helper_evidence``);
* a co-helper may serve one model per campaign (static helper evidence is per artifact);
* a co-helper only adds layers to the primary phone's assisted route, so a model whose plan gives a
  co-helper layers must give the primary (the rig topology's ``phone_device_id``) at least one layer too;
  a plan that leaves a model wholly on the desktop CPU is refused as well (the base arm's primary layout
  would otherwise run unchanged). Plans from an inventory that marks co-helpers ``route_anchor: false``
  never contain either case (WS12);
* layers no phone owns stay on the desktop CPU (nothing to configure).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_measured_placement \\
        --inputs BASE_ARM_DIR --plan PLAN.json [--plan-key ideal] --out NEW_ARM_DIR \\
        --primary-index MODEL=INDEX.json=PHONE_DIR ... \\
        --helper-index MODEL:DEVICE=INDEX.json=PHONE_DIR ... --helper-evidence MODEL:DEVICE=EVIDENCE.json ... \\
        [--shadow-profile PROFILE.json --shadow-inventory INVENTORY.json] [--tag measured]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

PRIMARY = "op15-phone"
DESKTOP_CPU = "desktop-cpu"
RECEIPT_SCHEMA = "ws11-measured-placement-materialization-v1"


class MaterializationRefused(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationRefused(message)


def _read(path: Path):
    return json.loads(Path(path).read_text())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def plan_masks(plan: dict, key: str | None = None) -> dict[str, dict[str, int]]:
    """model -> owner -> layer mask, from a PlacementPlan JSON or one plan of a placement report."""
    if key is not None:
        plan = plan[key]
    _require(plan.get("schema") == "research-scheduler-layer-placement-v1", "not a layer placement plan")
    return {model: {owner: int(value["layer_mask"], 16) for owner, value in row["owners"].items() if value["count"]}
            for model, row in plan["models"].items()}


def _split(text: str, parts: int) -> list[str]:
    values = text.split("=")
    _require(len(values) == parts, f"argument {text!r} needs {parts} '='-separated fields")
    return values


def materialize(inputs: Path, plan: dict, out: Path, *, primary_indexes: dict[str, tuple[Path, str]],
                helper_indexes: dict[tuple[str, str], tuple[Path, str]], helper_evidence: dict[tuple[str, str], Path],
                plan_key: str | None = None, shadow: tuple[Path, Path] | None = None, tag: str = "measured",
                helper_anchored: bool = False) -> dict:
    """``helper_anchored`` (WS23, off = today's behaviour): the runtime was built with campaign
    ``elastic_phones.helper_independence``, so a co-helper may anchor a model's route and the plan may leave
    the primary phone no layer of that model. The arm then drops the primary's shard index for such a model
    (its stored slice must not overlap the helper's layers) and turns ``helper_independence`` on."""
    campaign, rig, models, evidence = [_read(inputs / (name + ".json")) for name in ("campaign", "rig", "models", "evidence")]
    masks = plan_masks(plan, plan_key)
    by_model = {row["model_id"]: row for row in models["models"]}
    primary = rig.get("topology", {}).get("phone_device_id", PRIMARY)
    for model_id, owners in sorted(masks.items()):
        co_helpers = sorted(owner for owner in owners if owner not in (primary, DESKTOP_CPU))
        _require(helper_anchored or not co_helpers or primary in owners,
                 f"{model_id}: the plan gives co-helper(s) {', '.join(co_helpers)} layers but the primary phone "
                 f"{primary} none; the runtime's assisted route belongs to the primary phone and a co-helper only "
                 "adds layers to it (plan from an inventory with route_anchor false co-helpers)")
        _require(co_helpers or primary in owners,
                 f"{model_id}: the plan leaves every layer on the desktop CPU; this arm would keep the base primary "
                 "layout (a desktop-only model is a different arm)")
    checks = []
    helper_models: dict[str, str] = {}
    for model_id, owners in sorted(masks.items()):
        for owner in owners:
            if owner not in (primary, DESKTOP_CPU):
                _require(helper_models.setdefault(owner, model_id) == model_id,
                         f"helper {owner} would serve two models ({helper_models[owner]}, {model_id}); static helper "
                         "evidence is per artifact (one Pixel worker per campaign)")
    for model_id, owners in sorted(masks.items()):
        _require(model_id in by_model, f"plan model {model_id} is not in models.json")
        row = by_model[model_id]
        artifact = None
        manifest_path = row.get("checked_manifest_path")
        if manifest_path and Path(manifest_path).exists():
            artifact = _read(Path(manifest_path)).get("artifact_sha256")
        if helper_anchored and primary not in owners and any(owner != DESKTOP_CPU for owner in owners):
            # the primary phone owns nothing of this model: its stored slice must not reach the helper's layers
            for name in ("phone_ffn_shard_index_path", "phone_ffn_shard_directory"):
                row.pop(name, None)
            checks.append({"model": model_id, "owner": primary, "status": "PRIMARY_OWNS_NONE_INDEX_DROPPED"})
        for owner, mask in sorted(owners.items()):
            if owner == DESKTOP_CPU:
                continue
            if owner == primary:
                _require(model_id in primary_indexes, f"{model_id}: no primary shard index given for {owner}")
                index_path, phone_dir = primary_indexes[model_id]
                index = _read(index_path)
                union = 0
                for record in index["shards"]:
                    union |= int(record["layer_mask"], 16)
                    _require(int(record["shard_bytes"]) <= int(row["phone_resident_limit_bytes"]),
                             f"{model_id}: {record['path']} exceeds the per-session resident limit")
                _require(union == mask, f"{model_id}: primary index layers {union:#x} != planned {mask:#x}")
                _require(artifact is None or index["parent_sha256"] == artifact,
                         f"{model_id}: primary index parent differs from the model artifact")
                row["phone_ffn_shard_index_path"] = str(index_path)
                row["phone_ffn_shard_directory"] = phone_dir
                checks.append({"model": model_id, "owner": owner, "mask": f"{mask:016x}",
                               "index": str(index_path), "index_sha256": _sha(index_path), "status": "PASS"})
                continue
            key = (model_id, owner)
            _require(key in helper_indexes and key in helper_evidence,
                     f"{model_id}: helper {owner} owns layers but has no index/evidence")
            index_path, phone_dir = helper_indexes[key]
            index = _read(index_path)
            _require(len(index["shards"]) == 1, f"{model_id}: helper {owner} index must hold one full-width shard")
            record = index["shards"][0]
            _require(int(record["layer_mask"], 16) == mask,
                     f"{model_id}: helper {owner} index layers {record['layer_mask']} != planned {mask:016x}")
            bundle = _read(helper_evidence[key])
            worker = bundle.get("worker", {})
            _require(bundle.get("schema") == "s42-static-helper-evidence-v1" and bundle.get("status") == "PASS",
                     f"{model_id}: helper {owner} evidence is not a PASS bundle")
            _require(int(worker.get("layer_mask", -1)) == mask,
                     f"{model_id}: helper {owner} evidence qualified layers {worker.get('layer_mask')} != planned {mask}")
            _require(worker.get("device_id") == owner, f"{model_id}: evidence is for {worker.get('device_id')}")
            _require(artifact is None or worker.get("artifact_sha256") == artifact,
                     f"{model_id}: helper {owner} evidence is for another artifact")
            shard_hash = worker.get("expected_sha256_by_path", {}).get(worker.get("shard_path"))
            _require(shard_hash == record["shard_sha256"],
                     f"{model_id}: helper {owner} evidence shard sha differs from the index record")
            row.setdefault("helper_phone_ffn_shards", {})[owner] = {"index_path": str(index_path), "directory": phone_dir}
            evidence.setdefault("helper_phone_evidence_paths", {})[owner] = str(helper_evidence[key])
            for rig_row in rig.get("helper_phones", ()):
                if rig_row["device_id"] == owner:
                    for name in ("worker_path", "library_directories", "worker_environment", "column_quantum",
                                 "max_tokens", "backend", "as_root", "phone_lock_path"):
                        if name in worker:
                            rig_row[name] = worker[name]
                    rig_row["worker_port"] = worker.get("phone_port", rig_row.get("worker_port"))
            checks.append({"model": model_id, "owner": owner, "mask": f"{mask:016x}", "index": str(index_path),
                           "index_sha256": _sha(index_path), "evidence": str(helper_evidence[key]),
                           "evidence_sha256": _sha(helper_evidence[key]), "status": "PASS"})
        # helpers the plan does not use for this model must not keep serving it
        for owner in list(row.get("helper_phone_ffn_shards", {})):
            if owner not in owners:
                del row["helper_phone_ffn_shards"][owner]
                checks.append({"model": model_id, "owner": owner, "status": "REMOVED_NOT_IN_PLAN"})
        if not row.get("helper_phone_ffn_shards"):
            row.pop("helper_phone_ffn_shards", None)
    used_helpers = set(helper_models)
    for owner in list(evidence.get("helper_phone_evidence_paths", {})):
        if owner not in used_helpers:
            del evidence["helper_phone_evidence_paths"][owner]
    if not evidence.get("helper_phone_evidence_paths"):
        evidence.pop("helper_phone_evidence_paths", None)
    campaign["campaign_id"] = campaign["campaign_id"] + "-" + tag
    if helper_models:
        # a measured plan is the layout to evaluate: the adaptive cold start must probe the full
        # device set first, not the primary phone alone (which may own a single anchor layer)
        overrides = dict(campaign.get("adaptive_decode_overrides") or {})
        overrides["cold_start_probe_widest_set_first"] = True
        campaign["adaptive_decode_overrides"] = overrides
    if helper_anchored:
        from research_dev.scheduler.campaigns.burstgpt.tools.template_variants import G11_ELASTIC_PHONES
        elastic = dict(campaign.get("elastic_phones") or G11_ELASTIC_PHONES)
        elastic["helper_independence"] = True
        campaign["elastic_phones"] = elastic
    if shadow is not None:
        policy = dict(campaign.get("dispatch_policy") or {})
        policy["measured_placement"] = {"mode": "shadow", "profile_path": str(shadow[0]),
                                        "inventory_path": str(shadow[1])}
        campaign["dispatch_policy"] = policy
    out.mkdir(parents=True, exist_ok=False)
    for name in ("rig", "models", "evidence"):
        campaign[name + "_manifest_path"] = str(out / (name + ".json"))
    for name, value in (("campaign", campaign), ("rig", rig), ("models", models), ("evidence", evidence)):
        (out / (name + ".json")).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    receipt = {"schema": RECEIPT_SCHEMA, "status": "PASS", "base_inputs": str(inputs), "plan_key": plan_key,
               "planned_masks": {model: {owner: f"{mask:016x}" for owner, mask in owners.items()}
                                 for model, owners in sorted(masks.items())},
               "checks": checks}
    (out / "PLACEMENT_MATERIALIZATION.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-key", choices=("ideal", "executable"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary-index", action="append", default=[], metavar="MODEL=INDEX.json=PHONE_DIR")
    parser.add_argument("--helper-index", action="append", default=[], metavar="MODEL:DEVICE=INDEX.json=PHONE_DIR")
    parser.add_argument("--helper-evidence", action="append", default=[], metavar="MODEL:DEVICE=EVIDENCE.json")
    parser.add_argument("--shadow-profile", type=Path)
    parser.add_argument("--shadow-inventory", type=Path)
    parser.add_argument("--tag", default="measured")
    parser.add_argument("--helper-anchored", action="store_true",
                        help="WS23: allow plans where the primary phone owns none of a model's layers "
                             "(campaign elastic_phones.helper_independence is turned on)")
    args = parser.parse_args(argv)
    try:
        primary = {}
        for text in args.primary_index:
            model, index, phone_dir = _split(text, 3)
            primary[model] = (Path(index), phone_dir)
        helpers = {}
        for text in args.helper_index:
            key, index, phone_dir = _split(text, 3)
            model, _, device = key.partition(":")
            helpers[(model, device)] = (Path(index), phone_dir)
        evidence = {}
        for text in args.helper_evidence:
            key, path = _split(text, 2)
            model, _, device = key.partition(":")
            evidence[(model, device)] = Path(path)
        shadow = None
        if args.shadow_profile or args.shadow_inventory:
            _require(bool(args.shadow_profile and args.shadow_inventory), "shadow needs both profile and inventory")
            shadow = (args.shadow_profile.resolve(), args.shadow_inventory.resolve())
        receipt = materialize(args.inputs, _read(args.plan), args.out, primary_indexes=primary, helper_indexes=helpers,
                              helper_evidence=evidence, plan_key=args.plan_key, shadow=shadow, tag=args.tag,
                              helper_anchored=args.helper_anchored)
    except (MaterializationRefused, OSError, KeyError, ValueError) as error:
        print("prepare_measured_placement: REFUSED: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"status": receipt["status"], "planned_masks": receipt["planned_masks"]}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
