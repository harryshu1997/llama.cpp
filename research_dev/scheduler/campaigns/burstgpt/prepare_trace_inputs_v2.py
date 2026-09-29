#!/usr/bin/env python3
"""Derive a new trace input set (campaign, rig, models, evidence) from an existing one.

The run7 inputs pin every path to the deploy they ran from. A rerun on a new deploy needs the same
four files with: the deploy paths repointed, the campaign id renamed, the restore policy that keeps
the page cache, an optional replay schedule and trace manifest for a different trace, and an
optional slot count. Everything else is copied unchanged so the diff between input sets is exactly
the intended change. Run on the desktop:

    python3 prepare_trace_inputs_v2.py --source /home/zhihao/s42-dormant-trace-20260917-v1-inputs \\
        --output /home/zhihao/s42-trace-v2-20260921-inputs --campaign-id s42-fast-path-trace-20260921-v2a \\
        --old-deploy /mnt/storage/s42-kv-decode-relocation-20260917-v1-eedc22 \\
        --new-deploy /mnt/storage/s42-trace-v2-20260921-prep [--parallel 1] [--ubatch 512] \\
        [--replay-schedule PATH --trace-manifest PATH --large-requests PATH --overlay-requests PATH] \\
        [--adaptive-decode-overrides-json '{"server_policy_coherence": true}'] \\
        [--qualify-phone-batch-plan hot=coalesced-batch] [--transport-receipts-dir DIR] \\
        [--phone-boot-image PATH --phone-boot-image-sha256 sha256:...] \\
        [--phone-resident-model-reprovisioning-json '{}'] \\
        [--dispatch-policy-json '{"work_conserving_admission": true, "model_affinity": true}' | --drop-dispatch-policy] \\
        [--elastic-phones-json '{"drop_recovery": true, "join": true}' | --drop-elastic-phones]

Coalesced phone calls: `--qualify-phone-batch-plan MODEL_KEY=PLAN` makes PLAN the model's only
qualified batch plan. Every declared plan stays declared (a plan named by a measured route profile
must keep its executor), and a common `usb_batch_plan` override is removed because it would give the
unsuffixed executor the same parameters as the explicit one (two helper identities that differ only
by executor id cannot share a server policy). The transport receipts must cover the coalesced payload
(rows x n_embd x 2 bytes), so `--transport-receipts-dir` repoints the qualification receipts.

All phone-assisted models share one direct phone session whose transport contract includes the batch
plan, so a session launched for one plan cannot be reconfigured for a helper of another plan: with
Qwen coalesced and Gemma split-row every Gemma layout transition fails ("exact partial phone residency
transition is unavailable") and Gemma is never assisted. Different qualified plans across assisted
models are therefore refused unless `--allow-mixed-phone-batch-plans` states that this is intended.

The transport qualification identity is NOT copied: materialize it for the new deploy's server build
(python3 -m research_dev.scheduler.adapters.materialize_transport_qualification) before preflight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PHONE_BATCH_PLANS = ("coalesced-batch", "split-row")


def rewrite_paths(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [rewrite_paths(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: rewrite_paths(v, old, new) for k, v in value.items()}
    return value


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def parse_qualified_batch_plan(value: str) -> tuple[str, str]:
    model_key, separator, plan = value.partition("=")
    if separator != "=" or not model_key or plan not in PHONE_BATCH_PLANS:
        raise argparse.ArgumentTypeError(
            "expected MODEL_KEY=PLAN with PLAN one of " + ", ".join(PHONE_BATCH_PLANS))
    return model_key, plan


def parse_adaptive_overrides(value: str) -> dict[str, Any]:
    overrides = json.loads(value)
    if not isinstance(overrides, dict) or not overrides:
        raise argparse.ArgumentTypeError("adaptive decode overrides must be a non-empty JSON object")
    return overrides


def parse_reprovisioning(value: str) -> dict[str, Any]:
    reprovisioning = json.loads(value)
    if not isinstance(reprovisioning, dict):
        raise argparse.ArgumentTypeError("phone resident-model reprovisioning must be a JSON object")
    return reprovisioning


def parse_dispatch_policy(value: str) -> dict[str, Any]:
    policy = json.loads(value)
    if not isinstance(policy, dict) or not policy:
        raise argparse.ArgumentTypeError("dispatch policy must be a non-empty JSON object")
    return policy


def parse_elastic_phones(value: str) -> dict[str, Any]:
    elastic = json.loads(value)
    if not isinstance(elastic, dict) or not elastic:
        raise argparse.ArgumentTypeError("elastic phones must be a non-empty JSON object")
    return elastic


def qualify_phone_batch_plan(model: dict[str, Any], plan: str, changes: list[str]) -> None:
    declared = list(model.get("phone_batch_plans", ["split-row"]))
    if plan not in declared:
        raise SystemExit(f"{model['model_id']}: batch plan {plan} is not declared in {declared}")
    model["qualified_phone_batch_plans"] = [plan]
    params = model.setdefault("phone_adapter_parameters", {})
    if "usb_batch_plan" in params:
        # A common override copies into the unsuffixed executor and duplicates the explicit one.
        params.pop("usb_batch_plan")
        changes.append(f"{model['model_id']}: removed common phone_adapter_parameters.usb_batch_plan")
    changes.append(f"{model['model_id']}: qualified_phone_batch_plans={[plan]} (declared {declared})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--old-deploy", required=True)
    ap.add_argument("--new-deploy", required=True)
    ap.add_argument("--parallel", type=int, default=None, help="server slots for the assisted models (default: unchanged)")
    ap.add_argument("--ubatch", type=int, default=None, help="ubatch for the assisted models (default: unchanged)")
    ap.add_argument("--drop-cache", type=int, default=0, help="ffn_host_share_drop_cache (0 keeps the page cache on release)")
    ap.add_argument("--populate", type=int, default=1, help="ffn_host_share_populate")
    ap.add_argument("--replay-schedule", type=Path, default=None)
    ap.add_argument("--trace-manifest", type=Path, default=None)
    ap.add_argument("--large-requests", type=Path, default=None)
    ap.add_argument("--overlay-requests", type=Path, default=None)
    ap.add_argument("--host-memory-budget-bytes", type=int, default=None)
    ap.add_argument("--adaptive-decode-overrides-json", type=parse_adaptive_overrides, default=None,
                    help="JSON object merged into campaign.adaptive_decode_overrides "
                         "(AdaptiveDecodeConfig fields, e.g. {\"server_policy_coherence\": true})")
    ap.add_argument("--qualify-phone-batch-plan", type=parse_qualified_batch_plan, action="append", default=[],
                    metavar="MODEL_KEY=PLAN",
                    help="make PLAN the model's only qualified phone batch plan (declared plans unchanged)")
    ap.add_argument("--allow-mixed-phone-batch-plans", action="store_true",
                    help="accept assisted models with different qualified phone batch plans (one phone session "
                         "cannot serve both, so the model not matching the session's plan loses the phone)")
    ap.add_argument("--transport-receipts-dir", type=Path, default=None,
                    help="evidence.transport_qualification_directories (receipts must cover the largest payload)")
    ap.add_argument("--phone-boot-image", type=Path, default=None,
                    help="rig.binaries.phone_boot_image; requires --phone-boot-image-sha256")
    ap.add_argument("--phone-boot-image-sha256", default=None,
                    help="rig.phone.boot_image_sha256 (sha256:...), the identity the transport receipts were taken under")
    ap.add_argument("--phone-resident-model-reprovisioning-json", type=parse_reprovisioning, default=None,
                    help="campaign.phone_resident_model_reprovisioning (e.g. '{}'; exclusive with fixed_phone_residency)")
    ap.add_argument("--dispatch-policy-json", type=parse_dispatch_policy, default=None,
                    help="campaign.dispatch_policy (RuntimeDispatchPolicy fields, e.g. "
                         "'{\"work_conserving_admission\": true, \"model_affinity\": true}')")
    ap.add_argument("--drop-dispatch-policy", action="store_true",
                    help="remove campaign.dispatch_policy (the legacy dispatcher, e.g. the all-desktop baseline)")
    ap.add_argument("--elastic-phones-json", type=parse_elastic_phones, default=None,
                    help="campaign.elastic_phones (phones drop or join at runtime; validated by the campaign "
                         "manifest at resolve), e.g. '{\"drop_recovery\": true, \"join\": true}'")
    ap.add_argument("--drop-elastic-phones", action="store_true",
                    help="remove campaign.elastic_phones (the static phone set)")
    args = ap.parse_args()

    if (args.phone_boot_image is None) != (args.phone_boot_image_sha256 is None):
        ap.error("--phone-boot-image and --phone-boot-image-sha256 go together")
    if args.drop_dispatch_policy and args.dispatch_policy_json is not None:
        ap.error("--drop-dispatch-policy and --dispatch-policy-json are exclusive")
    if args.drop_elastic_phones and args.elastic_phones_json is not None:
        ap.error("--drop-elastic-phones and --elastic-phones-json are exclusive")
    if args.phone_boot_image_sha256 is not None and (
            not args.phone_boot_image_sha256.startswith("sha256:") or len(args.phone_boot_image_sha256) != 71):
        ap.error("--phone-boot-image-sha256 must be sha256:<64 hex>")
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    args.output.mkdir(parents=True)
    changes: list[str] = []

    campaign = rewrite_paths(load(args.source / "campaign.json"), args.old_deploy, args.new_deploy)
    rig = rewrite_paths(load(args.source / "rig.json"), args.old_deploy, args.new_deploy)
    models = rewrite_paths(load(args.source / "models.json"), args.old_deploy, args.new_deploy)
    evidence = rewrite_paths(load(args.source / "evidence.json"), args.old_deploy, args.new_deploy)
    changes.append(f"deploy paths {args.old_deploy} -> {args.new_deploy}")

    campaign["campaign_id"] = args.campaign_id
    for key, name in (("rig_manifest_path", "rig.json"), ("models_manifest_path", "models.json"),
                      ("evidence_manifest_path", "evidence.json")):
        campaign[key] = str(args.output / name)
    if args.host_memory_budget_bytes is not None:
        campaign["host_memory_budget_bytes"] = args.host_memory_budget_bytes
        changes.append(f"host_memory_budget_bytes={args.host_memory_budget_bytes}")
    if args.adaptive_decode_overrides_json is not None:
        overrides = dict(campaign.get("adaptive_decode_overrides") or {})
        overrides.update(args.adaptive_decode_overrides_json)
        campaign["adaptive_decode_overrides"] = overrides
        changes.append("adaptive_decode_overrides=" + json.dumps(overrides, sort_keys=True))
    if args.phone_resident_model_reprovisioning_json is not None:
        if campaign.get("fixed_phone_residency") is not None:
            raise SystemExit("fixed phone residency cannot be re-provisioned")
        campaign["phone_resident_model_reprovisioning"] = args.phone_resident_model_reprovisioning_json
        changes.append("phone_resident_model_reprovisioning="
                       + json.dumps(args.phone_resident_model_reprovisioning_json, sort_keys=True))
    if args.dispatch_policy_json is not None:
        campaign["dispatch_policy"] = args.dispatch_policy_json
        changes.append("dispatch_policy=" + json.dumps(args.dispatch_policy_json, sort_keys=True))
    if args.drop_dispatch_policy and campaign.pop("dispatch_policy", None) is not None:
        changes.append("dispatch_policy removed (legacy dispatcher)")
    if args.elastic_phones_json is not None:
        campaign["elastic_phones"] = args.elastic_phones_json
        changes.append("elastic_phones=" + json.dumps(args.elastic_phones_json, sort_keys=True))
    if args.drop_elastic_phones and campaign.pop("elastic_phones", None) is not None:
        changes.append("elastic_phones removed (static phone set)")
    trace = campaign["trace"]
    if args.replay_schedule is not None:
        trace["replay_schedule_path"] = str(args.replay_schedule.resolve())
        changes.append(f"replay schedule {args.replay_schedule}")
    if args.trace_manifest is not None:
        trace["trace_manifest_path"] = str(args.trace_manifest.resolve())
        changes.append(f"trace manifest {args.trace_manifest}")
    if args.large_requests is not None:
        trace["large_requests_path"] = str(args.large_requests.resolve())
    if args.overlay_requests is not None:
        trace["overlay_requests_path"] = str(args.overlay_requests.resolve())

    qualified_plans = dict(args.qualify_phone_batch_plan)
    for model in models["models"]:
        if model.get("kind") != "assisted":
            continue
        params = model.setdefault("phone_adapter_parameters", {})
        # The cache policy belongs to the dormant host share: only a model that releases its share can
        # apply it, and the launch contract rejects the flags on a model without release.
        if params.get("ffn_host_share_release") == 1:
            params["ffn_host_share_drop_cache"] = args.drop_cache
            params["ffn_host_share_populate"] = args.populate
            changes.append(f"{model['model_id']}: drop_cache={args.drop_cache} populate={args.populate}")
        else:
            params.pop("ffn_host_share_drop_cache", None)
            params.pop("ffn_host_share_populate", None)
            changes.append(f"{model['model_id']}: no release configured, cache policy left at defaults")
        if args.parallel is not None or args.ubatch is not None:
            for key, value in model.items():
                if isinstance(value, dict):
                    if args.parallel is not None and "parallel" in value:
                        value["parallel"] = args.parallel
                        changes.append(f"{model['model_id']}.{key}.parallel={args.parallel}")
                    if args.ubatch is not None and "ubatch_size" in value:
                        value["ubatch_size"] = args.ubatch
                        changes.append(f"{model['model_id']}.{key}.ubatch_size={args.ubatch}")
        plan = qualified_plans.pop(model.get("model_key"), None)
        if plan is not None:
            qualify_phone_batch_plan(model, plan, changes)
    if qualified_plans:
        raise SystemExit("no assisted model has model_key " + ", ".join(sorted(qualified_plans)))
    plans_by_model = {model["model_id"]: tuple(model.get("qualified_phone_batch_plans") or ())
                      for model in models["models"]
                      if model.get("kind") == "assisted" and model.get("phone_batch_plans")}
    if len(set(plans_by_model.values())) > 1:
        if not args.allow_mixed_phone_batch_plans:
            raise SystemExit("assisted models qualify different phone batch plans " + json.dumps(plans_by_model)
                             + ": one phone session cannot serve both (pass --allow-mixed-phone-batch-plans"
                             " to accept that the other model loses the phone)")
        changes.append("mixed qualified phone batch plans accepted: " + json.dumps(plans_by_model, sort_keys=True))

    if args.transport_receipts_dir is not None:
        evidence["transport_qualification_directories"] = [str(args.transport_receipts_dir.resolve())]
        changes.append(f"transport_qualification_directories={evidence['transport_qualification_directories']}")
    if args.phone_boot_image is not None:
        rig["binaries"]["phone_boot_image"] = str(args.phone_boot_image.resolve())
        rig["phone"]["boot_image_sha256"] = args.phone_boot_image_sha256
        changes.append(f"rig phone boot image {args.phone_boot_image} {args.phone_boot_image_sha256}")
    evidence["transport_qualification_identity_path"] = str(args.output / "TRANSPORT_QUALIFICATION_IDENTITY.json")
    changes.append("transport identity path points at the new inputs; materialize it before preflight")

    dump(args.output / "campaign.json", campaign)
    dump(args.output / "rig.json", rig)
    dump(args.output / "models.json", models)
    dump(args.output / "evidence.json", evidence)
    (args.output / "CHANGES.txt").write_text("\n".join(changes) + "\n")
    print("\n".join(changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
