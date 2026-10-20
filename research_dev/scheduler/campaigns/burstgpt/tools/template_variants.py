#!/usr/bin/env python3
"""Desktop campaign-template variants for the batch-formation ablation and the fault-injection series (WS13).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.template_variants derive \\
        --base $R/template-eval2-s2-dp-online --variant no-batch-formation --out $R/template-eval2-s2-dp-online-nobf
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.template_variants derive \\
        --base $R/template-eval2-s2-dp-online --variant elastic-mask-out --out $R/template-eval2-s2-dp-online-elastic
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.template_variants check --template DIR [--base DIR]

A template is a directory with campaign.json, rig.json, models.json and evidence.json (plus CHANGES.txt).
``prepare_campaign_eval.py`` copies campaign.json wholesale into every arm it prepares (it rewrites only the
campaign id, the manifest paths and the selection mode), and ``launch.py`` passes ``dispatch_policy``,
``adaptive_decode_overrides`` and ``elastic_phones`` to the runner unchanged, so a field set here is what the run
executes. ``derive`` copies rig/models/evidence byte for byte and changes exactly the variant's fields:

* ``no-batch-formation``: ``dispatch_policy.continuous_join`` false (and its continuous-join-only bound
  ``max_barrier_extension_s`` dropped); ``adaptive_decode_overrides`` ``server_policy_coherence``,
  ``batch_growth_verdict_inheritance`` and ``late_helper_adoption`` false (``late_helper_adoption_minimum_tokens``
  dropped). False is each flag's default, i.e. the behaviour before the feature. Work-conserving admission, model
  affinity, residency hysteresis, phone re-provisioning and device_power stay as in the base.
* ``elastic-mask-out``: adds ``elastic_phones`` with the g11 settings (drop recovery with ``mask_out``, join, probe
  every 10 s, at most 3 readmissions per device, 60 s cooldown).

``check`` reports what the runner will execute: the campaign validator's view, the runner's RuntimeDispatchPolicy
and AdaptiveDecodeConfig built exactly as ``runner.py`` builds them (including the continuous-join prerequisite),
and the elastic configuration; with ``--base`` it also lists every campaign field that differs.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Mapping

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

TEMPLATE_FILES = ("campaign.json", "rig.json", "models.json", "evidence.json")
G11_ELASTIC_PHONES = {
    "drop_recovery": True,
    "helper_loss_recovery": "mask_out",
    "join": True,
    "join_probe_interval_s": 10,
    "max_readmissions_per_device": 3,
    "readmission_cooldown_s": 60,
}
BATCH_FORMATION_FLAGS = ("server_policy_coherence", "batch_growth_verdict_inheritance", "late_helper_adoption")


class TemplateVariantError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TemplateVariantError(message)


def no_batch_formation(campaign: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """The base with continuous joining and the shared/inherited/late helper policies off."""
    result = copy.deepcopy(dict(campaign))
    policy = dict(result.get("dispatch_policy") or {})
    overrides = dict(result.get("adaptive_decode_overrides") or {})
    _need(policy.get("continuous_join") is True and overrides.get("server_policy_coherence") is True,
          "the base template does not form batches (continuous_join and server_policy_coherence must be true)")
    policy["continuous_join"] = False
    policy.pop("max_barrier_extension_s", None)   # bounds continuous-join bypasses only
    for name in BATCH_FORMATION_FLAGS:
        overrides[name] = False
    overrides.pop("late_helper_adoption_minimum_tokens", None)
    result["dispatch_policy"] = policy
    result["adaptive_decode_overrides"] = overrides
    return result, ("no-batch-formation = base with dispatch_policy.continuous_join=false (max_barrier_extension_s "
                    "dropped) and adaptive_decode_overrides server_policy_coherence / batch_growth_verdict_"
                    "inheritance / late_helper_adoption = false (late_helper_adoption_minimum_tokens dropped); "
                    "everything else unchanged")


def elastic_mask_out(campaign: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """The base with the g11 elastic-phones settings (drop recovery by masking the lost helper out)."""
    result = copy.deepcopy(dict(campaign))
    _need("elastic_phones" not in result, "the base template already declares elastic_phones")
    result["elastic_phones"] = dict(G11_ELASTIC_PHONES)
    return result, "elastic-mask-out = base + elastic_phones " + json.dumps(G11_ELASTIC_PHONES, sort_keys=True)


VARIANTS = {"no-batch-formation": no_batch_formation, "elastic-mask-out": elastic_mask_out}


def effective(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """What the runner executes for these campaign fields, built with the scheduler's own parsers."""
    from research_dev.scheduler.campaigns.burstgpt import runner
    from research_dev.scheduler.configuration import campaign as campaign_config

    policy_json = campaign.get("dispatch_policy")
    overrides_json = campaign.get("adaptive_decode_overrides")
    checked_policy = campaign_config._dispatch_policy(policy_json)
    checked_overrides = campaign_config._adaptive_decode_overrides(overrides_json)
    elastic = campaign_config.elastic_phones_configuration(campaign.get("elastic_phones"))
    policy = (runner._dispatch_policy_from_json(json.dumps(dict(checked_policy), sort_keys=True))
              if checked_policy is not None else None)
    adaptive = runner._configured_adaptive_decode_config(SimpleNamespace(
        adaptive_decode_overrides_json=(json.dumps(dict(checked_overrides), sort_keys=True)
                                        if checked_overrides is not None else None),
        adaptive_minimum_remaining_tokens=campaign.get("adaptive_minimum_remaining_tokens"),
        adaptive_maximum_probe_attempts_per_context=campaign.get("adaptive_maximum_probe_attempts_per_context"),
    ))
    if policy is not None:
        runner._require_continuous_join_prerequisites(policy, adaptive)
    return {
        "dispatch_policy": None if policy is None else {
            name: getattr(policy, name) for name in (
                "work_conserving_admission", "model_affinity", "continuous_join", "max_barrier_extension_s",
                "residency_hysteresis_s", "event_replanning")},
        "adaptive": None if adaptive is None else {
            name: getattr(adaptive, name) for name in (*BATCH_FORMATION_FLAGS, "late_helper_adoption_minimum_tokens",
                                                       "minimum_remaining_tokens",
                                                       "maximum_probe_attempts_per_context")},
        "elastic_phones": None if elastic is None else dict(elastic),
        "device_power": campaign.get("device_power") is not None,
        "phone_resident_model_reprovisioning": campaign.get("phone_resident_model_reprovisioning"),
    }


def differences(base: Mapping[str, Any], other: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted paths of every campaign field that differs (added, removed or changed)."""
    rows = []
    for key in sorted(set(base) | set(other)):
        path = prefix + key
        if key not in base:
            rows.append("+" + path)
        elif key not in other:
            rows.append("-" + path)
        elif isinstance(base[key], dict) and isinstance(other[key], dict):
            rows.extend(differences(base[key], other[key], path + "."))
        elif base[key] != other[key] or type(base[key]) is not type(other[key]):
            rows.append("~" + path)
    return rows


def derive(base: Path, variant: str, out: Path, *, today: str | None = None) -> dict[str, Any]:
    _need(variant in VARIANTS, "unknown variant " + repr(variant))
    _need(all((base / name).is_file() for name in TEMPLATE_FILES), f"{base} is not a campaign template")
    _need(not out.exists(), f"{out} already exists")
    campaign = json.loads((base / "campaign.json").read_text())
    derived, note = VARIANTS[variant](campaign)
    effective(derived)   # refuse a template the runner would refuse
    out.mkdir(parents=True)
    try:
        for name in TEMPLATE_FILES[1:]:
            shutil.copyfile(base / name, out / name)
        (out / "campaign.json").write_text(json.dumps(derived, indent=4, sort_keys=True) + "\n")
        changes = (base / "CHANGES.txt").read_text() if (base / "CHANGES.txt").is_file() else ""
        stamp = today or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        (out / "CHANGES.txt").write_text(changes + ("" if not changes or changes.endswith("\n") else "\n")
                                         + f"{stamp}: {out.name} from {base.name}: {note}.\n")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {"base": str(base), "out": str(out), "variant": variant,
            "differences": differences(campaign, derived), "effective": effective(derived)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("derive")
    command.add_argument("--base", type=Path, required=True)
    command.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("check")
    command.add_argument("--template", type=Path, required=True)
    command.add_argument("--base", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "derive":
            report = derive(args.base, args.variant, args.out)
        else:
            campaign = json.loads((args.template / "campaign.json").read_text())
            report = {"template": str(args.template), "effective": effective(campaign)}
            if args.base is not None:
                report["differences"] = differences(json.loads((args.base / "campaign.json").read_text()), campaign)
    except (TemplateVariantError, OSError, ValueError) as error:
        print("template_variants: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
