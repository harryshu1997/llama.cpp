"""Typed scheduler configuration manifests: the site profile (what was measured on a site).

A site file says what the machine is; the profile says how it performs. It holds, per model, the host's
measured numbers; per phone, what it can keep and how its energy is known; per (phone, model) pair, the
shared-layer fit, the static share plan and the runs they came from; and the same measurements in the
layer-placement planner's own profile schema, so the planner prices this site without a conversion.

Records are written by the onboarding commands (``campaigns/portable/onboard.py``), never by hand; every
number carries the run directory it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

from .common import SchedulerConfigurationError, _require


SITE_PROFILE_SCHEMA = "research-scheduler-site-profile-v1"
HOST_MODEL_KEYS = ("artifact_sha256", "architecture", "n_embd", "n_ff", "block_count", "weight_type", "gpu_layers",
                   "cpu_layers", "layer_bytes", "token_ms_alone", "p_idle_w", "p_decode_w", "host_share",
                   "memory_available_bytes", "evidence")
PHONE_KEYS = ("transport", "kind", "usable_bytes", "sessions", "session_limit_bytes", "power_quality",
              "awake", "cpu_caps_khz", "evidence")
PAIR_KEYS = ("device_id", "model_id", "transport", "fit", "share_plan", "evidence")


def _record(value: object, keys: tuple[str, ...], name: str) -> dict[str, object]:
    _require(type(value) is dict, name + " must be an object")
    missing = [key for key in keys if key not in value]
    _require(not missing, name + " lacks " + ",".join(missing))
    return dict(value)


@dataclass
class SiteProfile:
    site_id: str
    revision: int = 0
    host_models: dict[str, dict[str, object]] = field(default_factory=dict)
    phones: dict[str, dict[str, object]] = field(default_factory=dict)
    pairs: list[dict[str, object]] = field(default_factory=list)
    layer_placement_profile: dict[str, object] | None = None

    # -- updates (each bumps the revision) --------------------------------------------------------
    def set_host_model(self, model_id: str, record: Mapping[str, object]) -> None:
        self.host_models[model_id] = _record(dict(record), HOST_MODEL_KEYS, "host model record")
        self.revision += 1

    def update_host_model(self, model_id: str, **values: object) -> None:
        """Numbers the first phone sweep adds to a host model (t_host_ms, c_host_ms, p_f_w ...)."""
        _require(model_id in self.host_models, "the profile has no host record for " + model_id)
        self.host_models[model_id].update(values)
        self.revision += 1

    def set_phone(self, device_id: str, record: Mapping[str, object]) -> None:
        self.phones[device_id] = _record(dict(record), PHONE_KEYS, "phone record")
        self.revision += 1

    def set_pair(self, record: Mapping[str, object]) -> None:
        row = _record(dict(record), PAIR_KEYS, "pair record")
        self.pairs = [pair for pair in self.pairs
                      if (pair["device_id"], pair["model_id"]) != (row["device_id"], row["model_id"])] + [row]
        self.revision += 1

    # -- lookups ---------------------------------------------------------------------------------
    def host_model(self, model_id: str) -> dict[str, object]:
        _require(model_id in self.host_models, "the profile has no host record for " + model_id
                 + " (run onboard-host)")
        return self.host_models[model_id]

    def pair(self, device_id: str, model_id: str) -> dict[str, object]:
        for row in self.pairs:
            if (row["device_id"], row["model_id"]) == (device_id, model_id):
                return row
        raise SchedulerConfigurationError(f"the profile has no pair {device_id} / {model_id} (run onboard-phone)")

    # -- persistence -----------------------------------------------------------------------------
    def to_json(self) -> dict[str, object]:
        return {
            "schema": SITE_PROFILE_SCHEMA, "site_id": self.site_id, "revision": self.revision,
            "host": {"models": {name: dict(row) for name, row in sorted(self.host_models.items())}},
            "phones": {name: dict(row) for name, row in sorted(self.phones.items())},
            "pairs": sorted((dict(row) for row in self.pairs), key=lambda row: (row["device_id"], row["model_id"])),
            **({"layer_placement_profile": self.layer_placement_profile}
               if self.layer_placement_profile is not None else {}),
        }

    @classmethod
    def from_json(cls, value: object) -> "SiteProfile":
        _require(type(value) is dict and value.get("schema") == SITE_PROFILE_SCHEMA, "not a site profile")
        _require(type(value.get("site_id")) is str and bool(value["site_id"]), "site profile needs a site id")
        _require(type(value.get("revision", 0)) is int and value.get("revision", 0) >= 0, "site profile revision")
        host = value.get("host", {})
        _require(type(host) is dict and type(host.get("models", {})) is dict, "site profile host is invalid")
        return cls(
            site_id=value["site_id"], revision=value.get("revision", 0),
            host_models={str(name): _record(row, HOST_MODEL_KEYS, "host model record")
                         for name, row in host.get("models", {}).items()},
            phones={str(name): _record(row, PHONE_KEYS, "phone record")
                    for name, row in dict(value.get("phones", {})).items()},
            pairs=[_record(row, PAIR_KEYS, "pair record") for row in value.get("pairs", [])],
            layer_placement_profile=value.get("layer_placement_profile"))

    @classmethod
    def load(cls, path: Path, site_id: str) -> "SiteProfile":
        """The profile at ``path`` (it must belong to ``site_id``), or a new one when the file is absent."""
        path = Path(path)
        if not path.exists():
            return cls(site_id)
        try:
            profile = cls.from_json(json.loads(path.read_text(encoding="ascii")))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SchedulerConfigurationError("site profile cannot be loaded: " + str(error)) from error
        _require(profile.site_id == site_id, f"{path} is the profile of site {profile.site_id}, not {site_id}")
        return profile

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n", encoding="ascii")
