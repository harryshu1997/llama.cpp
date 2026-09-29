"""Typed scheduler configuration manifests: resolved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .._internal.types import canonical_json, canonical_sha256
from .campaign import CampaignManifest
from .overrides import ConfigurationOverride, ManifestIdentity
from .evidence import EvidenceManifest
from .models import ModelsManifest
from .common import RESOLVED_CONFIGURATION_SCHEMA
from .rig import RigManifest


@dataclass(frozen=True)
class ResolvedSchedulerConfiguration:
    campaign: CampaignManifest
    rig: RigManifest
    models: ModelsManifest
    evidence: EvidenceManifest
    manifest_identities: Mapping[str, ManifestIdentity]
    overrides: tuple[ConfigurationOverride, ...]

    def _json_without_hash(self) -> dict[str, object]:
        return {
            "campaign": self.campaign.to_json(),
            "evidence": self.evidence.to_json(),
            "manifest_identities": {
                key: value.to_json()
                for key, value in sorted(self.manifest_identities.items())
            },
            "models": self.models.to_json(),
            "overrides": [row.to_json() for row in self.overrides],
            "rig": self.rig.to_json(),
            "schema": RESOLVED_CONFIGURATION_SCHEMA,
        }

    @property
    def configuration_sha256(self) -> str:
        return canonical_sha256(self._json_without_hash())

    def to_json(self) -> dict[str, object]:
        return {
            **self._json_without_hash(),
            "configuration_sha256": self.configuration_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return (canonical_json(self.to_json()) + "\n").encode("ascii")
