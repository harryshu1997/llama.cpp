#!/usr/bin/env python3

import unittest

from research_dev.scheduler._internal.runtime_search import (
    BoundedPlacementCompiler,
    RoughPlacementVisit,
)


class RuntimeSearchTests(unittest.TestCase):
    def test_mandatory_fallback_survives_required_group_pressure(self) -> None:
        compiler = BoundedPlacementCompiler(
            search_budget=4,
            refinement_budget=4,
        )
        visits = tuple(
            RoughPlacementVisit(
                route_key="helper-" + str(index),
                residency_variant="cold",
                rough_latency_us=10 + index,
                rough_energy_uj=10 + index,
                required_group="helper-group-" + str(index),
            )
            for index in range(8)
        ) + (RoughPlacementVisit(
            route_key="recovery-fallback",
            residency_variant="cold",
            rough_latency_us=1_000,
            rough_energy_uj=1_000,
            required_group="fallback",
            mandatory=True,
        ),)

        frontier, _ = compiler.compile(
            artifact_sha256="sha256:" + "5" * 64,
            capability_generation_sha256="sha256:" + "6" * 64,
            input_tokens=16,
            output_tokens=4,
            quality_requirement="exact",
            visits=visits,
        )

        self.assertIn(
            "recovery-fallback",
            {row.route_key for row in frontier.visits},
        )

    def test_required_group_prefers_memory_feasible_plan(self) -> None:
        compiler = BoundedPlacementCompiler(search_budget=1)
        frontier, cache_hit = compiler.compile(
            artifact_sha256="sha256:" + "1" * 64,
            capability_generation_sha256="sha256:" + "2" * 64,
            input_tokens=16,
            output_tokens=4,
            quality_requirement="exact",
            visits=(
                RoughPlacementVisit(
                    route_key="offload-too-wide",
                    residency_variant="cold",
                    rough_latency_us=10,
                    rough_energy_uj=10,
                    required_group="phone-offload:ffn",
                    rough_memory_feasible=False,
                ),
                RoughPlacementVisit(
                    route_key="offload-fits",
                    residency_variant="cold",
                    rough_latency_us=20,
                    rough_energy_uj=20,
                    required_group="phone-offload:ffn",
                    rough_memory_feasible=True,
                ),
            ),
        )

        self.assertFalse(cache_hit)
        self.assertEqual(
            tuple(row.route_key for row in frontier.visits),
            ("offload-fits",),
        )

    def test_device_family_coverage_survives_required_group_budget(self) -> None:
        compiler = BoundedPlacementCompiler(
            search_budget=3,
            refinement_budget=1,
        )
        visits = tuple(
            RoughPlacementVisit(
                route_key="fast-split-" + str(index),
                residency_variant="cold",
                rough_latency_us=10 + index,
                rough_energy_uj=10 + index,
                required_group="split-width-" + str(index),
                coverage_group="devices:cpu+gpu+phone",
            )
            for index in range(5)
        ) + (
            RoughPlacementVisit(
                route_key="desktop-baseline",
                residency_variant="cold",
                rough_latency_us=100,
                rough_energy_uj=100,
                required_group="desktop-baseline",
                coverage_group="devices:cpu+gpu",
            ),
            RoughPlacementVisit(
                route_key="phone-too-large",
                residency_variant="cold",
                rough_latency_us=1_000,
                rough_energy_uj=1_000,
                required_group="whole-phone",
                coverage_group="devices:phone",
                rough_memory_feasible=False,
            ),
        )

        frontier, _ = compiler.compile(
            artifact_sha256="sha256:" + "3" * 64,
            capability_generation_sha256="sha256:" + "4" * 64,
            input_tokens=16,
            output_tokens=4,
            quality_requirement="exact",
            visits=visits,
        )

        self.assertEqual(
            {row.coverage_group for row in frontier.visits},
            {
                "devices:cpu+gpu",
                "devices:cpu+gpu+phone",
                "devices:phone",
            },
        )
        self.assertIn(
            "phone-too-large",
            {row.route_key for row in frontier.visits},
        )
        self.assertTrue(all(
            row.coverage_only
            for row in frontier.visits
            if row.route_key in {
                "desktop-baseline", "phone-too-large"
            }
        ))


if __name__ == "__main__":
    unittest.main()
