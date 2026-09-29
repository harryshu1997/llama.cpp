"""Fail-closed rules on the campaign models manifest."""
import unittest

from research_dev.scheduler.configuration.common import SchedulerConfigurationError
from research_dev.scheduler.configuration.models import validate_host_share_policy


class HostSharePolicyTests(unittest.TestCase):
    def test_flags_are_accepted_only_with_release(self):
        params = {"ffn_host_share_release": 1, "ffn_host_share_drop_cache": 0, "ffn_host_share_populate": 1}
        self.assertIs(validate_host_share_policy(params), params)
        self.assertEqual(validate_host_share_policy({}), {})
        self.assertEqual(validate_host_share_policy({"ffn_host_share_release": 1}), {"ffn_host_share_release": 1})

    def test_flags_without_release_are_rejected(self):
        # The v2r inputs of 2026-09-21 carried these on Gemma (no release) and every Gemma cold-desktop
        # launch was refused by the launch contract, quarantining the baseline route mid-trace.
        with self.assertRaisesRegex(SchedulerConfigurationError, "requires ffn_host_share_release"):
            validate_host_share_policy({"ffn_host_share_drop_cache": 0, "ffn_host_share_populate": 1})
        with self.assertRaisesRegex(SchedulerConfigurationError, "requires ffn_host_share_release"):
            validate_host_share_policy({"ffn_host_share_release": 0, "ffn_host_share_populate": 1})

    def test_flag_values_must_be_binary(self):
        with self.assertRaisesRegex(SchedulerConfigurationError, "0 or 1"):
            validate_host_share_policy({"ffn_host_share_release": 1, "ffn_host_share_drop_cache": 2})


if __name__ == "__main__":
    unittest.main()
