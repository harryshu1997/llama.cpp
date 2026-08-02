#!/usr/bin/env python3

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("builder", HERE / "build_contract_v26.py")
authority = load("authority", HERE / "cp0_r1_evidence_v26.py")


class AuthorityTests(unittest.TestCase):
    def test_contract_and_composition_are_current(self):
        contract = builder.build_contract()
        paths = authority.validate_composition(contract)
        self.assertIn("v26.authority", paths)
        self.assertTrue(contract["claim_boundary"]["acquisition_authorized"])
        self.assertFalse(contract["claim_boundary"]["b_only_authorized"])

    def test_composition_digest_mutation_is_rejected(self):
        contract = builder.build_contract()
        bad = copy.deepcopy(contract)
        bad["composition"]["inner_v24"]["authority"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(authority.EvidenceError, "E_COMPOSITION_SHA"):
            authority.validate_composition(bad)

    def test_legacy_status_is_not_authorization(self):
        contract = builder.build_contract()
        contract["claim_boundary"]["b_only_authorized"] = True
        fd, name = tempfile.mkstemp(prefix="s39-v26-bad-")
        os.close(fd)
        path = Path(name)
        try:
            path.write_bytes(builder.canonical_bytes(contract))
            with self.assertRaisesRegex(authority.EvidenceError, "contract.b_only"):
                authority.load_contract(path)
        finally:
            path.unlink()

    def test_wrong_phase_is_rejected(self):
        contract = builder.build_contract()
        bad = copy.deepcopy(contract)
        bad["claim_boundary"]["authorized_phase"] = "PAIR"
        fd, name = tempfile.mkstemp(prefix="s39-v26-phase-")
        os.close(fd)
        path = Path(name)
        try:
            path.write_bytes(builder.canonical_bytes(bad))
            with self.assertRaisesRegex(authority.EvidenceError, "contract.phase"):
                authority.load_contract(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
