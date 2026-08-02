#!/usr/bin/env python3

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest


PATH = pathlib.Path(__file__).with_name("validate_chain.py")
SPEC = importlib.util.spec_from_file_location("s32_validate_chain", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def compute_map(backend, host_backend=None):
    result = {"MUL_MAT": {backend: 8}}
    if host_backend:
        result["GET_ROWS"] = {host_backend: 1}
    return result


def worker_log(start, end, backend, steps, pid, host_backend=None):
    placement = compute_map(backend, host_backend)
    total = sum(sum(buffers.values()) for buffers in placement.values())
    session = {
        "schema": "ls-stagenet-session-v2",
        "session_end": "STOP",
        "expected_backend": backend,
        "worker_pid": pid,
        "layer_start": start,
        "layer_end": end,
        "n_layer": 48,
        "steps_session": steps,
        "missing_buffer_compute_nodes": 0,
        "compute_by_op_and_buffer": placement,
        "placement_status": "SCHEDULED_PLACEMENT_OK",
    }
    cert = {
        "schema": "layersplit-scheduled-placement-v2",
        "pid": pid,
        "run_rc": 0,
        "layer_start": start,
        "layer_end": end,
        "n_layer": 48,
        "compute_nodes": total,
        "missing_buffer_compute_nodes": 0,
        "compute_by_op_and_buffer": placement,
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    return b"SESSIONCERT " + MOD.canonical(session) + b"PLACEMENTCERT " + MOD.canonical(cert)


def status(pid, swap=0):
    return f"Pid:\t{pid}\nVmHWM:\t100 kB\nVmRSS:\t90 kB\nVmSwap:\t{swap} kB\n".encode("ascii")


def probe():
    tokens = [[100 + row, 200 + row, 300 + row, 400 + row] for row in range(32)]
    chain_us = [
        {"head_us": 10 + index, "middle_us": 20 + index, "tail_us": 30 + index}
        for index in range(4)
    ]
    reference_us = [{"head_us": 2 + index, "tail_us": 3 + index} for index in range(4)]
    hellos = {}
    for label, start, end, capabilities in (
        ("head", 0, 4, 15),
        ("middle", 4, 16, 15),
        ("tail", 16, 48, 31),
        ("reference", 0, 16, 15),
    ):
        hellos[label] = {
            "layer_start": start,
            "layer_end": end,
            "n_layer": 48,
            "n_embd": 3840,
            "max_streams": 32,
            "capabilities": capabilities,
        }
    digest = MOD.sha256(MOD.canonical(tokens))
    return {
        "schema": MOD.PROBE_SCHEMA,
        "status": "TOKEN_EXACT_PASS",
        "scheduler_eligible": False,
        "model_sha256": MOD.MODEL_SHA256,
        "batch": 32,
        "steps": 4,
        "initial_tokens": list(range(2, 34)),
        "route": [["OP12", 0, 4], ["OP15", 4, 16], ["CUDA", 16, 48]],
        "reference_route": [["CUDA", 0, 16], ["CUDA", 16, 48]],
        "hellos": hellos,
        "matching_requests": 32,
        "chain_tokens": tokens,
        "reference_tokens": copy.deepcopy(tokens),
        "chain_tokens_sha256": digest,
        "reference_tokens_sha256": digest,
        "chain_stage_us": chain_us,
        "reference_stage_us": reference_us,
        "chain_step_median_us": 64.5,
        "reference_step_median_us": 8.0,
    }


class ValidateChainTests(unittest.TestCase):
    def test_valid_probe(self):
        MOD.validate_probe(MOD.canonical(probe()))

    def test_rejects_token_mutation(self):
        value = probe()
        value["reference_tokens"][0][0] += 1
        with self.assertRaises(MOD.ChainError):
            MOD.validate_probe(MOD.canonical(value))

    def test_accepts_honest_negative_probe(self):
        value = probe()
        value["reference_tokens"][0][0] += 1
        value["reference_tokens_sha256"] = MOD.sha256(MOD.canonical(value["reference_tokens"]))
        value["matching_requests"] = 31
        value["status"] = "TOKEN_EXACT_FAIL"
        parsed = MOD.validate_probe(MOD.canonical(value))
        self.assertEqual(parsed["status"], "TOKEN_EXACT_FAIL")

    def test_rejects_timing_mutation(self):
        value = probe()
        value["chain_step_median_us"] = 1
        with self.assertRaises(MOD.ChainError):
            MOD.validate_probe(MOD.canonical(value))

    def test_rejects_compute_fallback(self):
        log = worker_log(0, 4, "HTP0", 128, 7).replace(b'"HTP0":8', b'"CPU":8')
        with self.assertRaises(MOD.ChainError):
            MOD.validate_worker(log, status(7), 0, 4, "HTP0", 128, "OP12")

    def test_swap_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            files = {
                "probe": MOD.canonical(probe()),
                "op12_log": worker_log(0, 4, "HTP0", 128, 11, "CPU"),
                "op15_log": worker_log(4, 16, "HTP0", 128, 12),
                "tail_log": worker_log(16, 48, "CUDA0", 256, 13),
                "reference_log": worker_log(0, 16, "CUDA0", 128, 14, "CUDA_Host"),
                "op12_status": status(11),
                "op15_status": status(12, 4),
                "op12_hash": f"{MOD.MODEL_SHA256}  /phone/model.gguf\n".encode("ascii"),
                "op15_hash": f"{MOD.MODEL_SHA256}  /phone/model.gguf\n".encode("ascii"),
                "host_hash": f"{MOD.MODEL_SHA256}  /host/model.gguf\n".encode("ascii"),
            }
            paths = {}
            for name, raw in files.items():
                paths[name] = root / name
                paths[name].write_bytes(raw)
            result = MOD.validate_chain(
                paths["probe"], paths["op12_log"], paths["op15_log"],
                paths["tail_log"], paths["reference_log"],
                paths["op12_status"], paths["op15_status"],
                paths["op12_hash"], paths["op15_hash"], paths["host_hash"],
                "/phone/model.gguf", "/host/model.gguf",
            )
            self.assertEqual(result["status"], "TOKEN_EXACT_PASS_RESOURCE_BLOCKED")
            self.assertEqual(result["resource_failures"], ["OP15_SWAP_NONZERO"])
            self.assertFalse(result["scheduler_eligible"])


if __name__ == "__main__":
    unittest.main()
