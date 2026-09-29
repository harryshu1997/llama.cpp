#!/usr/bin/env python3

import unittest

from hierarchical_scheduler import (
    LayerModel,
    OperatorShape,
    Request,
    RouteJob,
    RouteProfile,
    decide_promotion,
    lower_bound_makespan_ms,
    optimize_task_assignment,
    plan_layers,
    plan_operator,
    schedule_jobs,
)


def request(index: int, arrival: float = 0, deadline: float = 1000) -> Request:
    return Request(index, "gemma", arrival, 100, 10, deadline)


class TaskSchedulerTest(unittest.TestCase):
    def test_fcfs_obeys_arrivals_and_slots(self):
        profile = RouteProfile(
            "cpu", "gemma", frozenset({"cpu"}), 2, 10
        )
        jobs = [
            RouteJob(request(0, 0), 30),
            RouteJob(request(1, 20), 40),
            RouteJob(request(2, 20), 10),
        ]
        result = schedule_jobs(profile, jobs)
        self.assertEqual([job.start_ms for job in result], [10, 20, 40])
        self.assertEqual([job.completion_ms for job in result], [40, 60, 50])

    def test_lpt_reduces_tail(self):
        profile = RouteProfile(
            "gpu", "gemma", frozenset({"gpu"}), 2, 0
        )
        jobs = [
            RouteJob(request(0), 2),
            RouteJob(request(1), 3),
            RouteJob(request(2), 5),
        ]
        fifo = max(job.completion_ms for job in schedule_jobs(profile, jobs))
        lpt = max(
            job.completion_ms for job in schedule_jobs(profile, jobs, "lpt")
        )
        self.assertEqual(fifo, 7)
        self.assertEqual(lpt, 5)

    def test_exact_assignment_balances_routes(self):
        requests = [request(0), request(1), request(2)]
        cpu = RouteProfile("cpu", "gemma", frozenset({"cpu"}), 1, 0)
        gpu = RouteProfile("gpu", "gemma", frozenset({"gpu"}), 1, 10)
        result = optimize_task_assignment(
            requests,
            cpu,
            gpu,
            {0: 20, 1: 20, 2: 20},
            {0: 5, 1: 5, 2: 5},
            makespan_first=True,
        )
        self.assertEqual(result.source_request_ids, (0,))
        self.assertEqual(result.target_request_ids, (1, 2))
        self.assertEqual(result.makespan_ms, 20)

    def test_promotion_hysteresis_and_protected_guard(self):
        blocked = decide_promotion(
            now_ms=100,
            protected_release_ms=100,
            next_protected_arrival_ms=120,
            switch_back_ms=10,
            guard_ms=10,
            min_gain_ms=5,
            source_only_makespan_ms=500,
            promoted_makespan_ms=200,
            target_load_ms=5,
        )
        self.assertFalse(blocked.promote)
        allowed = decide_promotion(
            now_ms=100,
            protected_release_ms=100,
            next_protected_arrival_ms=None,
            switch_back_ms=10,
            guard_ms=10,
            min_gain_ms=5,
            source_only_makespan_ms=500,
            promoted_makespan_ms=200,
            target_load_ms=5,
        )
        self.assertTrue(allowed.promote)
        self.assertEqual(allowed.saved_ms, 300)


class LayerSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.model = LayerModel(
            model_id="gemma",
            total_layers=48,
            fixed_gpu_bytes=100,
            gpu_bytes_per_layer=50,
            cpu_ms_per_layer=2,
            gpu_ms_per_layer=0.5,
            boundary_ms=1,
        )

    def test_uses_capacity_when_unprotected(self):
        plan = plan_layers(self.model, 650, 50, False, 0, None)
        self.assertEqual(plan.gpu_layers, 10)
        self.assertEqual(plan.mode, "partial_gpu")

    def test_rejects_unmeasured_protected_contention(self):
        plan = plan_layers(self.model, 650, 50, True, 0.01, None)
        self.assertEqual(plan.gpu_layers, 0)
        self.assertIn("slowdown", plan.reason)

    def test_applies_measured_qos_cap(self):
        plan = plan_layers(self.model, 650, 50, True, 0.03, 0.01)
        self.assertEqual(plan.gpu_layers, 3)


class OperatorSchedulerTest(unittest.TestCase):
    def test_exact_gemma_ffn_uses_htp_table(self):
        plan = plan_operator(
            OperatorShape(
                "gemma-4-12b-it-q4_0", "gated_ffn", 1,
                15360, 3840, "Q4_0"
            ),
            "gemma_cpu_op15",
            4 * 1024**3,
            False,
        )
        self.assertTrue(plan.qualified)
        self.assertEqual(plan.backend, "cpu+op15_htp")
        self.assertEqual(plan.split_amount, 9664)

    def test_large_prefill_stays_cpu(self):
        plan = plan_operator(
            OperatorShape(
                "gemma-4-12b-it-q4_0", "gated_ffn", 513,
                15360, 3840, "Q4_0"
            ),
            "gemma_cpu_op15",
            4 * 1024**3,
            False,
        )
        self.assertEqual(plan.backend, "cpu")

    def test_ffn_residency_checks_all_layers(self):
        plan = plan_operator(
            OperatorShape(
                "gemma-4-12b-it-q4_0", "gated_ffn", 1,
                15360, 3840, "Q4_0"
            ),
            "gemma_cpu_op15",
            2 * 1024**3,
            False,
        )
        self.assertEqual(plan.backend, "cpu")
        self.assertIn("budget", plan.reason)

    def test_cuda_route_rejects_unmeasured_phone_staging(self):
        plan = plan_operator(
            OperatorShape(
                "qwen", "lm_head", 1, 151936, 5120, "Q6_K"
            ),
            "qwen_cuda_full",
            4 * 1024**3,
            False,
        )
        self.assertEqual(plan.backend, "cuda")
        self.assertIn("staging", plan.reason)

    def test_lower_bound(self):
        profile = RouteProfile(
            "gpu", "gemma", frozenset({"gpu"}), 2, 10
        )
        jobs = [RouteJob(request(0), 8), RouteJob(request(1), 4)]
        self.assertEqual(lower_bound_makespan_ms(jobs, profile), 18)


if __name__ == "__main__":
    unittest.main()
