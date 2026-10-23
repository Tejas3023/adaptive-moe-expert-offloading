"""Tests for the capacity-aware timing simulator.

Run from project root:
    python -m unittest scripts.test_timing_aware_simulator -v
"""

import unittest

from src.cache.cost_model import CacheCostModel
from src.cache.timing_model import TimingAwareSimulator


class TestTimingAwareSimulator(unittest.TestCase):
    def setUp(self):
        self.costs = CacheCostModel(
            gpu_hit_ms=0.05,
            cpu_to_gpu_ms=1.0,
            disk_to_gpu_ms=10.0,
        )

    def test_prefetch_finishes_during_computation_with_no_stall(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=2, cpu_capacity=2)
        sim.prefetch(0, 1, source="disk")
        sim.advance_compute(12.0)
        result = sim.request(0, 1)
        self.assertEqual(result["status"], "gpu")
        self.assertAlmostEqual(result["stall_ms"], 0.0)
        self.assertAlmostEqual(sim.current_time_ms, 12.05)

    def test_partial_overlap_reduces_stall(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=2, cpu_capacity=2)
        sim.prefetch(0, 1, source="disk")
        sim.advance_compute(8.0)
        result = sim.request(0, 1)
        self.assertEqual(result["status"], "prefetch_wait")
        self.assertAlmostEqual(result["stall_ms"], 2.0)
        self.assertAlmostEqual(sim.current_time_ms, 10.0)

    def test_demand_disk_miss_stalls_for_transfer_duration(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=2, cpu_capacity=2)
        result = sim.request(0, 1)
        self.assertEqual(result["status"], "miss")
        self.assertAlmostEqual(result["stall_ms"], 10.0)
        self.assertAlmostEqual(sim.current_time_ms, 10.0)
        self.assertEqual(sim.statistics()["demand_disk_misses"], 1)

    def test_duplicate_prefetch_does_not_schedule_second_transfer(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=2, cpu_capacity=2)
        first = sim.prefetch(0, 1)
        second = sim.prefetch(0, 1)
        self.assertEqual(first["status"], "scheduled")
        self.assertEqual(second["status"], "already_in_flight")
        self.assertEqual(len(sim.events), 1)
        self.assertEqual(sim.statistics()["prefetch_deduplicated"], 1)

    def test_prefetch_of_resident_expert_is_not_scheduled(self):
        sim = TimingAwareSimulator(
            self.costs, gpu_capacity=2, cpu_capacity=2,
            initially_resident={(0, 1)}
        )
        result = sim.prefetch(0, 1)
        self.assertEqual(result["status"], "already_resident")
        self.assertEqual(len(sim.events), 0)

    def test_transfer_channel_serializes_two_prefetches(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=3, cpu_capacity=2)
        first = sim.prefetch(0, 1, source="disk")
        # The second key is absent from CPU, so schedule it as a disk transfer.
        second = sim.prefetch(0, 2, source="disk")
        self.assertAlmostEqual(first["event"].start_ms, 0.0)
        self.assertAlmostEqual(first["event"].finish_ms, 10.0)
        self.assertAlmostEqual(second["event"].start_ms, 10.0)
        self.assertAlmostEqual(second["event"].finish_ms, 20.0)

    def test_transfer_overlaps_compute(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=2, cpu_capacity=2)
        sim.prefetch(0, 1)
        sim.advance_compute(10.0)
        stats = sim.statistics()
        self.assertAlmostEqual(stats["total_transfer_ms"], 10.0)
        self.assertAlmostEqual(stats["simulated_time_ms"], 10.0)
        self.assertAlmostEqual(stats["total_stall_ms"], 0.0)

    def test_negative_compute_duration_is_rejected(self):
        sim = TimingAwareSimulator(self.costs)
        with self.assertRaises(ValueError):
            sim.advance_compute(-1.0)

    def test_gpu_capacity_and_lru_eviction_are_enforced(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=2, cpu_capacity=2)
        sim.request(0, 1)
        sim.request(0, 2)
        sim.request(0, 1)  # touch expert 1; expert 2 becomes LRU
        sim.request(0, 3)
        self.assertEqual(sim.cached_gpu(), [(0, 1), (0, 3)])
        self.assertIn((0, 2), sim.cached_cpu())
        self.assertLessEqual(len(sim.cached_gpu()), 2)
        self.assertLessEqual(len(sim.cached_cpu()), 2)

    def test_cpu_expert_is_promoted_and_cpu_to_gpu_cost_is_used(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=1, cpu_capacity=2)
        sim.request(0, 1)
        sim.request(0, 2)  # expert 1 moves to CPU
        self.assertIn((0, 1), sim.cached_cpu())
        result = sim.request(0, 1)
        self.assertEqual(result["status"], "miss")
        self.assertAlmostEqual(result["stall_ms"], 1.0)
        self.assertIn((0, 1), sim.cached_gpu())
        self.assertNotIn((0, 1), sim.cached_cpu())

    def test_layer_id_is_part_of_expert_key(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=2, cpu_capacity=2)
        sim.request(0, 7)
        sim.request(1, 7)
        self.assertIn((0, 7), sim.cached_gpu())
        self.assertIn((1, 7), sim.cached_gpu())

    def test_cpu_source_cannot_be_used_for_absent_expert_prefetch(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=2, cpu_capacity=2)
        with self.assertRaises(ValueError):
            sim.prefetch(0, 99, source="cpu")


    def test_request_waits_for_queued_transfer_finish_not_just_transfer_duration(self):
        sim = TimingAwareSimulator(self.costs, gpu_capacity=4, cpu_capacity=2)
        sim.prefetch(0, 1, source="disk")  # finishes at 10 ms
        sim.prefetch(0, 2, source="disk")  # queued, finishes at 20 ms
        result = sim.request(0, 2)

        self.assertEqual(result["status"], "prefetch_wait")
        self.assertAlmostEqual(result["stall_ms"], 20.0)
        self.assertAlmostEqual(sim.current_time_ms, 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
