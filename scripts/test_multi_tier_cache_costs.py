"""Automated cost-accounting tests for MultiTierExpertCache.

Copy this file to scripts/test_multi_tier_cache_costs.py, then run:
    python -m unittest scripts.test_multi_tier_cache_costs -v

These tests verify modeled costs, not real inference latency.
"""
import math
import unittest

from src.cache.multi_tier_cache import MultiTierExpertCache


class TestMultiTierExpertCacheCosts(unittest.TestCase):
    TOL = 1e-9

    def make_cache(self, gpu_capacity=2, cpu_capacity=2):
        return MultiTierExpertCache(gpu_capacity, cpu_capacity)

    def assert_costs(self, stats, demand, prefetch, total):
        self.assertTrue(math.isclose(stats["demand_latency_ms"], demand, abs_tol=self.TOL),
                        f"Expected demand cost {demand}, got {stats['demand_latency_ms']}")
        self.assertTrue(math.isclose(stats["prefetch_transfer_cost_ms"], prefetch, abs_tol=self.TOL),
                        f"Expected prefetch cost {prefetch}, got {stats['prefetch_transfer_cost_ms']}")
        self.assertTrue(math.isclose(stats["total_latency_ms"], total, abs_tol=self.TOL),
                        f"Expected total cost {total}, got {stats['total_latency_ms']}")
        self.assertTrue(math.isclose(
            stats["total_latency_ms"],
            stats["demand_latency_ms"] + stats["prefetch_transfer_cost_ms"],
            abs_tol=self.TOL),
            "Total cost must equal demand cost plus prefetch transfer cost.")

    def test_disk_demand_request_is_demand_cost_only(self):
        cache = self.make_cache()
        cache.request(0, 1)
        stats = cache.statistics()
        self.assertEqual(stats["total_requests"], 1)
        self.assert_costs(stats, demand=10.0, prefetch=0.0, total=10.0)

    def test_disk_prefetch_is_prefetch_cost_only(self):
        cache = self.make_cache()
        cache.prefetch(0, 1)
        stats = cache.statistics()
        self.assertEqual(stats["total_requests"], 0)
        self.assert_costs(stats, demand=0.0, prefetch=10.0, total=10.0)

    def test_cpu_to_gpu_prefetch_uses_cpu_transfer_cost(self):
        cache = self.make_cache(gpu_capacity=1, cpu_capacity=2)
        cache.request(0, 1)
        cache.request(0, 2)  # Evicts (0, 1) from GPU into CPU.
        cache.prefetch(0, 1)  # CPU -> GPU should cost 1 ms.
        stats = cache.statistics()
        self.assertEqual(stats["total_requests"], 2)
        self.assert_costs(stats, demand=20.0, prefetch=1.0, total=21.0)

    def test_prefetch_of_gpu_resident_expert_has_no_transfer_cost(self):
        cache = self.make_cache()
        cache.request(0, 1)
        before = cache.statistics()
        cache.prefetch(0, 1)
        after = cache.statistics()
        self.assertEqual(after["total_requests"], before["total_requests"])
        self.assert_costs(after, demand=10.0, prefetch=0.0, total=10.0)

    def test_clear_resets_cost_counters(self):
        cache = self.make_cache()
        cache.request(0, 1)
        cache.prefetch(0, 2)
        cache.clear()
        stats = cache.statistics()
        self.assertEqual(stats["total_requests"], 0)
        self.assert_costs(stats, demand=0.0, prefetch=0.0, total=0.0)

    def test_cost_invariant_survives_mixed_operations(self):
        cache = self.make_cache(gpu_capacity=1, cpu_capacity=2)
        cache.request(0, 1)
        cache.request(0, 2)
        cache.prefetch(0, 1)
        cache.request(0, 1)
        stats = cache.statistics()
        self.assertTrue(math.isclose(
            stats["total_latency_ms"],
            stats["demand_latency_ms"] + stats["prefetch_transfer_cost_ms"],
            abs_tol=self.TOL))


if __name__ == "__main__":
    unittest.main(verbosity=2)
