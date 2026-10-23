"""Invariant tests for MultiTierExpertCache.

Run from the project root:
    python -m unittest scripts.test_multi_tier_cache_invariants -v

These tests verify cache state and simulated accounting, not real inference latency.
"""

import math
import unittest

from src.cache.multi_tier_cache import MultiTierExpertCache


def tier_keys(cache, tier_name):
    """Return a snapshot of keys stored in a named cache tier."""
    attr_candidates = {
        "gpu": ("gpu_cache", "gpu"),
        "cpu": ("cpu_cache", "cpu"),
    }[tier_name]
    for attr in attr_candidates:
        value = getattr(cache, attr, None)
        if value is not None:
            # OrderedDict and dict both expose keys(); list-like test doubles
            # are supported too.
            if hasattr(value, "keys"):
                return list(value.keys())
            try:
                return list(value)
            except TypeError:
                pass
    raise AssertionError(
        f"Could not locate {tier_name.upper()} cache storage. "
        f"Inspect MultiTierExpertCache and update tier_keys() to match its "
        f"actual attribute names."
    )


class TestMultiTierExpertCacheInvariants(unittest.TestCase):
    TOL = 1e-9

    def make_cache(self, gpu_capacity=2, cpu_capacity=2):
        return MultiTierExpertCache(gpu_capacity, cpu_capacity)

    def assert_capacity_and_unique_keys(self, cache):
        gpu_keys = tier_keys(cache, "gpu")
        cpu_keys = tier_keys(cache, "cpu")
        stats = cache.statistics()

        self.assertLessEqual(len(gpu_keys), stats["gpu_capacity"])
        self.assertLessEqual(len(cpu_keys), stats["cpu_capacity"])
        self.assertEqual(len(gpu_keys), len(set(gpu_keys)), "Duplicate keys in GPU")
        self.assertEqual(len(cpu_keys), len(set(cpu_keys)), "Duplicate keys in CPU")
        self.assertTrue(
            set(gpu_keys).isdisjoint(set(cpu_keys)),
            "An expert key must not exist in both GPU and CPU tiers.",
        )

    def assert_cost_invariant(self, cache):
        stats = cache.statistics()
        self.assertTrue(
            math.isclose(
                stats["total_latency_ms"],
                stats["demand_latency_ms"]
                + stats["prefetch_transfer_cost_ms"],
                abs_tol=self.TOL,
            ),
            "Total cost must equal demand cost plus prefetch transfer cost.",
        )

    def test_capacities_are_never_exceeded(self):
        cache = self.make_cache(gpu_capacity=2, cpu_capacity=2)
        operations = [
            ("request", 0, 1),
            ("request", 0, 2),
            ("request", 0, 3),
            ("request", 1, 1),
            ("request", 1, 2),
            ("request", 1, 3),
            ("prefetch", 0, 1),
            ("prefetch", 1, 1),
            ("request", 2, 5),
        ]
        for method_name, layer_id, expert_id in operations:
            getattr(cache, method_name)(layer_id, expert_id)
            self.assert_capacity_and_unique_keys(cache)
            self.assert_cost_invariant(cache)

    def test_lru_eviction_moves_oldest_gpu_key_to_cpu(self):
        cache = self.make_cache(gpu_capacity=2, cpu_capacity=2)
        cache.request(0, 1)
        cache.request(0, 2)
        self.assertEqual(tier_keys(cache, "gpu"), [(0, 1), (0, 2)])

        # Touch expert 1 so expert 2 becomes least recently used.
        cache.request(0, 1)
        cache.request(0, 3)

        gpu_keys = tier_keys(cache, "gpu")
        cpu_keys = tier_keys(cache, "cpu")
        self.assertEqual(gpu_keys, [(0, 1), (0, 3)])
        self.assertIn((0, 2), cpu_keys)
        self.assertNotIn((0, 2), gpu_keys)
        self.assert_capacity_and_unique_keys(cache)

    def test_cpu_resident_prefetch_promotes_key_to_gpu(self):
        cache = self.make_cache(gpu_capacity=1, cpu_capacity=2)
        cache.request(0, 1)
        cache.request(0, 2)  # evicts expert 1 to CPU
        self.assertIn((0, 1), tier_keys(cache, "cpu"))

        cache.prefetch(0, 1)
        self.assertIn((0, 1), tier_keys(cache, "gpu"))
        self.assertNotIn((0, 1), tier_keys(cache, "cpu"))
        self.assert_capacity_and_unique_keys(cache)

        stats = cache.statistics()
        self.assertTrue(
            math.isclose(
                stats["prefetch_transfer_cost_ms"], 1.0, abs_tol=self.TOL
            ),
            "CPU-to-GPU prefetch should incur the configured 1 ms transfer cost.",
        )
        self.assert_cost_invariant(cache)

    def test_same_expert_id_in_different_layers_is_distinct(self):
        cache = self.make_cache(gpu_capacity=2, cpu_capacity=2)
        cache.request(0, 7)
        cache.request(1, 7)

        gpu_keys = tier_keys(cache, "gpu")
        self.assertIn((0, 7), gpu_keys)
        self.assertIn((1, 7), gpu_keys)
        self.assertEqual(len(gpu_keys), 2)
        self.assert_capacity_and_unique_keys(cache)

    def test_clear_resets_storage_and_accounting(self):
        cache = self.make_cache(gpu_capacity=2, cpu_capacity=2)
        cache.request(0, 1)
        cache.request(0, 2)
        cache.prefetch(0, 3)
        cache.clear()

        stats = cache.statistics()
        self.assertEqual(tier_keys(cache, "gpu"), [])
        self.assertEqual(tier_keys(cache, "cpu"), [])
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["gpu_evictions"], 0)
        self.assertEqual(stats["cpu_evictions"], 0)
        self.assertEqual(stats["prefetches"], 0)
        self.assertEqual(stats["prefetch_hits"], 0)
        self.assertTrue(
            math.isclose(stats["total_latency_ms"], 0.0, abs_tol=self.TOL)
        )
        self.assertTrue(
            math.isclose(stats["demand_latency_ms"], 0.0, abs_tol=self.TOL)
        )
        self.assertTrue(
            math.isclose(
                stats["prefetch_transfer_cost_ms"], 0.0, abs_tol=self.TOL
            )
        )
        self.assert_capacity_and_unique_keys(cache)
        self.assert_cost_invariant(cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
