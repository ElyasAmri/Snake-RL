"""
Test optimization performance

Tests for BFS optimization and feature computation speed.
"""
import pytest
import time
from core.environment_vectorized import VectorizedSnakeEnv


class TestBFSOptimizations:
    """Test BFS and feature computation optimizations"""

    def test_selective_features_shape(self):
        """Test selective features produce correct shape"""
        env = VectorizedSnakeEnv(
            num_envs=8,
            use_flood_fill=True,
            use_selective_features=True
        )
        obs = env.reset()
        # Selective features: 10 base + 3 flood-fill + 5 selective = 18
        assert obs.shape == (8, 18), f"Expected (8, 18), got {obs.shape}"

    def test_enhanced_features_shape(self):
        """Test enhanced features produce correct shape"""
        env = VectorizedSnakeEnv(
            num_envs=8,
            use_flood_fill=True,
            use_enhanced_features=True
        )
        obs = env.reset()
        # Enhanced features: 10 base + 3 flood-fill + 10 enhanced = 23
        assert obs.shape == (8, 23), f"Expected (8, 23), got {obs.shape}"

    def test_selective_bfs_performance(self):
        """Test that selective BFS completes in reasonable time"""
        env = VectorizedSnakeEnv(
            num_envs=8,
            use_flood_fill=True,
            use_selective_features=True
        )

        start = time.time()
        for _ in range(10):
            env.reset()
        elapsed = time.time() - start

        # Should complete 10 resets in under 5 seconds
        assert elapsed < 5.0, f"Selective BFS too slow: {elapsed:.3f}s for 10 resets"

    def test_enhanced_bfs_performance(self):
        """Test that enhanced BFS completes in reasonable time"""
        env = VectorizedSnakeEnv(
            num_envs=8,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        start = time.time()
        for _ in range(10):
            env.reset()
        elapsed = time.time() - start

        # Should complete 10 resets in under 5 seconds
        assert elapsed < 5.0, f"Enhanced BFS too slow: {elapsed:.3f}s for 10 resets"

    def test_base_features_faster_than_enhanced(self):
        """Test that base features are faster than enhanced (no flood-fill overhead)"""
        env_base = VectorizedSnakeEnv(
            num_envs=8,
            use_flood_fill=False,
            use_enhanced_features=False
        )
        env_enhanced = VectorizedSnakeEnv(
            num_envs=8,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        # Warmup
        env_base.reset()
        env_enhanced.reset()

        # Time base features
        start = time.time()
        for _ in range(20):
            env_base.reset()
        base_time = time.time() - start

        # Time enhanced features
        start = time.time()
        for _ in range(20):
            env_enhanced.reset()
        enhanced_time = time.time() - start

        # Base should be faster (no flood-fill overhead)
        # Allow some margin for variance
        assert base_time < enhanced_time * 1.5, \
            f"Base features ({base_time:.3f}s) should be faster than enhanced ({enhanced_time:.3f}s)"


class TestFeatureComputationCorrectness:
    """Test that feature computations are correct after optimization"""

    def test_danger_features_binary(self):
        """Test that danger features are binary (0 or 1)"""
        env = VectorizedSnakeEnv(
            num_envs=16,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        for _ in range(5):
            obs = env.reset()
            # Danger features are indices 0-2
            danger = obs[:, 0:3]
            assert ((danger == 0) | (danger == 1)).all(), \
                "Danger features should be binary"

    def test_direction_features_one_hot(self):
        """Test that direction features are one-hot encoded"""
        env = VectorizedSnakeEnv(
            num_envs=16,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        for _ in range(5):
            obs = env.reset()
            # Direction features are indices 7-9 (3 bits for 4 directions)
            direction = obs[:, 7:10]
            # At most one feature should be 1 (or all zeros for direction 3)
            assert (direction.sum(dim=1) <= 1).all(), \
                "Direction should be one-hot encoded"

    def test_flood_fill_in_range(self):
        """Test that flood-fill features are in valid range"""
        env = VectorizedSnakeEnv(
            num_envs=16,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        for _ in range(5):
            obs = env.reset()
            # Flood-fill features are indices 10-12
            flood_fill = obs[:, 10:13]
            assert (flood_fill >= 0).all(), "Flood-fill should be >= 0"
            assert (flood_fill <= 1).all(), "Flood-fill should be <= 1"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
