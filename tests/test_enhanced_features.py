"""
Test enhanced features implementation

Tests for VectorizedSnakeEnv and FeatureEncoder with enhanced features.
"""
import pytest
import torch
import numpy as np
from core.environment_vectorized import VectorizedSnakeEnv
from core.state_representations import FeatureEncoder


class TestVectorizedEnvEnhancedFeatures:
    """Test VectorizedSnakeEnv with enhanced features"""

    def test_enhanced_features_shape(self):
        """Test that enhanced features produce correct observation shape"""
        env = VectorizedSnakeEnv(
            num_envs=8,
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        obs = env.reset()
        assert obs.shape == (8, 23), f"Expected shape (8, 23), got {obs.shape}"

    def test_step_maintains_shape(self):
        """Test that shape is maintained through multiple steps"""
        env = VectorizedSnakeEnv(
            num_envs=8,
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        obs = env.reset()
        for _ in range(10):
            actions = torch.randint(0, 3, (env.num_envs,), device=env.device)
            obs, rewards, dones, info = env.step(actions)
            assert obs.shape == (8, 23), f"Expected shape (8, 23) after step, got {obs.shape}"

    def test_feature_value_ranges(self):
        """Test that all features are in valid range [0, 1]"""
        env = VectorizedSnakeEnv(
            num_envs=16,
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        obs = env.reset()
        for _ in range(20):
            actions = torch.randint(0, 3, (env.num_envs,), device=env.device)
            obs, _, _, _ = env.step(actions)

            assert obs.min().item() >= 0.0, f"Feature values should be >= 0, got min {obs.min().item()}"
            assert obs.max().item() <= 1.0, f"Feature values should be <= 1, got max {obs.max().item()}"


class TestFeatureEncoderEnhanced:
    """Test FeatureEncoder with enhanced features"""

    def test_encoder_output_shape(self):
        """Test that encoder produces correct output shape"""
        encoder = FeatureEncoder(
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1  # RIGHT

        obs = encoder.encode(snake, food, direction)
        assert obs.shape == (23,), f"Expected shape (23,), got {obs.shape}"

    def test_feature_breakdown(self):
        """Test that feature breakdown is correct"""
        encoder = FeatureEncoder(
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1  # RIGHT

        obs = encoder.encode(snake, food, direction)

        # Verify feature dimensions
        assert len(obs[0:3]) == 3, "Danger detection should have 3 features"
        assert len(obs[3:7]) == 4, "Food direction should have 4 features"
        assert len(obs[7:10]) == 3, "Current direction should have 3 features"
        assert len(obs[10:13]) == 3, "Flood-fill should have 3 features"
        assert len(obs[13:16]) == 3, "Escape routes should have 3 features"
        assert len(obs[16:20]) == 4, "Tail direction should have 4 features"
        # obs[20] = tail reachability
        # obs[21] = distance to tail
        # obs[22] = snake length ratio

    def test_all_features_in_valid_range(self):
        """Test that all encoded features are in [0, 1]"""
        encoder = FeatureEncoder(
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1

        obs = encoder.encode(snake, food, direction)
        assert np.all(obs >= 0), "All features should be >= 0"
        assert np.all(obs <= 1), "All features should be <= 1"


class TestBackwardCompatibility:
    """Test backward compatibility with different feature configurations"""

    def test_base_features_shape(self):
        """Test base features (10-dim)"""
        env = VectorizedSnakeEnv(
            num_envs=4,
            grid_size=10,
            use_flood_fill=False,
            use_enhanced_features=False
        )
        obs = env.reset()
        assert obs.shape == (4, 10), f"Expected (4, 10), got {obs.shape}"

    def test_flood_fill_features_shape(self):
        """Test flood-fill features (13-dim)"""
        env = VectorizedSnakeEnv(
            num_envs=4,
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=False
        )
        obs = env.reset()
        assert obs.shape == (4, 13), f"Expected (4, 13), got {obs.shape}"

    def test_all_features_shape(self):
        """Test all features enabled (23-dim)"""
        env = VectorizedSnakeEnv(
            num_envs=4,
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )
        obs = env.reset()
        assert obs.shape == (4, 23), f"Expected (4, 23), got {obs.shape}"

    def test_encoder_base_features(self):
        """Test encoder with base features only"""
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=False, use_enhanced_features=False)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1
        obs = encoder.encode(snake, food, direction)
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"

    def test_encoder_flood_fill_features(self):
        """Test encoder with flood-fill features"""
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=True, use_enhanced_features=False)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1
        obs = encoder.encode(snake, food, direction)
        assert obs.shape == (13,), f"Expected (13,), got {obs.shape}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
