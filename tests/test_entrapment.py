"""
Unit tests for entrapment detection and reward modification feature.
"""

import pytest
import torch
import numpy as np
from core.environment_vectorized import VectorizedSnakeEnv, EntrapmentConfig


class TestEntrapmentConfig:
    """Test EntrapmentConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = EntrapmentConfig()
        assert config.enabled == False
        assert config.threshold == 0.30
        assert config.reward_step_entrapped == 0.02
        assert config.reward_food_entrapped == -5.0
        assert config.include_feature == False

    def test_custom_config(self):
        """Test custom configuration"""
        config = EntrapmentConfig(
            enabled=True,
            threshold=0.25,
            reward_step_entrapped=0.05,
            reward_food_entrapped=-10.0,
            include_feature=True
        )
        assert config.enabled == True
        assert config.threshold == 0.25
        assert config.reward_step_entrapped == 0.05
        assert config.reward_food_entrapped == -10.0
        assert config.include_feature == True


class TestEntrapmentDetection:
    """Test entrapment detection logic"""

    @pytest.fixture
    def env_with_entrapment(self):
        """Create environment with entrapment enabled"""
        config = EntrapmentConfig(enabled=True, threshold=0.30)
        return VectorizedSnakeEnv(
            num_envs=4,
            grid_size=10,
            use_flood_fill=True,
            entrapment_config=config,
            device=torch.device('cpu')
        )

    @pytest.fixture
    def env_without_entrapment(self):
        """Create environment without entrapment"""
        return VectorizedSnakeEnv(
            num_envs=4,
            grid_size=10,
            use_flood_fill=True,
            device=torch.device('cpu')
        )

    def test_entrapment_state_computation(self, env_with_entrapment):
        """Test that entrapment state is computed correctly"""
        env = env_with_entrapment
        env.reset(seed=67)

        # Get flood-fill features
        flood_fill_features = env._get_flood_fill_features_for_entrapment()

        # Compute entrapment state
        entrapped = env._compute_entrapment_state(flood_fill_features)

        assert entrapped.shape == (env.num_envs,)
        assert entrapped.dtype == torch.bool

    def test_flood_fill_features_shape(self, env_with_entrapment):
        """Test flood-fill features have correct shape"""
        env = env_with_entrapment
        env.reset(seed=67)

        flood_fill_features = env._get_flood_fill_features_for_entrapment()

        assert flood_fill_features.shape == (env.num_envs, 3)
        assert (flood_fill_features >= 0).all()
        assert (flood_fill_features <= 1).all()

    def test_high_flood_fill_not_entrapped(self, env_with_entrapment):
        """Test that high flood-fill values = not entrapped"""
        env = env_with_entrapment
        env.reset(seed=67)

        # At start, snake has lots of space (length 3, grid 10x10)
        flood_fill_features = env._get_flood_fill_features_for_entrapment()
        max_ff = flood_fill_features.max(dim=1)[0]

        # Most environments should have high free space at start
        # With length 3 on 10x10 grid, max free space should be > 90%
        assert (max_ff > 0.5).sum() >= env.num_envs // 2, \
            "At least half the envs should have >50% free space at start"

    def test_entrapment_disabled_returns_false(self, env_without_entrapment):
        """Test that disabled entrapment always returns False"""
        env = env_without_entrapment
        env.reset(seed=67)

        entrapped = env._compute_entrapment_state()

        assert entrapped.shape == (env.num_envs,)
        assert not entrapped.any(), "Disabled entrapment should return all False"


class TestEntrapmentRewards:
    """Test reward modification when entrapped"""

    @pytest.fixture
    def env_entrapment_enabled(self):
        """Create environment with entrapment rewards enabled"""
        config = EntrapmentConfig(
            enabled=True,
            threshold=0.30,
            reward_step_entrapped=0.02,
            reward_food_entrapped=-5.0
        )
        return VectorizedSnakeEnv(
            num_envs=4,
            grid_size=10,
            use_flood_fill=True,
            entrapment_config=config,
            device=torch.device('cpu')
        )

    def test_step_returns_soft_entrapped(self, env_entrapment_enabled):
        """Test that step info contains soft_entrapped flag"""
        env = env_entrapment_enabled
        env.reset(seed=67)

        actions = torch.zeros(env.num_envs, dtype=torch.long)
        _, _, _, info = env.step(actions)

        assert 'soft_entrapped' in info
        assert info['soft_entrapped'].shape == (env.num_envs,)
        assert info['soft_entrapped'].dtype == torch.bool

    def test_normal_step_reward_when_not_entrapped(self, env_entrapment_enabled):
        """Test that non-entrapped snakes get normal step reward"""
        env = env_entrapment_enabled
        env.reset(seed=67)

        # Take a step - at start, snakes should not be entrapped
        actions = torch.zeros(env.num_envs, dtype=torch.long)
        _, rewards, _, info = env.step(actions)

        not_entrapped = ~info['soft_entrapped']
        if not_entrapped.any():
            # Non-entrapped snakes should get close to normal reward
            # (reward_step + distance shaping)
            not_entrapped_rewards = rewards[not_entrapped]
            # Check rewards are in reasonable range for non-entrapped
            assert (not_entrapped_rewards < 0.1).all(), \
                "Non-entrapped rewards should be close to negative (step penalty)"

    def test_positive_step_reward_when_entrapped(self, env_entrapment_enabled):
        """Test that entrapped snakes get positive step reward"""
        env = env_entrapment_enabled
        env.reset(seed=67)

        # Take a step
        actions = torch.zeros(env.num_envs, dtype=torch.long)
        _, rewards, _, info = env.step(actions)

        entrapped = info['soft_entrapped']
        if entrapped.any():
            entrapped_rewards = rewards[entrapped]
            # Entrapped snakes should get positive reward
            assert (entrapped_rewards > 0).all(), \
                "Entrapped snakes should get positive step reward"


class TestEntrapmentFeature:
    """Test entrapped feature in observation"""

    def test_feature_dimension_without_entrapment_feature(self):
        """Test feature dimension without entrapped feature"""
        config = EntrapmentConfig(enabled=True, include_feature=False)
        env = VectorizedSnakeEnv(
            num_envs=2,
            grid_size=10,
            use_flood_fill=True,
            entrapment_config=config,
            device=torch.device('cpu')
        )

        obs = env.reset(seed=67)
        assert obs.shape[1] == 13  # 10 base + 3 flood-fill

    def test_feature_dimension_with_entrapment_feature(self):
        """Test that feature dimension increases with entrapment feature"""
        config = EntrapmentConfig(enabled=True, include_feature=True)
        env = VectorizedSnakeEnv(
            num_envs=2,
            grid_size=10,
            use_flood_fill=True,
            entrapment_config=config,
            device=torch.device('cpu')
        )

        obs = env.reset(seed=67)
        assert obs.shape[1] == 14  # 13 (with flood-fill) + 1 (entrapped)

    def test_entrapped_feature_values(self):
        """Test that entrapped feature is 0 or 1"""
        config = EntrapmentConfig(enabled=True, include_feature=True)
        env = VectorizedSnakeEnv(
            num_envs=8,
            grid_size=10,
            use_flood_fill=True,
            entrapment_config=config,
            device=torch.device('cpu')
        )

        obs = env.reset(seed=67)
        entrapped_feature = obs[:, -1]

        # Should be binary
        assert ((entrapped_feature == 0) | (entrapped_feature == 1)).all()


class TestBackwardCompatibility:
    """Test that feature is backward compatible when disabled"""

    def test_disabled_entrapment_same_obs_shape(self):
        """Test that disabled entrapment doesn't change observation shape"""
        # Environment without entrapment
        env_normal = VectorizedSnakeEnv(
            num_envs=4, grid_size=10, use_flood_fill=True,
            device=torch.device('cpu')
        )

        # Environment with entrapment disabled (default)
        config = EntrapmentConfig(enabled=False)
        env_disabled = VectorizedSnakeEnv(
            num_envs=4, grid_size=10, use_flood_fill=True,
            entrapment_config=config,
            device=torch.device('cpu')
        )

        obs_normal = env_normal.reset(seed=67)
        obs_disabled = env_disabled.reset(seed=67)

        assert obs_normal.shape == obs_disabled.shape

    def test_disabled_entrapment_same_observations(self):
        """Test that disabled entrapment produces same observations"""
        # Environment without entrapment
        env_normal = VectorizedSnakeEnv(
            num_envs=4, grid_size=10, use_flood_fill=True,
            device=torch.device('cpu')
        )

        # Environment with entrapment disabled (default)
        config = EntrapmentConfig(enabled=False)
        env_disabled = VectorizedSnakeEnv(
            num_envs=4, grid_size=10, use_flood_fill=True,
            entrapment_config=config,
            device=torch.device('cpu')
        )

        # Same seed should produce same observations
        obs_normal = env_normal.reset(seed=67)
        obs_disabled = env_disabled.reset(seed=67)

        assert torch.allclose(obs_normal, obs_disabled)

    def test_info_contains_soft_entrapped_even_when_disabled(self):
        """Test that info always contains soft_entrapped key"""
        config = EntrapmentConfig(enabled=False)
        env = VectorizedSnakeEnv(
            num_envs=4, grid_size=10, use_flood_fill=True,
            entrapment_config=config,
            device=torch.device('cpu')
        )

        env.reset(seed=67)
        actions = torch.zeros(env.num_envs, dtype=torch.long)
        _, _, _, info = env.step(actions)

        assert 'soft_entrapped' in info
        # When disabled, all should be False
        assert not info['soft_entrapped'].any()


class TestEntrapmentAutoEnableFloodFill:
    """Test that entrapment auto-enables flood-fill"""

    def test_entrapment_auto_enables_flood_fill(self):
        """Test that enabling entrapment auto-enables flood-fill"""
        config = EntrapmentConfig(enabled=True, include_feature=False)
        env = VectorizedSnakeEnv(
            num_envs=2,
            grid_size=10,
            use_flood_fill=False,  # Explicitly disabled
            entrapment_config=config,
            device=torch.device('cpu')
        )

        # Environment should have auto-enabled flood-fill
        assert env.use_flood_fill == True

        # Observation should have flood-fill features
        obs = env.reset(seed=67)
        assert obs.shape[1] >= 13  # At least 10 base + 3 flood-fill


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
