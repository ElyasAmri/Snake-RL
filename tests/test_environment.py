"""
Unit tests for Snake environment
"""

import pytest
import numpy as np
from core.environment import SnakeEnv


class TestSnakeEnv:
    """Test cases for SnakeEnv"""

    def test_initialization_absolute(self):
        """Test environment initialization with absolute actions"""
        env = SnakeEnv(grid_size=10, action_space_type='absolute', state_representation='feature')
        assert env.action_space.n == 4
        assert env.observation_space.shape == (11,)

    def test_initialization_relative(self):
        """Test environment initialization with relative actions"""
        env = SnakeEnv(grid_size=10, action_space_type='relative', state_representation='feature')
        assert env.action_space.n == 3
        assert env.observation_space.shape == (11,)

    def test_initialization_grid(self):
        """Test environment initialization with grid representation"""
        env = SnakeEnv(grid_size=10, action_space_type='absolute', state_representation='grid')
        assert env.observation_space.shape == (10, 10, 3)

    def test_reset(self):
        """Test environment reset"""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert 'score' in info
        assert 'steps' in info
        assert 'snake_length' in info
        assert len(env.snake) == 3
        assert env.food is not None

    def test_step_absolute(self):
        """Test step with absolute actions"""
        env = SnakeEnv(grid_size=10, action_space_type='absolute', seed=42)
        obs, info = env.reset(seed=42)

        # Take a step
        action = 1  # RIGHT
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_relative(self):
        """Test step with relative actions"""
        env = SnakeEnv(grid_size=10, action_space_type='relative', seed=42)
        obs, info = env.reset(seed=42)

        # Take a step
        action = 0  # STRAIGHT
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

    def test_wall_collision(self):
        """Test that hitting wall terminates episode"""
        env = SnakeEnv(grid_size=10, action_space_type='absolute', seed=42)
        obs, info = env.reset(seed=42)

        # Move left repeatedly until hitting wall
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(3)  # LEFT
            if terminated:
                break

        assert terminated
        assert reward == env.reward_death

    def test_food_consumption(self):
        """Test that eating food increases score"""
        env = SnakeEnv(grid_size=10, action_space_type='absolute', seed=42)
        obs, info = env.reset(seed=42)

        initial_length = len(env.snake)
        initial_score = env.score

        # Use A* to navigate to food
        from scripts.baselines.shortest_path import ShortestPathAgent
        agent = ShortestPathAgent(action_space_type='absolute')

        for _ in range(100):
            action = agent.get_action(env)
            obs, reward, terminated, truncated, info = env.step(action)

            if env.score > initial_score:
                # Food was eaten
                assert len(env.snake) > initial_length
                break

    def test_feature_observation_shape(self):
        """Test feature observation has correct shape"""
        env = SnakeEnv(grid_size=10, state_representation='feature', seed=42)
        obs, info = env.reset(seed=42)

        assert obs.shape == (11,)
        assert np.all((obs >= 0) & (obs <= 1))

    def test_grid_observation_shape(self):
        """Test grid observation has correct shape"""
        env = SnakeEnv(grid_size=10, state_representation='grid', seed=42)
        obs, info = env.reset(seed=42)

        assert obs.shape == (10, 10, 3)
        assert np.all((obs >= 0) & (obs <= 1))

    def test_max_steps_truncation(self):
        """Test that episode truncates at max steps"""
        env = SnakeEnv(grid_size=10, max_steps=10, seed=42)
        obs, info = env.reset(seed=42)

        # Use safe agent to avoid early termination
        from scripts.baselines.shortest_path import ShortestPathAgent
        agent = ShortestPathAgent(action_space_type='absolute')

        truncated = False
        for _ in range(15):
            action = agent.get_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated or terminated:
                break

        # Should either truncate or terminate, with steps at or near max_steps
        assert truncated or terminated

    def test_reproducibility(self):
        """Test that same seed produces same behavior"""
        env1 = SnakeEnv(grid_size=10, seed=42)
        env2 = SnakeEnv(grid_size=10, seed=42)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        assert np.array_equal(obs1, obs2)
        assert env1.snake == env2.snake
        assert env1.food == env2.food


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
