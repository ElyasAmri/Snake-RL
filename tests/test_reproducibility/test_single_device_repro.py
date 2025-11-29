"""
Test single device reproducibility - two runs on same device match

These tests verify that running the same operations twice on the same
device with the same seed produces identical results.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.utils import set_seed, get_device
from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DQN_MLP
from tests.test_reproducibility.utils import check_within_tolerance


class TestSingleDeviceReproducibility:
    """Verify identical runs on the same device"""

    @pytest.fixture
    def seed(self):
        return 67

    @pytest.fixture
    def device(self):
        return get_device()

    @pytest.fixture
    def small_env_config(self):
        return {
            "num_envs": 4,
            "grid_size": 10,
            "action_space_type": "relative",
            "state_representation": "feature"
        }

    def test_network_initialization_reproducibility(self, seed, device):
        """Same seed -> same initial network weights"""
        set_seed(seed)
        model1 = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(device)

        set_seed(seed)
        model2 = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(device)

        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert n1 == n2, f"Parameter names differ: {n1} vs {n2}"
            assert torch.equal(p1, p2), f"Parameter {n1} values differ"

    def test_network_forward_pass_reproducibility(self, seed, device):
        """Same weights + same input -> same output"""
        set_seed(seed)
        model = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(device)

        # Same input, two forward passes
        set_seed(seed)
        input_tensor = torch.rand(32, 11, device=device)

        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)

        assert torch.equal(output1, output2), "Forward pass outputs differ"

    def test_vectorized_env_reset_reproducibility(self, seed, device, small_env_config):
        """Same seed -> same initial environment state"""
        env1 = VectorizedSnakeEnv(**small_env_config, device=device)
        env2 = VectorizedSnakeEnv(**small_env_config, device=device)

        obs1 = env1.reset(seed=seed)
        obs2 = env2.reset(seed=seed)

        assert torch.equal(obs1, obs2), "Environment reset observations differ"

    def test_vectorized_env_step_sequence(self, seed, device, small_env_config):
        """Same actions -> same state transitions across multiple runs"""
        num_steps = 5  # Short sequence to avoid deaths causing divergence

        # Generate actions first
        set_seed(seed)
        actions = torch.randint(0, 3, (num_steps, small_env_config["num_envs"]), device=device)

        # First run
        set_seed(seed)
        env1 = VectorizedSnakeEnv(**small_env_config, device=device)
        obs1 = env1.reset(seed=seed)

        rewards1 = []
        for i in range(num_steps):
            obs1, r1, d1, _ = env1.step(actions[i])
            rewards1.append(r1.clone())

        # Second run
        set_seed(seed)
        env2 = VectorizedSnakeEnv(**small_env_config, device=device)
        obs2 = env2.reset(seed=seed)

        rewards2 = []
        for i in range(num_steps):
            obs2, r2, d2, _ = env2.step(actions[i])  # Same actions
            rewards2.append(r2.clone())

        # Compare final observations
        assert torch.equal(obs1, obs2), "Final observations differ after same action sequence"

        # Compare reward sequences
        for i, (r1, r2) in enumerate(zip(rewards1, rewards2)):
            assert torch.equal(r1, r2), f"Rewards differ at step {i}"

    def test_food_spawn_reproducibility(self, seed, device, small_env_config):
        """Food spawning is deterministic with same seed"""
        set_seed(seed)
        env1 = VectorizedSnakeEnv(**small_env_config, device=device)
        env1.reset(seed=seed)

        set_seed(seed)
        env2 = VectorizedSnakeEnv(**small_env_config, device=device)
        env2.reset(seed=seed)

        # Food positions should be identical (stored in self.foods tensor)
        assert torch.equal(env1.foods, env2.foods), "Food positions differ"

    def test_observation_generation_reproducibility(self, seed, device, small_env_config):
        """Feature observations match across runs"""
        env1 = VectorizedSnakeEnv(**small_env_config, device=device)
        env2 = VectorizedSnakeEnv(**small_env_config, device=device)

        obs1 = env1.reset(seed=seed)
        obs2 = env2.reset(seed=seed)

        # Observations should be exactly equal
        assert check_within_tolerance(obs1, obs2, "observation"), \
            "Observations differ beyond tolerance"

    def test_gradient_computation_reproducibility(self, seed, device):
        """Same loss -> same gradients"""
        set_seed(seed)
        model1 = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(device)

        set_seed(seed)
        model2 = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(device)

        # Same input and target
        set_seed(seed)
        input_tensor = torch.rand(32, 11, device=device)
        target = torch.rand(32, 3, device=device)

        # Compute gradients for model1
        model1.zero_grad()
        output1 = model1(input_tensor)
        loss1 = torch.nn.functional.mse_loss(output1, target)
        loss1.backward()

        # Compute gradients for model2 (need fresh input due to graph)
        set_seed(seed)
        input_tensor2 = torch.rand(32, 11, device=device)
        target2 = torch.rand(32, 3, device=device)

        model2.zero_grad()
        output2 = model2(input_tensor2)
        loss2 = torch.nn.functional.mse_loss(output2, target2)
        loss2.backward()

        # Compare gradients
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                assert check_within_tolerance(p1.grad, p2.grad, "gradients"), \
                    f"Gradients differ for {n1}"

    def test_replay_buffer_sampling_reproducibility(self, seed):
        """Same seed -> same sample order from replay buffer"""
        import random
        from collections import deque

        buffer = deque(maxlen=1000)
        for i in range(100):
            buffer.append(i)

        # First sampling
        set_seed(seed)
        sample1 = random.sample(list(buffer), 10)

        # Second sampling
        set_seed(seed)
        sample2 = random.sample(list(buffer), 10)

        assert sample1 == sample2, "Replay buffer sampling differs with same seed"

    def test_epsilon_greedy_reproducibility(self, seed, device):
        """Epsilon-greedy exploration is deterministic"""
        import random

        set_seed(seed)
        model = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(device)

        epsilon = 0.5
        num_decisions = 100

        # First run
        set_seed(seed)
        actions1 = []
        for _ in range(num_decisions):
            if random.random() < epsilon:
                actions1.append(random.randint(0, 2))
            else:
                obs = torch.rand(1, 11, device=device)
                with torch.no_grad():
                    q = model(obs)
                actions1.append(q.argmax().item())

        # Second run
        set_seed(seed)
        actions2 = []
        for _ in range(num_decisions):
            if random.random() < epsilon:
                actions2.append(random.randint(0, 2))
            else:
                obs = torch.rand(1, 11, device=device)
                with torch.no_grad():
                    q = model(obs)
                actions2.append(q.argmax().item())

        assert actions1 == actions2, "Epsilon-greedy actions differ with same seed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
