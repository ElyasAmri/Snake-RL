"""
Unit tests for neural network architectures
"""

import pytest
import torch
from core.networks import (
    DQN_MLP,
    DQN_CNN,
    DuelingDQN_MLP,
    PPO_Actor_MLP,
    PPO_Critic_MLP,
    PPO_Actor_CNN,
    PPO_Critic_CNN,
    count_parameters
)


class TestDQNNetworks:
    """Test DQN architectures"""

    def test_dqn_mlp_forward(self):
        """Test DQN MLP forward pass"""
        model = DQN_MLP(input_dim=11, output_dim=3)
        x = torch.randn(32, 11)
        out = model(x)

        assert out.shape == (32, 3)
        assert not torch.isnan(out).any()

    def test_dqn_cnn_forward(self):
        """Test DQN CNN forward pass"""
        model = DQN_CNN(grid_size=10, input_channels=3, output_dim=3)
        x = torch.randn(32, 10, 10, 3)
        out = model(x)

        assert out.shape == (32, 3)
        assert not torch.isnan(out).any()

    def test_dueling_dqn_forward(self):
        """Test Dueling DQN forward pass"""
        model = DuelingDQN_MLP(input_dim=11, output_dim=3)
        x = torch.randn(32, 11)
        out = model(x)

        assert out.shape == (32, 3)
        assert not torch.isnan(out).any()


class TestPPONetworks:
    """Test PPO architectures"""

    def test_ppo_actor_mlp(self):
        """Test PPO Actor MLP"""
        model = PPO_Actor_MLP(input_dim=11, output_dim=3)
        x = torch.randn(32, 11)

        logits = model(x)
        probs = model.get_action_probs(x)

        assert logits.shape == (32, 3)
        assert probs.shape == (32, 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-6)

    def test_ppo_critic_mlp(self):
        """Test PPO Critic MLP"""
        model = PPO_Critic_MLP(input_dim=11)
        x = torch.randn(32, 11)
        value = model(x)

        assert value.shape == (32, 1)
        assert not torch.isnan(value).any()

    def test_ppo_actor_cnn(self):
        """Test PPO Actor CNN"""
        model = PPO_Actor_CNN(grid_size=10, input_channels=3, output_dim=3)
        x = torch.randn(32, 10, 10, 3)

        logits = model(x)
        probs = model.get_action_probs(x)

        assert logits.shape == (32, 3)
        assert probs.shape == (32, 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-6)

    def test_ppo_critic_cnn(self):
        """Test PPO Critic CNN"""
        model = PPO_Critic_CNN(grid_size=10, input_channels=3)
        x = torch.randn(32, 10, 10, 3)
        value = model(x)

        assert value.shape == (32, 1)
        assert not torch.isnan(value).any()


class TestNetworkUtils:
    """Test network utilities"""

    def test_count_parameters(self):
        """Test parameter counting"""
        model = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128))
        param_count = count_parameters(model)

        assert param_count > 0
        assert param_count == 18435  # Known value for this architecture

    def test_gpu_compatibility(self):
        """Test models can be moved to GPU"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device('cuda')
        model = DQN_MLP(input_dim=11, output_dim=3)
        model = model.to(device)

        x = torch.randn(32, 11, device=device)
        out = model(x)

        assert out.device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
