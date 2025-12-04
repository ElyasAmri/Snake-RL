"""
Neural Network Architectures for Snake RL

Implements:
- DQN with MLP (for feature-based state)
- DQN with CNN (for grid-based state)
- PPO Actor-Critic networks (both MLP and CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQN_MLP(nn.Module):
    """
    Deep Q-Network with Multi-Layer Perceptron

    For feature-based state representation (10-dimensional input)
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128)
    ):
        """
        Initialize DQN MLP

        Args:
            input_dim: Input feature dimension
            output_dim: Number of actions
            hidden_dims: Tuple of hidden layer sizes
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim) Q-values
        """
        return self.network(x)


class DQN_CNN(nn.Module):
    """
    Deep Q-Network with Convolutional Neural Network

    For grid-based state representation (H x W x 3)
    """

    def __init__(
        self,
        grid_size: int = 10,
        input_channels: int = 3,
        output_dim: int = 3
    ):
        """
        Initialize DQN CNN

        Args:
            grid_size: Size of game grid
            input_channels: Number of input channels
            output_dim: Number of actions
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate size after convolutions (no pooling, so size stays same)
        conv_output_size = grid_size * grid_size * 64

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, height, width, channels) grid tensor

        Returns:
            (batch_size, output_dim) Q-values
        """
        # Reshape from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DuelingDQN_MLP(nn.Module):
    """
    Dueling DQN Architecture with MLP

    Separates value and advantage streams for better learning
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128)
    ):
        """
        Initialize Dueling DQN MLP

        Args:
            input_dim: Input feature dimension
            output_dim: Number of actions
            hidden_dims: Tuple of hidden layer sizes
        """
        super().__init__()

        # Shared layers
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim) Q-values
        """
        shared = self.shared(x)

        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)

        # Combine value and advantage: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for Noisy DQN

    Implements factorized Gaussian noise as per the paper:
    "Noisy Networks for Exploration" (Fortunato et al., 2017)
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        """
        Initialize noisy linear layer

        Args:
            in_features: Input dimension
            out_features: Output dimension
            sigma_init: Initial noise standard deviation
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters: mean weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register noise buffers (not parameters, won't be trained)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1.0 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Factorized Gaussian noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Generate scaled noise for factorized Gaussian"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights"""
        if self.training:
            # Use noisy weights during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use mean weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisyDQN_MLP(nn.Module):
    """
    Noisy Deep Q-Network with MLP

    Uses NoisyLinear layers instead of regular Linear layers.
    Removes need for epsilon-greedy exploration.
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128),
        sigma_init: float = 0.5
    ):
        """
        Initialize Noisy DQN MLP

        Args:
            input_dim: Input feature dimension
            output_dim: Number of actions
            hidden_dims: Tuple of hidden layer sizes
            sigma_init: Initial noise standard deviation
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers with NoisyLinear
        for hidden_dim in hidden_dims:
            layers.append(NoisyLinear(prev_dim, hidden_dim, sigma_init))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer with NoisyLinear
        layers.append(NoisyLinear(prev_dim, output_dim, sigma_init))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        for module in self.network.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PPO_Actor_MLP(nn.Module):
    """
    PPO Actor (Policy) Network with MLP

    Outputs action probabilities for stochastic policy
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128)
    ):
        """
        Initialize PPO Actor MLP

        Args:
            input_dim: Input feature dimension
            output_dim: Number of actions
            hidden_dims: Tuple of hidden layer sizes
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (logits)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim) action logits
        """
        logits = self.network(x)
        return logits

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class PPO_Critic_MLP(nn.Module):
    """
    PPO Critic (Value) Network with MLP

    Outputs state value estimate
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: Tuple[int, ...] = (128, 128)
    ):
        """
        Initialize PPO Critic MLP

        Args:
            input_dim: Input feature dimension
            hidden_dims: Tuple of hidden layer sizes
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, 1) state value
        """
        return self.network(x)


class PPO_Actor_CNN(nn.Module):
    """
    PPO Actor Network with CNN

    For grid-based state representation
    """

    def __init__(
        self,
        grid_size: int = 10,
        input_channels: int = 3,
        output_dim: int = 3
    ):
        """
        Initialize PPO Actor CNN

        Args:
            grid_size: Size of game grid
            input_channels: Number of input channels
            output_dim: Number of actions
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        conv_output_size = grid_size * grid_size * 64

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns logits"""
        # Reshape from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class PPO_Critic_CNN(nn.Module):
    """
    PPO Critic Network with CNN

    For grid-based state representation
    """

    def __init__(
        self,
        grid_size: int = 10,
        input_channels: int = 3
    ):
        """
        Initialize PPO Critic CNN

        Args:
            grid_size: Size of game grid
            input_channels: Number of input channels
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        conv_output_size = grid_size * grid_size * 64

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns state value"""
        # Reshape from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        value = self.fc2(x)

        return value


class CategoricalDQN_MLP(nn.Module):
    """
    Categorical (Distributional) DQN with MLP

    Instead of learning Q(s,a) directly, learns the distribution of returns.
    Uses N_ATOMS discrete supports to approximate the value distribution.

    Based on "A Distributional Perspective on Reinforcement Learning"
    (Bellemare et al., 2017)
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128),
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        """
        Initialize Categorical DQN MLP

        Args:
            input_dim: Input feature dimension
            output_dim: Number of actions
            hidden_dims: Tuple of hidden layer sizes
            n_atoms: Number of atoms in the distribution
            v_min: Minimum value of support
            v_max: Maximum value of support
        """
        super().__init__()

        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support: z_i = v_min + i * delta_z
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Output layer: for each action, output n_atoms probabilities
        self.fc_out = nn.Linear(prev_dim, output_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns action-value distributions

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim, n_atoms) probability distributions
        """
        features = self.features(x)
        logits = self.fc_out(features)

        # Reshape to (batch_size, n_actions, n_atoms)
        logits = logits.view(-1, self.output_dim, self.n_atoms)

        # Apply softmax over atoms dimension to get probabilities
        probs = F.softmax(logits, dim=-1)

        return probs

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values as expected value of distributions

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim) Q-values
        """
        probs = self.forward(x)
        # Q(s,a) = sum_i z_i * p_i(s,a)
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values


class NoisyCategoricalDQN_MLP(nn.Module):
    """
    Noisy Categorical DQN with MLP

    Combines distributional RL with noisy networks for exploration.
    This is a key component of Rainbow DQN.
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128),
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma_init: float = 0.5
    ):
        """
        Initialize Noisy Categorical DQN MLP

        Args:
            input_dim: Input feature dimension
            output_dim: Number of actions
            hidden_dims: Tuple of hidden layer sizes
            n_atoms: Number of atoms in the distribution
            v_min: Minimum value of support
            v_max: Maximum value of support
            sigma_init: Initial noise standard deviation
        """
        super().__init__()

        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Build network with NoisyLinear layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(NoisyLinear(prev_dim, hidden_dim, sigma_init))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # Output layer with NoisyLinear
        self.fc_out = NoisyLinear(prev_dim, output_dim * n_atoms, sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns action-value distributions

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim, n_atoms) probability distributions
        """
        features = self.features(x)
        logits = self.fc_out(features)

        logits = logits.view(-1, self.output_dim, self.n_atoms)
        probs = F.softmax(logits, dim=-1)

        return probs

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values as expected value of distributions

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim) Q-values
        """
        probs = self.forward(x)
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        for module in self.features.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        self.fc_out.reset_noise()


class DuelingCategoricalDQN_MLP(nn.Module):
    """
    Dueling Categorical DQN with MLP

    Combines distributional RL with dueling architecture.
    Separates value and advantage streams, each outputting distributions.
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128),
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        """
        Initialize Dueling Categorical DQN MLP

        Args:
            input_dim: Input feature dimension
            output_dim: Number of actions
            hidden_dims: Tuple of hidden layer sizes
            n_atoms: Number of atoms in the distribution
            v_min: Minimum value of support
            v_max: Maximum value of support
        """
        super().__init__()

        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Shared layers
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        # Value stream (outputs distribution over single value)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], n_atoms)
        )

        # Advantage stream (outputs distributions for each action)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], output_dim * n_atoms)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns action-value distributions

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim, n_atoms) probability distributions
        """
        batch_size = x.size(0)
        shared = self.shared(x)

        # Value distribution: (batch_size, n_atoms)
        value = self.value_stream(shared)
        value = value.view(batch_size, 1, self.n_atoms)

        # Advantage distributions: (batch_size, n_actions, n_atoms)
        advantage = self.advantage_stream(shared)
        advantage = advantage.view(batch_size, self.output_dim, self.n_atoms)

        # Combine: Q = V + (A - mean(A)) in log space before softmax
        q_logits = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Apply softmax over atoms
        probs = F.softmax(q_logits, dim=-1)

        return probs

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values as expected value of distributions"""
        probs = self.forward(x)
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values


class RainbowDQN_MLP(nn.Module):
    """
    Rainbow DQN with MLP

    Combines all DQN improvements:
    - Double DQN (handled in training)
    - Prioritized Experience Replay (handled in buffer)
    - Dueling Architecture
    - Noisy Networks
    - N-step Returns (handled in buffer)
    - Distributional RL (Categorical)

    This network implements: Dueling + Noisy + Distributional
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128),
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma_init: float = 0.5
    ):
        """
        Initialize Rainbow DQN MLP

        Args:
            input_dim: Input feature dimension
            output_dim: Number of actions
            hidden_dims: Tuple of hidden layer sizes
            n_atoms: Number of atoms in the distribution
            v_min: Minimum value of support
            v_max: Maximum value of support
            sigma_init: Initial noise standard deviation for NoisyLinear
        """
        super().__init__()

        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Shared feature extraction with NoisyLinear
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(NoisyLinear(prev_dim, hidden_dim, sigma_init))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        # Value stream with NoisyLinear
        self.value_hidden = NoisyLinear(prev_dim, hidden_dims[-1], sigma_init)
        self.value_out = NoisyLinear(hidden_dims[-1], n_atoms, sigma_init)

        # Advantage stream with NoisyLinear
        self.advantage_hidden = NoisyLinear(prev_dim, hidden_dims[-1], sigma_init)
        self.advantage_out = NoisyLinear(hidden_dims[-1], output_dim * n_atoms, sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns action-value distributions

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim, n_atoms) probability distributions
        """
        batch_size = x.size(0)
        shared = self.shared(x)

        # Value stream
        value = F.relu(self.value_hidden(shared))
        value = self.value_out(value)
        value = value.view(batch_size, 1, self.n_atoms)

        # Advantage stream
        advantage = F.relu(self.advantage_hidden(shared))
        advantage = self.advantage_out(advantage)
        advantage = advantage.view(batch_size, self.output_dim, self.n_atoms)

        # Combine using dueling formula
        q_logits = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Softmax over atoms to get probability distribution
        probs = F.softmax(q_logits, dim=-1)

        return probs

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values as expected value of distributions

        Args:
            x: (batch_size, input_dim) state tensor

        Returns:
            (batch_size, output_dim) Q-values
        """
        probs = self.forward(x)
        q_values = (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return q_values

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers"""
        for module in self.shared.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
