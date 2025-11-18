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

    For feature-based state representation (11-dimensional input)
    """

    def __init__(
        self,
        input_dim: int = 11,
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
        input_dim: int = 11,
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


class PPO_Actor_MLP(nn.Module):
    """
    PPO Actor (Policy) Network with MLP

    Outputs action probabilities for stochastic policy
    """

    def __init__(
        self,
        input_dim: int = 11,
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
        input_dim: int = 11,
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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
