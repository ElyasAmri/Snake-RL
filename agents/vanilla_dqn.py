"""
Vanilla DQN Agent for Two-Snake Competitive Training

This is the simple, fast DQN implementation from the archive that achieved
excellent results (7.16/10 avg score, 39% win rate) in 4-8 hours of training.

Migrated from: archive/experiments/vanilla_dqn.py

Key features:
- Simple 2-layer MLP Q-network
- Experience replay buffer
- Epsilon-greedy exploration
- Optional target network
- Proven fast and effective
"""

from typing import Tuple, List, Optional, Deque, Dict, Any
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """
    Simple experience replay buffer for storing and sampling transitions.

    Stores (state, action, reward, next_state, done) tuples and samples
    random batches for training.
    """

    def __init__(self, capacity: int = 10000, seed: int = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for reproducibility
        """
        self.buffer: Deque = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.

    Simple feedforward network:
    Input -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Output
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128
    ):
        """
        Initialize Q-network.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_size: Number of neurons in hidden layers
        """
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: State tensor

        Returns:
            Q-values for each action
        """
        return self.network(state)


class VanillaDQNAgent:
    """
    Vanilla DQN agent with experience replay and epsilon-greedy exploration.

    This is the baseline RL algorithm. Features:
    - Q-network for value function approximation
    - Experience replay for breaking correlations
    - Epsilon-greedy for exploration
    - Target network (optional) for stability
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        use_target_network: bool = False,
        target_update_freq: int = 100,
        device: Optional[str] = None,
        seed: int = 67
    ):
        """
        Initialize Vanilla DQN agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_size: Hidden layer size
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay factor for epsilon
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            use_target_network: Whether to use target network
            target_update_freq: Steps between target network updates
            device: Device to use (cuda/cpu)
            seed: Random seed for reproducibility
        """
        # Basic attributes
        self.name = "Vanilla DQN"
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_target_network = use_target_network
        self.target_update_freq = target_update_freq

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Training stats
        self.training_steps = 0
        self.episodes = 0
        self.loss_history = []

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        if use_target_network:
            self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
        else:
            self.target_network = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation: choose action with highest Q-value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> dict:
        """
        Store transition and train if buffer has enough samples.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated

        Returns:
            Dictionary with training metrics
        """
        # Store transition in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        self.training_steps += 1

        # Train if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            if self.use_target_network and self.target_network is not None:
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            else:
                next_q_values = self.q_network(next_states).max(dim=1)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.use_target_network and self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Record loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)

        return {'loss': loss_value}

    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def save(self, path: str) -> None:
        """
        Save agent to file.

        Args:
            path: File path to save to
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'config': self.get_config()
        }

        if self.use_target_network and self.target_network is not None:
            checkpoint['target_network_state_dict'] = self.target_network.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load agent from file.

        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']

        if self.use_target_network and 'target_network_state_dict' in checkpoint:
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])

    def get_config(self) -> dict:
        """Get agent configuration."""
        return {
            'name': self.name,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'buffer_size': self.replay_buffer.buffer.maxlen,
            'use_target_network': self.use_target_network,
            'target_update_freq': self.target_update_freq,
            'device': str(self.device),
        }
