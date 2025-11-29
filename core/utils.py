"""
Utility Functions for Snake RL

Includes:
- Replay buffer for experience storage
- Helper functions for training
- Metric tracking utilities
"""

import numpy as np
import torch
from collections import deque
from typing import Tuple, List, Optional
import random


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN

    Stores transitions (s, a, r, s', done) and supports random sampling
    """

    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for reproducibility
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

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
    ):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples to start training"""
        return len(self.buffer) >= min_size


class SumTree:
    """
    Sum Tree data structure for Prioritized Experience Replay

    Binary tree where parent node value = sum of children
    Allows O(log n) updates and O(log n) sampling
    """

    def __init__(self, capacity: int):
        """
        Initialize sum tree

        Args:
            capacity: Maximum number of leaf nodes (experiences)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for priority value s"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Get sum of all priorities"""
        return self.tree[0]

    def add(self, priority: float, data: tuple):
        """Add new experience with given priority"""
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority of experience at tree index"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, tuple]:
        """
        Get experience corresponding to priority value s

        Returns:
            (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer

    Samples experiences based on TD error magnitude
    Uses sum-tree for efficient priority-based sampling
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        seed: Optional[int] = None
    ):
        """
        Initialize prioritized replay buffer

        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
            seed: Random seed for reproducibility
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6  # Small constant to prevent zero priority

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_priority(self, td_error: float) -> float:
        """Convert TD error to priority"""
        return (abs(td_error) + self.epsilon) ** self.alpha

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = None
    ):
        """Add transition with priority based on TD error"""
        # Use max priority if TD error not provided
        if td_error is None:
            max_priority = np.max(self.tree.tree[-self.tree.capacity:])
            if max_priority == 0:
                max_priority = 1.0
            priority = max_priority
        else:
            priority = self._get_priority(td_error)

        data = (state, action, reward, next_state, done)
        self.tree.add(priority, data)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with importance sampling weights

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        batch = []
        indices = []
        priorities = []

        # Divide priority range into batch_size segments
        segment = self.tree.total() / batch_size

        # Calculate beta for current frame
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Calculate importance sampling weights
        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights = weights / weights.max()  # Normalize by max weight

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        """Return current buffer size"""
        return self.tree.n_entries

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples to start training"""
        return self.tree.n_entries >= min_size


class EpsilonScheduler:
    """
    Epsilon-greedy exploration scheduler

    Supports linear decay and exponential decay
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        decay_type: str = 'exponential'
    ):
        """
        Initialize epsilon scheduler

        Args:
            epsilon_start: Initial epsilon value
            epsilon_end: Final epsilon value
            epsilon_decay: Decay rate (for exponential) or decay steps (for linear)
            decay_type: 'exponential' or 'linear'
        """
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type
        self.step_count = 0

    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.epsilon

    def step(self):
        """Decay epsilon by one step"""
        if self.decay_type == 'exponential':
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        elif self.decay_type == 'linear':
            # Linear decay over epsilon_decay steps
            self.step_count += 1
            decay_fraction = min(1.0, self.step_count / self.epsilon_decay)
            self.epsilon = self.epsilon_start - decay_fraction * (self.epsilon_start - self.epsilon_end)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")


class MetricsTracker:
    """
    Track training metrics (rewards, losses, etc.)
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker

        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.losses = []

    def add_episode(self, reward: float, length: int, score: int):
        """Record episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_scores.append(score)

    def add_loss(self, loss: float):
        """Record training loss"""
        self.losses.append(loss)

    def get_recent_stats(self) -> dict:
        """Get statistics for recent episodes"""
        if not self.episode_rewards:
            return {}

        window = min(self.window_size, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_scores = self.episode_scores[-window:]

        stats = {
            'avg_reward': np.mean(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'avg_score': np.mean(recent_scores),
            'max_score': max(recent_scores),
            'episodes': len(self.episode_rewards)
        }

        if self.losses:
            recent_losses = self.losses[-window:]
            stats['avg_loss'] = np.mean(recent_losses)

        return stats

    def save_to_csv(self, filepath: str):
        """Save metrics to CSV file"""
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'length', 'score'])

            for i, (reward, length, score) in enumerate(
                zip(self.episode_rewards, self.episode_lengths, self.episode_scores)
            ):
                writer.writerow([i + 1, reward, length, score])


def set_seed(seed: int, strict_determinism: bool = False):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
        strict_determinism: If True, enable strict deterministic mode
                           (may raise errors for non-deterministic ops)
    """
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if strict_determinism:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_device() -> torch.device:
    """
    Get PyTorch device (CUDA if available, otherwise CPU)

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def save_model(model: torch.nn.Module, filepath: str, additional_info: dict = None):
    """
    Save model weights and optional metadata

    Args:
        model: PyTorch model
        filepath: Path to save file
        additional_info: Optional dictionary with hyperparameters, etc.
    """
    save_dict = {
        'model_state_dict': model.state_dict()
    }

    if additional_info:
        save_dict.update(additional_info)

    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: torch.nn.Module, filepath: str, device: torch.device = None) -> dict:
    """
    Load model weights and metadata

    Args:
        model: PyTorch model to load weights into
        filepath: Path to saved file
        device: Device to load model onto

    Returns:
        Dictionary with additional info (if saved)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Model loaded from {filepath}")

    # Return additional info
    info = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    return info
