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
        """Propagate priority change up the tree (iterative to avoid stack overflow)"""
        while idx != 0:
            parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

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


class NStepReplayBuffer:
    """
    N-Step Replay Buffer for Rainbow DQN

    Stores transitions and computes n-step returns:
    R_n = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^{n-1}*r_{t+n-1}

    The buffer accumulates n transitions before storing the full n-step transition.
    """

    def __init__(
        self,
        capacity: int,
        n_step: int = 3,
        gamma: float = 0.99,
        seed: Optional[int] = None
    ):
        """
        Initialize N-step replay buffer

        Args:
            capacity: Maximum number of n-step transitions to store
            n_step: Number of steps for n-step returns
            gamma: Discount factor
            seed: Random seed for reproducibility
        """
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _compute_n_step_return(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """
        Compute n-step return from accumulated transitions

        Returns:
            (first_state, first_action, n_step_reward, last_next_state, done)
        """
        # Get first transition
        first_state, first_action, _, _, _ = self.n_step_buffer[0]

        # Compute n-step discounted reward
        n_step_reward = 0.0
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * reward
            if done:
                break

        # Get last transition's next_state and done
        _, _, _, last_next_state, last_done = self.n_step_buffer[-1]

        return first_state, first_action, n_step_reward, last_next_state, last_done

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the n-step buffer

        When n transitions are accumulated, compute n-step return and store
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If episode ends early, flush remaining transitions
        if done:
            while len(self.n_step_buffer) > 0:
                n_step_transition = self._compute_n_step_return()
                self.buffer.append(n_step_transition)
                self.n_step_buffer.popleft()
        # Store n-step transition when buffer is full
        elif len(self.n_step_buffer) == self.n_step:
            n_step_transition = self._compute_n_step_return()
            self.buffer.append(n_step_transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of n-step transitions

        Returns:
            Tuple of (states, actions, n_step_rewards, next_states, dones) as tensors
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

    def reset_n_step_buffer(self):
        """Reset the n-step accumulation buffer (call at episode start)"""
        self.n_step_buffer.clear()

    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples to start training"""
        return len(self.buffer) >= min_size


class PrioritizedNStepReplayBuffer:
    """
    Prioritized N-Step Replay Buffer for Rainbow DQN

    Combines prioritized experience replay with n-step returns.
    Uses sum-tree for efficient priority-based sampling.
    Supports vectorized environments with per-env n-step buffers.
    """

    def __init__(
        self,
        capacity: int,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        num_envs: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialize prioritized n-step replay buffer

        Args:
            capacity: Maximum number of transitions to store
            n_step: Number of steps for n-step returns
            gamma: Discount factor
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
            num_envs: Number of parallel environments (for vectorized training)
            seed: Random seed for reproducibility
        """
        self.tree = SumTree(capacity)
        self.num_envs = num_envs
        # Separate n-step buffer for each environment
        self.n_step_buffers = [deque(maxlen=n_step) for _ in range(num_envs)]
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_priority(self, td_error: float) -> float:
        """Convert TD error to priority"""
        return (abs(td_error) + self.epsilon) ** self.alpha

    def _compute_n_step_return(self, env_id: int) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Compute n-step return from accumulated transitions for specific env"""
        n_step_buffer = self.n_step_buffers[env_id]
        first_state, first_action, _, _, _ = n_step_buffer[0]

        n_step_reward = 0.0
        for i, (_, _, reward, _, done) in enumerate(n_step_buffer):
            n_step_reward += (self.gamma ** i) * reward
            if done:
                break

        _, _, _, last_next_state, last_done = n_step_buffer[-1]

        return first_state, first_action, n_step_reward, last_next_state, last_done

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        env_id: int = 0,
        td_error: float = None
    ):
        """Add transition with n-step return computation and priority"""
        n_step_buffer = self.n_step_buffers[env_id]
        n_step_buffer.append((state, action, reward, next_state, done))

        if done:
            while len(n_step_buffer) > 0:
                n_step_transition = self._compute_n_step_return(env_id)
                self._add_to_tree(n_step_transition, td_error)
                n_step_buffer.popleft()
        elif len(n_step_buffer) == self.n_step:
            n_step_transition = self._compute_n_step_return(env_id)
            self._add_to_tree(n_step_transition, td_error)

    def _add_to_tree(self, transition: tuple, td_error: float = None):
        """Add transition to sum tree with priority"""
        if td_error is None:
            max_priority = np.max(self.tree.tree[-self.tree.capacity:])
            if max_priority == 0:
                max_priority = 1.0
            priority = max_priority
        else:
            priority = self._get_priority(td_error)

        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with importance sampling weights

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        batch = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size

        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        priorities = np.array(priorities)
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights = weights / weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)

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

    def reset_n_step_buffer(self, env_id: int = None):
        """Reset the n-step accumulation buffer for specific env or all envs"""
        if env_id is not None:
            self.n_step_buffers[env_id].clear()
        else:
            for buf in self.n_step_buffers:
                buf.clear()

    def __len__(self) -> int:
        """Return current buffer size"""
        return self.tree.n_entries

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples"""
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
    Track training metrics (rewards, losses, death causes, etc.)
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

        # Death cause tracking (per episode for cumulative plots)
        self.wall_deaths_per_episode = []
        self.self_deaths_per_episode = []
        self.entrapments_per_episode = []
        self.timeouts_per_episode = []

    def add_episode(self, reward: float, length: int, score: int, death_cause: str = 'timeout'):
        """
        Record episode metrics including death cause

        Args:
            reward: Total episode reward
            length: Episode length in steps
            score: Food items eaten
            death_cause: 'wall', 'self', 'entrapment', or 'timeout'
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_scores.append(score)

        # Track death cause (one-hot style for cumulative plotting)
        self.wall_deaths_per_episode.append(1 if death_cause == 'wall' else 0)
        self.self_deaths_per_episode.append(1 if death_cause == 'self' else 0)
        self.entrapments_per_episode.append(1 if death_cause == 'entrapment' else 0)
        self.timeouts_per_episode.append(1 if death_cause == 'timeout' else 0)

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

    def get_death_stats(self) -> dict:
        """Get death cause statistics"""
        total = len(self.episode_rewards)
        if total == 0:
            return {
                'wall_deaths': 0,
                'self_deaths': 0,
                'entrapments': 0,
                'timeouts': 0,
                'wall_death_rate': 0.0,
                'self_death_rate': 0.0,
                'entrapment_rate': 0.0,
                'timeout_rate': 0.0,
            }

        wall_deaths = sum(self.wall_deaths_per_episode)
        self_deaths = sum(self.self_deaths_per_episode)
        entrapments = sum(self.entrapments_per_episode)
        timeouts = sum(self.timeouts_per_episode)

        return {
            'wall_deaths': wall_deaths,
            'self_deaths': self_deaths,
            'entrapments': entrapments,
            'timeouts': timeouts,
            'wall_death_rate': wall_deaths / total,
            'self_death_rate': self_deaths / total,
            'entrapment_rate': entrapments / total,
            'timeout_rate': timeouts / total,
        }

    def get_cumulative_deaths(self) -> dict:
        """Get cumulative death counts for plotting"""
        return {
            'wall_cumulative': np.cumsum(self.wall_deaths_per_episode),
            'self_cumulative': np.cumsum(self.self_deaths_per_episode),
            'entrapment_cumulative': np.cumsum(self.entrapments_per_episode),
            'timeout_cumulative': np.cumsum(self.timeouts_per_episode),
        }

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


def project_distribution(
    next_dist: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
    gamma: float,
    n_step: int = 1
) -> torch.Tensor:
    """
    Project the target distribution onto the fixed support for Categorical DQN.

    This implements the distributional Bellman update:
    T_z = r + gamma^n * z (clipped to [v_min, v_max])

    Then projects onto the fixed support using linear interpolation.

    Args:
        next_dist: (batch_size, n_atoms) - distribution of next state-action
        rewards: (batch_size,) - n-step rewards
        dones: (batch_size,) - done flags
        support: (n_atoms,) - fixed support values [v_min, ..., v_max]
        gamma: discount factor
        n_step: number of steps for n-step returns

    Returns:
        (batch_size, n_atoms) - projected target distribution
    """
    batch_size = rewards.size(0)
    n_atoms = support.size(0)
    v_min = support[0].item()
    v_max = support[-1].item()
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # Compute projected support: T_z = r + gamma^n * z
    # Shape: (batch_size, n_atoms)
    gamma_n = gamma ** n_step
    t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma_n * support.unsqueeze(0)

    # Clip to valid range
    t_z = t_z.clamp(v_min, v_max)

    # Compute projection indices
    b = (t_z - v_min) / delta_z  # (batch_size, n_atoms)
    l = b.floor().long()  # Lower index
    u = b.ceil().long()   # Upper index

    # Clamp indices to valid range
    l = l.clamp(0, n_atoms - 1)
    u = u.clamp(0, n_atoms - 1)

    # Handle edge case when l == u (b is exactly an integer)
    # In this case, all probability should go to that single atom
    eq_mask = (l == u)
    l_offset = torch.zeros_like(l)
    l_offset[eq_mask] = 1
    l = (l - l_offset).clamp(0, n_atoms - 1)

    # Initialize projected distribution
    proj_dist = torch.zeros_like(next_dist)

    # Distribute probability mass
    # For each atom j in next_dist, distribute to atoms l[j] and u[j]
    offset = torch.arange(batch_size, device=rewards.device).unsqueeze(1) * n_atoms

    # Flatten for scatter_add
    proj_dist_flat = proj_dist.view(-1)

    # Lower projection: add p_j * (u - b) to atom l
    lower_proj = next_dist * (u.float() - b)
    proj_dist_flat.scatter_add_(0, (l + offset).view(-1), lower_proj.view(-1))

    # Upper projection: add p_j * (b - l) to atom u
    upper_proj = next_dist * (b - l.float())
    proj_dist_flat.scatter_add_(0, (u + offset).view(-1), upper_proj.view(-1))

    proj_dist = proj_dist_flat.view(batch_size, n_atoms)

    return proj_dist


def categorical_dqn_loss(
    current_dist: torch.Tensor,
    target_dist: torch.Tensor,
    support: torch.Tensor,
    weights: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross-entropy loss for Categorical DQN.

    Args:
        current_dist: (batch_size, n_atoms) - predicted distribution
        target_dist: (batch_size, n_atoms) - target distribution (from projection)
        support: (n_atoms,) - support values for computing Q-values
        weights: (batch_size,) - importance sampling weights (for PER)

    Returns:
        Tuple of (loss, td_errors for priority update)
    """
    # Cross-entropy loss: -sum(target * log(current))
    # Add small epsilon for numerical stability
    log_current = torch.log(current_dist + 1e-8)
    elementwise_loss = -(target_dist * log_current).sum(dim=1)

    # Compute proper TD errors for PER using Q-values (expected values of distributions)
    # Q = sum(p_i * z_i) where z_i are support values
    with torch.no_grad():
        current_q = (current_dist * support.unsqueeze(0)).sum(dim=1)
        target_q = (target_dist * support.unsqueeze(0)).sum(dim=1)
        td_errors = torch.abs(target_q - current_q)

    if weights is not None:
        loss = (weights * elementwise_loss).mean()
    else:
        loss = elementwise_loss.mean()

    return loss, td_errors
