"""
Core Snake RL Components

This package contains the foundational components for Snake reinforcement learning:
- Environment implementations (single and dual snake)
- State representation encoders
- Neural network architectures
- Utility functions
"""

from core.environment import SnakeEnv
from core.environment_vectorized import VectorizedSnakeEnv
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

__all__ = [
    'SnakeEnv',
    'VectorizedSnakeEnv',
    'DQN_MLP',
    'DQN_CNN',
    'DuelingDQN_MLP',
    'PPO_Actor_MLP',
    'PPO_Critic_MLP',
    'PPO_Actor_CNN',
    'PPO_Critic_CNN',
    'count_parameters'
]
