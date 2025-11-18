"""
Random Agent Baseline

Agent that selects random valid actions
"""

import numpy as np


class RandomAgent:
    """
    Random action agent for Snake

    Selects actions uniformly at random from the action space.
    Useful as a baseline to measure learning progress.
    """

    def __init__(self, action_space_type: str = 'absolute', seed: int = None):
        """
        Initialize random agent

        Args:
            action_space_type: 'absolute' (4 actions) or 'relative' (3 actions)
            seed: Random seed for reproducibility
        """
        self.action_space_type = action_space_type

        if action_space_type == 'absolute':
            self.n_actions = 4
        else:  # relative
            self.n_actions = 3

        if seed is not None:
            np.random.seed(seed)

    def get_action(self, env=None) -> int:
        """
        Get random action

        Args:
            env: SnakeEnv instance (not used, for API consistency)

        Returns:
            Random action
        """
        return np.random.randint(0, self.n_actions)
