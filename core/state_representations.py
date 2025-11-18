"""
State Representation Encoders

Provides different state encoders for the Snake environment:
- Feature-based: 11-dimensional feature vector
- Grid-based: Multi-channel grid representation
- Normalization utilities
"""

import numpy as np
from typing import Tuple, List


class FeatureEncoder:
    """
    Encodes Snake game state as an 11-dimensional feature vector

    Features:
    [0-3]: Danger detection (straight, left, right, back)
    [4-7]: Food direction (up, right, down, left)
    [8-10]: Current direction (one-hot, 3 bits for 4 directions)
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def encode(
        self,
        snake: List[Tuple[int, int]],
        food: Tuple[int, int],
        direction: int,
        grid_size: int = None
    ) -> np.ndarray:
        """
        Encode game state as feature vector

        Args:
            snake: List of (x, y) positions, head at index 0
            food: (x, y) position of food
            direction: Current direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            grid_size: Grid size (optional, uses self.grid_size if None)

        Returns:
            11-dimensional feature vector
        """
        if grid_size is None:
            grid_size = self.grid_size

        head_x, head_y = snake[0]

        # Danger detection (4 features: straight, left, right, back)
        direction_deltas = {
            0: (0, -1),  # UP
            1: (1, 0),   # RIGHT
            2: (0, 1),   # DOWN
            3: (-1, 0)   # LEFT
        }

        dangers = []
        for offset in [0, -1, 1, 2]:  # straight, left, right, back
            check_dir = (direction + offset) % 4
            dx, dy = direction_deltas[check_dir]
            next_pos = (head_x + dx, head_y + dy)

            # Danger if wall or body
            is_danger = (
                not self._is_within_bounds(next_pos, grid_size) or
                next_pos in snake
            )
            dangers.append(float(is_danger))

        # Food direction (4 features: up, right, down, left)
        food_x, food_y = food
        food_direction = [
            float(food_y < head_y),  # UP
            float(food_x > head_x),  # RIGHT
            float(food_y > head_y),  # DOWN
            float(food_x < head_x)   # LEFT
        ]

        # Current direction (3 features - one-hot)
        # Only need 3 bits for 4 directions (last is implicit)
        direction_encoding = [0.0, 0.0, 0.0]
        if direction < 3:
            direction_encoding[direction] = 1.0

        features = dangers + food_direction + direction_encoding
        return np.array(features, dtype=np.float32)

    def _is_within_bounds(self, pos: Tuple[int, int], grid_size: int) -> bool:
        """Check if position is within grid bounds"""
        x, y = pos
        return 0 <= x < grid_size and 0 <= y < grid_size


class GridEncoder:
    """
    Encodes Snake game state as a multi-channel grid

    Channels:
    0: Snake head position
    1: Snake body positions
    2: Food position
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def encode(
        self,
        snake: List[Tuple[int, int]],
        food: Tuple[int, int],
        direction: int = None,  # Not used in grid encoding
        grid_size: int = None
    ) -> np.ndarray:
        """
        Encode game state as multi-channel grid

        Args:
            snake: List of (x, y) positions, head at index 0
            food: (x, y) position of food
            direction: Not used (for API consistency)
            grid_size: Grid size (optional, uses self.grid_size if None)

        Returns:
            (grid_size, grid_size, 3) array
        """
        if grid_size is None:
            grid_size = self.grid_size

        grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

        # Channel 0: Head
        if snake:
            head_x, head_y = snake[0]
            grid[head_y, head_x, 0] = 1.0

        # Channel 1: Body
        for x, y in snake[1:]:
            grid[y, x, 1] = 1.0

        # Channel 2: Food
        if food:
            food_x, food_y = food
            grid[food_y, food_x, 2] = 1.0

        return grid


def normalize_features(features: np.ndarray, feature_type: str = 'binary') -> np.ndarray:
    """
    Normalize features to appropriate range

    Args:
        features: Input features
        feature_type: 'binary' (already in [0, 1]) or 'continuous' (normalize to [-1, 1])

    Returns:
        Normalized features
    """
    if feature_type == 'binary':
        # Already in [0, 1], no normalization needed
        return features
    elif feature_type == 'continuous':
        # Normalize to [-1, 1]
        feature_min = features.min()
        feature_max = features.max()
        if feature_max - feature_min > 0:
            return 2 * (features - feature_min) / (feature_max - feature_min) - 1
        else:
            return features
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    """
    Normalize grid values to [0, 1]

    Grid is already in [0, 1] for binary occupancy, but this function
    can be extended for other grid representations.
    """
    return grid.astype(np.float32)
