"""
State Representation Encoders

Provides different state encoders for the Snake environment:
- Feature-based: 10-dimensional feature vector
- Grid-based: Multi-channel grid representation
- Normalization utilities
"""

import numpy as np
from collections import deque
from typing import Tuple, List


class FeatureEncoder:
    """
    Encodes Snake game state as a 10 or 13-dimensional feature vector

    Features (10-dimensional):
    [0-2]: Danger detection (straight, left, right)
    [3-6]: Food direction (up, right, down, left)
    [7-9]: Current direction (one-hot, 3 bits for 4 directions)

    Features (13-dimensional with flood-fill):
    [0-2]: Danger detection (straight, left, right)
    [3-6]: Food direction (up, right, down, left)
    [7-9]: Current direction (one-hot, 3 bits for 4 directions)
    [10-12]: Flood-fill free space (straight, right, left)

    Features (23-dimensional with all enhancements):
    [0-2]: Danger detection (straight, left, right)
    [3-6]: Food direction (up, right, down, left)
    [7-9]: Current direction (one-hot, 3 bits for 4 directions)
    [10-12]: Flood-fill free space (straight, right, left)
    [13-15]: Escape route count (straight, right, left) - number of safe adjacent cells
    [16-19]: Tail direction (up, right, down, left) - direction to tail
    [20]: Tail reachability (0-1) - can reach tail via flood-fill
    [21]: Distance to tail (normalized)
    [22]: Snake length ratio (current_length / max_length)
    """

    def __init__(self, grid_size: int, use_flood_fill: bool = False, use_enhanced_features: bool = False):
        self.grid_size = grid_size
        self.use_flood_fill = use_flood_fill
        self.use_enhanced_features = use_enhanced_features

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
            10-dimensional feature vector (or more with flood-fill/enhanced)
        """
        if grid_size is None:
            grid_size = self.grid_size

        head_x, head_y = snake[0]

        # Danger detection (3 features: straight, left, right)
        direction_deltas = {
            0: (0, -1),  # UP
            1: (1, 0),   # RIGHT
            2: (0, 1),   # DOWN
            3: (-1, 0)   # LEFT
        }

        dangers = []
        for offset in [0, -1, 1]:  # straight, left, right
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

        # Add flood-fill features if enabled
        if self.use_flood_fill:
            flood_fill_features = self._compute_flood_fill_features(
                snake, direction, grid_size
            )
            features = features + flood_fill_features

        # Add enhanced features if enabled
        if self.use_enhanced_features:
            enhanced_features = self._compute_enhanced_features(
                snake, direction, grid_size
            )
            features = features + enhanced_features

        return np.array(features, dtype=np.float32)

    def _is_within_bounds(self, pos: Tuple[int, int], grid_size: int) -> bool:
        """Check if position is within grid bounds"""
        x, y = pos
        return 0 <= x < grid_size and 0 <= y < grid_size

    def _compute_flood_fill_features(
        self,
        snake: List[Tuple[int, int]],
        direction: int,
        grid_size: int
    ) -> List[float]:
        """
        Compute flood-fill free space features for straight, right, and left directions.

        Args:
            snake: List of (x, y) positions, head at index 0
            direction: Current direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            grid_size: Grid size

        Returns:
            List of 3 flood-fill features [straight, right, left]
        """
        head_x, head_y = snake[0]
        direction_deltas = {
            0: (0, -1),  # UP
            1: (1, 0),   # RIGHT
            2: (0, 1),   # DOWN
            3: (-1, 0)   # LEFT
        }

        # Compute positions for straight, right, and left
        straight_dir = direction
        right_dir = (direction + 1) % 4
        left_dir = (direction - 1) % 4

        dx_s, dy_s = direction_deltas[straight_dir]
        dx_r, dy_r = direction_deltas[right_dir]
        dx_l, dy_l = direction_deltas[left_dir]

        straight_pos = (head_x + dx_s, head_y + dy_s)
        right_pos = (head_x + dx_r, head_y + dy_r)
        left_pos = (head_x + dx_l, head_y + dy_l)

        # Check if positions are safe before flood-fill
        snake_set = set(snake)
        free_space_straight = self._flood_fill_free_space(
            straight_pos, snake_set, grid_size
        ) if self._is_safe_position(straight_pos, snake_set, grid_size) else 0.0

        free_space_right = self._flood_fill_free_space(
            right_pos, snake_set, grid_size
        ) if self._is_safe_position(right_pos, snake_set, grid_size) else 0.0

        free_space_left = self._flood_fill_free_space(
            left_pos, snake_set, grid_size
        ) if self._is_safe_position(left_pos, snake_set, grid_size) else 0.0

        return [free_space_straight, free_space_right, free_space_left]

    def _is_safe_position(
        self,
        pos: Tuple[int, int],
        snake_set: set,
        grid_size: int
    ) -> bool:
        """Check if position is safe (within bounds and not in snake)"""
        return self._is_within_bounds(pos, grid_size) and pos not in snake_set

    def _flood_fill_free_space(
        self,
        start_pos: Tuple[int, int],
        snake_set: set,
        grid_size: int
    ) -> float:
        """
        Calculate reachable free space from a position using BFS flood-fill.

        This helps the agent avoid trapping itself by moving into areas
        with limited space.

        Args:
            start_pos: Starting position to flood-fill from
            snake_set: Set of snake body positions
            grid_size: Grid size

        Returns:
            Normalized free space (0-1), where 1 = entire grid accessible
        """
        # BFS flood-fill using deque for O(1) popleft
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)

        while queue:
            current = queue.popleft()
            x, y = current

            # Check all 4 neighbors (UP, RIGHT, DOWN, LEFT)
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor

                # Check if valid and not visited
                if (0 <= nx < grid_size and
                    0 <= ny < grid_size and
                    neighbor not in visited and
                    neighbor not in snake_set):
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Normalize by grid size (exclude snake body)
        max_free_space = grid_size * grid_size - len(snake_set)
        if max_free_space <= 0:
            return 0.0

        free_space_ratio = len(visited) / max_free_space
        return min(1.0, free_space_ratio)  # Clamp to [0, 1]

    def _compute_enhanced_features(
        self,
        snake: List[Tuple[int, int]],
        direction: int,
        grid_size: int
    ) -> List[float]:
        """
        Compute enhanced features: escape routes, tail info, and body awareness.

        Args:
            snake: List of (x, y) positions, head at index 0
            direction: Current direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            grid_size: Grid size

        Returns:
            List of enhanced features:
            [0-2]: Escape route count (straight, right, left)
            [3-6]: Tail direction (up, right, down, left)
            [7]: Tail reachability (0-1)
            [8]: Distance to tail (normalized)
            [9]: Snake length ratio
        """
        features = []

        # 1. Escape route detection (3 features)
        escape_routes = self._compute_escape_routes(snake, direction, grid_size)
        features.extend(escape_routes)

        # 2. Tail direction (4 features)
        tail_direction = self._compute_tail_direction(snake)
        features.extend(tail_direction)

        # 3. Tail reachability (1 feature)
        tail_reachability = self._compute_tail_reachability(snake, grid_size)
        features.append(tail_reachability)

        # 4. Distance to tail (1 feature, normalized)
        distance_to_tail = self._compute_distance_to_tail(snake, grid_size)
        features.append(distance_to_tail)

        # 5. Snake length ratio (1 feature)
        length_ratio = len(snake) / (grid_size * grid_size)
        features.append(length_ratio)

        return features

    def _compute_escape_routes(
        self,
        snake: List[Tuple[int, int]],
        direction: int,
        grid_size: int
    ) -> List[float]:
        """
        Count the number of safe adjacent cells (escape routes) for each direction.

        This complements flood-fill by providing immediate escape options.
        A higher count means more flexibility to maneuver.

        Returns:
            List of 3 normalized counts [straight, right, left] (0-1, divided by 4)
        """
        head_x, head_y = snake[0]
        snake_set = set(snake)

        direction_deltas = {
            0: (0, -1),  # UP
            1: (1, 0),   # RIGHT
            2: (0, 1),   # DOWN
            3: (-1, 0)   # LEFT
        }

        escape_counts = []

        # Check straight, right, and left
        for offset in [0, 1, -1]:
            check_dir = (direction + offset) % 4
            dx, dy = direction_deltas[check_dir]
            check_pos = (head_x + dx, head_y + dy)

            # If the position itself is not safe, no escape routes
            if not self._is_safe_position(check_pos, snake_set, grid_size):
                escape_counts.append(0.0)
                continue

            # Count safe adjacent cells from this position
            safe_count = 0
            cx, cy = check_pos
            for adj_dx, adj_dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                adj_pos = (cx + adj_dx, cy + adj_dy)
                if self._is_safe_position(adj_pos, snake_set, grid_size):
                    safe_count += 1

            # Normalize by max possible (4 directions)
            escape_counts.append(safe_count / 4.0)

        return escape_counts

    def _compute_tail_direction(
        self,
        snake: List[Tuple[int, int]]
    ) -> List[float]:
        """
        Compute direction to tail (4 features: up, right, down, left).

        Following the tail is often a safe strategy since the tail moves away.
        """
        if len(snake) < 2:
            return [0.0, 0.0, 0.0, 0.0]

        head_x, head_y = snake[0]
        tail_x, tail_y = snake[-1]

        return [
            float(tail_y < head_y),  # UP
            float(tail_x > head_x),  # RIGHT
            float(tail_y > head_y),  # DOWN
            float(tail_x < head_x)   # LEFT
        ]

    def _compute_tail_reachability(
        self,
        snake: List[Tuple[int, int]],
        grid_size: int
    ) -> float:
        """
        Check if tail is reachable via flood-fill from head.

        If tail is reachable, the snake has a safe path to follow.
        This is a high-impact feature for preventing self-traps.
        """
        if len(snake) < 2:
            return 1.0  # Always reachable if very short

        head = snake[0]
        tail = snake[-1]

        # Create snake set without tail (tail will move away)
        snake_set = set(snake[:-1])

        # BFS to check if tail is reachable using deque for O(1) popleft
        visited = set()
        queue = deque([head])
        visited.add(head)

        while queue:
            current = queue.popleft()
            if current == tail:
                return 1.0  # Tail is reachable

            x, y = current

            # Check all 4 neighbors
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor

                # Check if valid and not visited
                if (0 <= nx < grid_size and
                    0 <= ny < grid_size and
                    neighbor not in visited and
                    neighbor not in snake_set):
                    visited.add(neighbor)
                    queue.append(neighbor)

        return 0.0  # Tail is not reachable

    def _compute_distance_to_tail(
        self,
        snake: List[Tuple[int, int]],
        grid_size: int
    ) -> float:
        """
        Compute Manhattan distance to tail, normalized by grid diagonal.

        Knowing the distance to tail helps with planning safe paths.
        """
        if len(snake) < 2:
            return 0.0

        head_x, head_y = snake[0]
        tail_x, tail_y = snake[-1]

        manhattan_dist = abs(head_x - tail_x) + abs(head_y - tail_y)

        # Normalize by max possible Manhattan distance (grid diagonal)
        max_dist = 2 * (grid_size - 1)
        if max_dist == 0:
            return 0.0

        return manhattan_dist / max_dist


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
