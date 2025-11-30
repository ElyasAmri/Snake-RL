"""
Single Snake Environment - Gymnasium Compatible

Implements a classic Snake game environment with support for:
- Both absolute (4) and relative (3) action spaces
- Configurable grid sizes
- Feature-based and grid-based state representations
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Literal
from enum import IntEnum


class Direction(IntEnum):
    """Cardinal directions for snake movement"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ActionType(IntEnum):
    """Action space types"""
    # Absolute actions (4)
    ABS_UP = 0
    ABS_RIGHT = 1
    ABS_DOWN = 2
    ABS_LEFT = 3

    # Relative actions (3)
    REL_STRAIGHT = 0
    REL_LEFT = 1
    REL_RIGHT = 2


class SnakeEnv(gym.Env):
    """
    Single Snake Environment

    Observation:
        Depends on state_representation:
        - 'feature': 11-dimensional feature vector
        - 'grid': Multi-channel grid (H x W x 3)

    Actions:
        Depends on action_space_type:
        - 'absolute': 4 actions (UP, RIGHT, DOWN, LEFT)
        - 'relative': 3 actions (STRAIGHT, LEFT, RIGHT)

    Reward:
        - Food: +10
        - Death: -10
        - Step penalty: -0.01
        - Distance reward: +1 if closer to food, -1 if farther
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        grid_size: int = 10,
        action_space_type: Literal['absolute', 'relative'] = 'relative',
        state_representation: Literal['feature', 'grid'] = 'feature',
        max_steps: int = 1000,
        reward_food: float = 10.0,
        reward_death: float = -10.0,
        reward_step: float = -0.01,
        reward_distance: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__()

        # Configuration
        self.grid_size = grid_size
        self.action_space_type = action_space_type
        self.state_representation = state_representation
        self.max_steps = max_steps

        # Rewards
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_distance = reward_distance

        # Define action and observation spaces
        if action_space_type == 'absolute':
            self.action_space = spaces.Discrete(4)
        else:  # relative
            self.action_space = spaces.Discrete(3)

        if state_representation == 'feature':
            # 10-dimensional feature vector
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(10,), dtype=np.float32
            )
        else:  # grid
            # Multi-channel grid: [snake_head, snake_body, food]
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(grid_size, grid_size, 3),
                dtype=np.float32
            )

        # Game state
        self.snake = []  # List of (x, y) positions, head at index 0
        self.direction = Direction.RIGHT
        self.food = None
        self.steps = 0
        self.score = 0
        self.done = False
        self.prev_food_distance = 0

        # Random number generator
        self.np_random = None
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        # Initialize snake in center, length 3
        center = self.grid_size // 2
        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center)
        ]
        self.direction = Direction.RIGHT

        # Spawn food
        self._spawn_food()

        # Reset counters
        self.steps = 0
        self.score = 0
        self.done = False
        self.prev_food_distance = self._manhattan_distance(self.snake[0], self.food)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Convert action to direction
        new_direction = self._action_to_direction(action)

        # Prevent 180-degree turns (for absolute actions)
        if self.action_space_type == 'absolute':
            if self._is_opposite_direction(new_direction, self.direction):
                new_direction = self.direction  # Invalid move, keep current direction

        self.direction = new_direction

        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self._direction_to_delta(self.direction)
        new_head = (head_x + dx, head_y + dy)

        # Check collisions
        terminated = False
        reward = self.reward_step

        # Wall collision
        if not self._is_within_bounds(new_head):
            terminated = True
            reward = self.reward_death
        # Self collision
        elif new_head in self.snake:
            terminated = True
            reward = self.reward_death
        else:
            # Move snake
            self.snake.insert(0, new_head)

            # Check if food eaten
            if new_head == self.food:
                reward = self.reward_food
                self.score += 1
                self._spawn_food()
            else:
                # Remove tail if no food eaten
                self.snake.pop()

                # Distance-based reward
                if self.reward_distance:
                    current_distance = self._manhattan_distance(new_head, self.food)
                    if current_distance < self.prev_food_distance:
                        reward += 1.0
                    else:
                        reward -= 1.0
                    self.prev_food_distance = current_distance

        self.steps += 1

        # Check truncation (max steps)
        truncated = self.steps >= self.max_steps

        self.done = terminated or truncated

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _action_to_direction(self, action: int) -> Direction:
        """Convert action to direction based on action space type"""
        if self.action_space_type == 'absolute':
            # Absolute: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
            return Direction(action)
        else:
            # Relative: 0=STRAIGHT, 1=LEFT, 2=RIGHT
            if action == ActionType.REL_STRAIGHT:
                return self.direction
            elif action == ActionType.REL_LEFT:
                return Direction((self.direction - 1) % 4)
            else:  # REL_RIGHT
                return Direction((self.direction + 1) % 4)

    def _is_opposite_direction(self, dir1: Direction, dir2: Direction) -> bool:
        """Check if two directions are opposite (180 degrees apart)"""
        return (dir1 - dir2) % 4 == 2

    def _direction_to_delta(self, direction: Direction) -> Tuple[int, int]:
        """Convert direction to (dx, dy) movement delta"""
        deltas = {
            Direction.UP: (0, -1),
            Direction.RIGHT: (1, 0),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0)
        }
        return deltas[direction]

    def _is_within_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds"""
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _spawn_food(self):
        """Spawn food at random empty position"""
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))

        if empty_cells:
            self.food = empty_cells[self.np_random.integers(0, len(empty_cells))]
        else:
            # Grid is full (snake won!)
            self.food = None

    def _get_observation(self) -> np.ndarray:
        """Get current observation based on state representation type"""
        if self.state_representation == 'feature':
            return self._get_feature_observation()
        else:
            return self._get_grid_observation()

    def _get_feature_observation(self) -> np.ndarray:
        """
        Get 10-dimensional feature vector:
        [0-2]: Danger straight, left, right (binary)
        [3-6]: Food direction (up, right, down, left) (binary)
        [7-9]: Current direction (one-hot encoded)
        """
        head_x, head_y = self.snake[0]

        # Danger detection (3 features: straight, left, right)
        dangers = []
        for offset in [0, -1, 1]:  # straight, left, right
            check_dir = Direction((self.direction + offset) % 4)
            dx, dy = self._direction_to_delta(check_dir)
            next_pos = (head_x + dx, head_y + dy)

            # Danger if wall or body
            is_danger = (
                not self._is_within_bounds(next_pos) or
                next_pos in self.snake
            )
            dangers.append(float(is_danger))

        # Food direction (4 features)
        food_direction = [0.0, 0.0, 0.0, 0.0]
        if self.food:
            food_x, food_y = self.food
            if food_y < head_y:
                food_direction[0] = 1.0  # UP
            if food_x > head_x:
                food_direction[1] = 1.0  # RIGHT
            if food_y > head_y:
                food_direction[2] = 1.0  # DOWN
            if food_x < head_x:
                food_direction[3] = 1.0  # LEFT

        # Current direction (3 features - one-hot)
        # Only need 3 bits for 4 directions (last is implicit)
        direction_encoding = [0.0, 0.0, 0.0]
        if self.direction < 3:
            direction_encoding[self.direction] = 1.0

        features = dangers + food_direction + direction_encoding
        return np.array(features, dtype=np.float32)

    def _get_grid_observation(self) -> np.ndarray:
        """
        Get multi-channel grid representation:
        Channel 0: Snake head position
        Channel 1: Snake body positions
        Channel 2: Food position
        """
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        # Channel 0: Head
        if self.snake:
            head_x, head_y = self.snake[0]
            grid[head_y, head_x, 0] = 1.0

        # Channel 1: Body
        for x, y in self.snake[1:]:
            grid[y, x, 1] = 1.0

        # Channel 2: Food
        if self.food:
            food_x, food_y = self.food
            grid[food_y, food_x, 2] = 1.0

        return grid

    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        return {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake)
        }

    def render(self, mode: str = 'human'):
        """Render the environment (placeholder for now)"""
        if mode == 'human':
            # Simple text rendering
            grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

            # Place snake
            for i, (x, y) in enumerate(self.snake):
                if i == 0:
                    grid[y][x] = 'H'  # Head
                else:
                    grid[y][x] = 'o'  # Body

            # Place food
            if self.food:
                food_x, food_y = self.food
                grid[food_y][food_x] = 'F'

            # Print
            print('+' + '-' * self.grid_size + '+')
            for row in grid:
                print('|' + ''.join(row) + '|')
            print('+' + '-' * self.grid_size + '+')
            print(f"Score: {self.score}, Steps: {self.steps}, Length: {len(self.snake)}")

        return None
