"""
Classic Two-Snake Competitive Environment

This is the simple, fast, CPU-based implementation from the archive that achieved
excellent results (7.16/10 avg score, 39% win rate) in 4-8 hours of training.

Migrated from: archive/core/snake_env.py (TwoSnakeCompetitiveEnv class)

Key differences from vectorized version:
- Single environment (not batched)
- CPU-based NumPy operations
- 20-feature state representation (vs 35)
- Simple gymnasium interface
- Proven fast and effective
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any


class TwoSnakeCompetitiveEnv(gym.Env):
    """
    Two-Snake Competitive Multi-Agent Environment

    Objective: Two snakes compete to reach a target score first

    Features:
    - Two snakes in the same 10x10 grid
    - Competitive: first to reach target_score wins
    - If one snake dies, the other continues
    - Both snakes see opponent positions in their observations

    Win conditions:
    - Reach target_score food items first
    - Survive while opponent dies

    Episode ends when:
    - One snake reaches target_score (winner declared)
    - Both snakes die (draw)
    - Max steps reached (timeout)

    State representation per agent (20 features):
    - Danger detection (3): straight, right, left
    - Own direction (4): up, right, down, left (one-hot)
    - Food position (2): relative x, y distance
    - Own snake length normalized (1)
    - Opponent head distance (2): relative x, y
    - Opponent direction (4): one-hot
    - Opponent length normalized (1)
    - Score difference (1): own_score - opponent_score (normalized)
    - Opponent danger zones (2): opponent ahead straight, opponent nearby

    Rewards:
    - +10 for eating food
    - +100 for winning (reaching target first)
    - -100 for dying
    - +50 if opponent dies and agent still alive
    - -0.01 per step (efficiency incentive)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Action constants (same as single-snake)
    STRAIGHT = 0
    RIGHT_TURN = 1
    LEFT_TURN = 2

    # Direction constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(
        self,
        grid_size: int = 10,
        initial_length: int = 3,
        target_score: int = 10,
        max_steps: int = 1000,
        render_mode: Optional[str] = None
    ):
        """
        Initialize two-snake competitive environment.

        Args:
            grid_size: Size of the grid (default 10x10)
            initial_length: Initial length of each snake
            target_score: Score needed to win (default 10)
            max_steps: Maximum steps per episode
            render_mode: Rendering mode
        """
        super().__init__()

        self.grid_size = grid_size
        self.initial_length = initial_length
        self.target_score = target_score
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 20-dimensional feature space per agent
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(20,),
            dtype=np.float32
        )

        # Each agent has 3 discrete actions
        self.action_space = spaces.Discrete(3)

        # Game state (initialized in reset)
        self.snake1_positions = []
        self.snake1_direction = self.UP
        self.snake1_alive = True
        self.score1 = 0

        self.snake2_positions = []
        self.snake2_direction = self.UP
        self.snake2_alive = True
        self.score2 = 0

        self.food_position = None
        self.steps = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to initial state.

        Returns:
            Tuple of (observations_dict, info)
            observations_dict contains 'agent1' and 'agent2' keys
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Initialize Snake 1 at top-left corner, heading right
        self.snake1_positions = [(1, 1 + i) for i in range(self.initial_length)]
        self.snake1_direction = self.RIGHT
        self.snake1_alive = True
        self.score1 = 0

        # Initialize Snake 2 at bottom-right corner, heading left
        start_x = self.grid_size - 2
        start_y = self.grid_size - 2
        self.snake2_positions = [(start_x, start_y - i) for i in range(self.initial_length)]
        self.snake2_direction = self.LEFT
        self.snake2_alive = True
        self.score2 = 0

        # Spawn initial food
        self._spawn_food()

        self.steps = 0

        # Get observations for both agents
        obs1 = self._get_observation(agent_id=1)
        obs2 = self._get_observation(agent_id=2)

        observations = {
            'agent1': obs1,
            'agent2': obs2
        }

        info = self._get_info()

        return observations, info

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one step with actions from both agents.

        Args:
            actions: Dict with 'agent1' and 'agent2' keys containing actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
            All return values are dicts with 'agent1' and 'agent2' keys
        """
        self.steps += 1

        action1 = actions.get('agent1', self.STRAIGHT)
        action2 = actions.get('agent2', self.STRAIGHT)

        # Initialize rewards
        reward1 = -0.01  # Time penalty
        reward2 = -0.01

        # Update directions
        if self.snake1_alive:
            self.snake1_direction = self._update_direction(self.snake1_direction, action1)
        if self.snake2_alive:
            self.snake2_direction = self._update_direction(self.snake2_direction, action2)

        # Get next head positions
        next_head1 = self._get_next_position(self.snake1_positions[0], self.snake1_direction) if self.snake1_alive else None
        next_head2 = self._get_next_position(self.snake2_positions[0], self.snake2_direction) if self.snake2_alive else None

        # Check collisions for both snakes
        collision1 = False
        collision2 = False

        if self.snake1_alive:
            collision1 = self._is_collision(next_head1, snake_positions=self.snake1_positions, opponent_positions=self.snake2_positions)

        if self.snake2_alive:
            collision2 = self._is_collision(next_head2, snake_positions=self.snake2_positions, opponent_positions=self.snake1_positions)

        # Handle head-on collision (both snakes hit each other's head)
        if (self.snake1_alive and self.snake2_alive and
            next_head1 is not None and next_head2 is not None and
            next_head1 == next_head2):
            collision1 = True
            collision2 = True

        # Process Snake 1
        if self.snake1_alive:
            if collision1:
                self.snake1_alive = False
                reward1 = -100.0
                # Check if snake2 still alive - if so, snake2 gets bonus
                if self.snake2_alive:
                    reward2 += 50.0
            else:
                # Move snake 1
                self.snake1_positions.insert(0, next_head1)

                # Check if ate food
                if next_head1 == self.food_position:
                    self.score1 += 1
                    reward1 += 10.0
                    self._spawn_food()
                    # Snake grows (don't remove tail)
                else:
                    # Maintain length
                    self.snake1_positions.pop()

        # Process Snake 2
        if self.snake2_alive:
            if collision2:
                self.snake2_alive = False
                reward2 = -100.0
                # Check if snake1 still alive - if so, snake1 gets bonus
                if self.snake1_alive:
                    reward1 += 50.0
            else:
                # Move snake 2
                self.snake2_positions.insert(0, next_head2)

                # Check if ate food
                if next_head2 == self.food_position:
                    self.score2 += 1
                    reward2 += 10.0
                    self._spawn_food()
                    # Snake grows (don't remove tail)
                else:
                    # Maintain length
                    self.snake2_positions.pop()

        # Check win conditions
        terminated1 = False
        terminated2 = False
        truncated1 = False
        truncated2 = False

        # Check if either snake reached target score
        if self.score1 >= self.target_score:
            reward1 += 100.0
            reward2 -= 50.0  # Penalty for losing
            terminated1 = True
            terminated2 = True
        elif self.score2 >= self.target_score:
            reward2 += 100.0
            reward1 -= 50.0  # Penalty for losing
            terminated1 = True
            terminated2 = True
        # Both snakes dead
        elif not self.snake1_alive and not self.snake2_alive:
            terminated1 = True
            terminated2 = True
        # Only one snake alive - continue until timeout or target reached
        # (already handled above)

        # Check timeout
        if self.steps >= self.max_steps:
            truncated1 = True
            truncated2 = True

        # Get observations
        obs1 = self._get_observation(agent_id=1)
        obs2 = self._get_observation(agent_id=2)

        observations = {
            'agent1': obs1,
            'agent2': obs2
        }

        rewards = {
            'agent1': reward1,
            'agent2': reward2
        }

        terminated = {
            'agent1': terminated1,
            'agent2': terminated2
        }

        truncated = {
            'agent1': truncated1,
            'agent2': truncated2
        }

        info = self._get_info()

        return observations, rewards, terminated, truncated, info

    def _update_direction(self, current_direction: int, action: int) -> int:
        """Update direction based on relative action."""
        if action == self.STRAIGHT:
            return current_direction
        elif action == self.RIGHT_TURN:
            return (current_direction + 1) % 4
        elif action == self.LEFT_TURN:
            return (current_direction - 1) % 4
        return current_direction

    def _get_next_position(self, head: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Get next head position given current head and direction."""
        x, y = head

        if direction == self.UP:
            return (x, y - 1)
        elif direction == self.RIGHT:
            return (x + 1, y)
        elif direction == self.DOWN:
            return (x, y + 1)
        elif direction == self.LEFT:
            return (x - 1, y)

        return head

    def _is_collision(
        self,
        position: Tuple[int, int],
        snake_positions: list,
        opponent_positions: list
    ) -> bool:
        """
        Check if position results in collision.

        Collision occurs if:
        - Hit wall
        - Hit own body
        - Hit opponent's body
        """
        x, y = position

        # Wall collision
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True

        # Own body collision (exclude head)
        if position in snake_positions[1:]:
            return True

        # Opponent body collision
        if position in opponent_positions:
            return True

        return False

    def _spawn_food(self) -> None:
        """Spawn food at random empty position."""
        empty_positions = []

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                # Check if position is empty (not occupied by either snake)
                if (pos not in self.snake1_positions and
                    pos not in self.snake2_positions):
                    empty_positions.append(pos)

        if empty_positions:
            # Use seeded RNG from gymnasium for reproducibility
            idx = self.np_random.integers(len(empty_positions)) if hasattr(self, 'np_random') and self.np_random is not None else np.random.randint(len(empty_positions))
            self.food_position = empty_positions[idx]
        else:
            # Grid full - no empty positions (edge case)
            self.food_position = None

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """
        Get observation for specified agent.

        20 features:
        - Danger (3): straight, right, left
        - Own direction (4): one-hot
        - Food position (2): normalized relative x, y
        - Own length normalized (1)
        - Opponent head distance (2): normalized relative x, y
        - Opponent direction (4): one-hot
        - Opponent length normalized (1)
        - Score difference (1): (own_score - opp_score) / target_score
        - Opponent body nearby (2): opponent body in front, opponent body adjacent
        """
        if agent_id == 1:
            snake_pos = self.snake1_positions
            snake_dir = self.snake1_direction
            snake_alive = self.snake1_alive
            own_score = self.score1
            opp_pos = self.snake2_positions
            opp_dir = self.snake2_direction
            opp_alive = self.snake2_alive
            opp_score = self.score2
        else:
            snake_pos = self.snake2_positions
            snake_dir = self.snake2_direction
            snake_alive = self.snake2_alive
            own_score = self.score2
            opp_pos = self.snake1_positions
            opp_dir = self.snake1_direction
            opp_alive = self.snake1_alive
            opp_score = self.score1

        # If agent is dead, return zero observation
        if not snake_alive:
            return np.zeros(20, dtype=np.float32)

        head = snake_pos[0]

        # 1-3: Danger detection (straight, right, left)
        danger_straight = int(self._is_collision(
            self._get_next_position(head, snake_dir),
            snake_pos,
            opp_pos if opp_alive else []
        ))

        right_dir = (snake_dir + 1) % 4
        danger_right = int(self._is_collision(
            self._get_next_position(head, right_dir),
            snake_pos,
            opp_pos if opp_alive else []
        ))

        left_dir = (snake_dir - 1) % 4
        danger_left = int(self._is_collision(
            self._get_next_position(head, left_dir),
            snake_pos,
            opp_pos if opp_alive else []
        ))

        # 4-7: Own direction (one-hot)
        dir_up = int(snake_dir == self.UP)
        dir_right = int(snake_dir == self.RIGHT)
        dir_down = int(snake_dir == self.DOWN)
        dir_left = int(snake_dir == self.LEFT)

        # 8-9: Food position (normalized relative distance)
        if self.food_position is not None:
            food_x = (self.food_position[0] - head[0]) / self.grid_size
            food_y = (self.food_position[1] - head[1]) / self.grid_size
        else:
            food_x = 0.0
            food_y = 0.0

        # 10: Own length (normalized)
        own_length = len(snake_pos) / (self.grid_size * self.grid_size)

        # 11-12: Opponent head distance (normalized)
        if opp_alive and len(opp_pos) > 0:
            opp_head = opp_pos[0]
            opp_head_x = (opp_head[0] - head[0]) / self.grid_size
            opp_head_y = (opp_head[1] - head[1]) / self.grid_size
        else:
            opp_head_x = 0.0
            opp_head_y = 0.0

        # 13-16: Opponent direction (one-hot)
        if opp_alive:
            opp_dir_up = int(opp_dir == self.UP)
            opp_dir_right = int(opp_dir == self.RIGHT)
            opp_dir_down = int(opp_dir == self.DOWN)
            opp_dir_left = int(opp_dir == self.LEFT)
        else:
            opp_dir_up = 0
            opp_dir_right = 0
            opp_dir_down = 0
            opp_dir_left = 0

        # 17: Opponent length (normalized)
        if opp_alive:
            opp_length = len(opp_pos) / (self.grid_size * self.grid_size)
        else:
            opp_length = 0.0

        # 18: Score difference (normalized)
        score_diff = (own_score - opp_score) / max(self.target_score, 1)

        # 19-20: Opponent body nearby detection
        opp_body_front = 0.0
        opp_body_adjacent = 0.0

        if opp_alive:
            # Check if opponent body is in front
            front_pos = self._get_next_position(head, snake_dir)
            if front_pos in opp_pos:
                opp_body_front = 1.0

            # Check if opponent body is adjacent (any of 8 directions)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    adj_pos = (head[0] + dx, head[1] + dy)
                    if adj_pos in opp_pos:
                        opp_body_adjacent = 1.0
                        break

        observation = np.array([
            danger_straight, danger_right, danger_left,
            dir_up, dir_right, dir_down, dir_left,
            food_x, food_y,
            own_length,
            opp_head_x, opp_head_y,
            opp_dir_up, opp_dir_right, opp_dir_down, opp_dir_left,
            opp_length,
            score_diff,
            opp_body_front, opp_body_adjacent
        ], dtype=np.float32)

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary with current game state."""
        return {
            'score1': self.score1,
            'score2': self.score2,
            'snake1_length': len(self.snake1_positions) if self.snake1_alive else 0,
            'snake2_length': len(self.snake2_positions) if self.snake2_alive else 0,
            'snake1_alive': self.snake1_alive,
            'snake2_alive': self.snake2_alive,
            'steps': self.steps,
            'winner': self._get_winner()
        }

    def _get_winner(self) -> Optional[int]:
        """
        Get winner of the episode.

        Returns:
            1 if snake1 won, 2 if snake2 won, 0 for draw, None if ongoing
        """
        if self.score1 >= self.target_score:
            return 1
        elif self.score2 >= self.target_score:
            return 2
        elif not self.snake1_alive and not self.snake2_alive:
            return 0  # Draw - both died simultaneously
        elif not self.snake1_alive and self.snake2_alive:
            # Snake2 wins immediately if snake1 dead
            return 2
        elif not self.snake2_alive and self.snake1_alive:
            # Snake1 wins immediately if snake2 dead
            return 1

        return None  # Ongoing

    def render(self):
        """Render the environment (optional - can be implemented later)."""
        if self.render_mode is None:
            return

        # Simple text rendering for now
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Draw snake 1
        for i, (x, y) in enumerate(self.snake1_positions):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y][x] = '1' if i == 0 else '#'

        # Draw snake 2
        for i, (x, y) in enumerate(self.snake2_positions):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y][x] = '2' if i == 0 else '@'

        # Draw food
        if self.food_position:
            fx, fy = self.food_position
            if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
                grid[fy][fx] = '*'

        # Print grid
        print('\n' + '='* (self.grid_size * 2 + 3))
        print(f"Score: Snake1={self.score1}, Snake2={self.score2} | Steps: {self.steps}")
        print('='* (self.grid_size * 2 + 3))
        for row in grid:
            print('|' + ' '.join(row) + '|')
        print('='* (self.grid_size * 2 + 3))

    def close(self):
        """Clean up resources."""
        pass
