"""
Vectorized Two-Snake Competitive Environment for GPU Acceleration

Competitive two-player snake game where two snakes compete for food.
Win condition: First to collect target_food items wins the round.
Both snakes run in parallel across N vectorized environments on GPU.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
import gymnasium as gym
from gymnasium import spaces
from core.state_representations_competitive import CompetitiveFeatureEncoder


class VectorizedTwoSnakeEnv:
    """
    Vectorized Two-Snake Competitive Environment

    Features:
    - N parallel competitive games on GPU
    - Two snakes per environment competing for single food
    - Win condition: First to target_food wins
    - Agent-centric observations (each snake sees self vs opponent)
    - Compatible with Independent Q-Learning and curriculum training
    """

    # Direction constants
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    DIRECTION_VECTORS = torch.tensor([[0, -1], [1, 0], [0, 1], [-1, 0]])  # (y, x) format

    # Action constants (relative)
    STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2

    def __init__(
        self,
        num_envs: int = 128,
        grid_size: int = None,  # DEPRECATED: Use grid_width and grid_height instead
        grid_width: int = None,
        grid_height: int = None,
        max_steps: int = 1000,
        target_food: int = 10,
        reward_food: float = 10.0,
        reward_opponent_food: float = 0.0,  # Changed from -5.0 (focus on own performance)
        reward_death: float = -50.0,
        reward_win: float = 100.0,
        reward_step: float = 0.0,  # Changed from 0.01 to 0.0 (focus on food)
        reward_stalemate: float = -10.0,
        device: torch.device = None
    ):
        """
        Initialize vectorized two-snake environment

        Args:
            num_envs: Number of parallel environments (reduced to 128 for 2x snakes)
            grid_size: DEPRECATED - Use grid_width and grid_height for square grids
            grid_width: Width of game grid (supports rectangular grids)
            grid_height: Height of game grid (supports rectangular grids)
            max_steps: Maximum steps before timeout/stalemate
            target_food: Food count needed to win round
            reward_food: Reward for collecting food
            reward_opponent_food: Penalty when opponent collects food
            reward_death: Penalty for dying
            reward_win: Bonus for winning round
            reward_step: Small reward per step alive
            reward_stalemate: Penalty if timeout without winner
            device: PyTorch device (GPU/CPU)
        """
        # Handle backward compatibility for grid dimensions
        if grid_width is None and grid_height is None:
            if grid_size is None:
                grid_size = 20  # Default square grid
            self.grid_width = grid_size
            self.grid_height = grid_size
        else:
            assert grid_width is not None and grid_height is not None, \
                "Both grid_width and grid_height must be specified"
            self.grid_width = grid_width
            self.grid_height = grid_height

        self.num_envs = num_envs
        self.max_steps = max_steps
        self.target_food = target_food

        # Rewards
        self.reward_food = reward_food
        self.reward_opponent_food = reward_opponent_food
        self.reward_death = reward_death
        self.reward_win = reward_win
        self.reward_step = reward_step
        self.reward_stalemate = reward_stalemate

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Move direction vectors to device
        self.DIRECTION_VECTORS = self.DIRECTION_VECTORS.to(self.device)

        # Action/observation spaces (for compatibility)
        self.action_space = spaces.Discrete(3)  # STRAIGHT, RIGHT, LEFT
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_envs, 35), dtype=np.float32
        )

        # Initialize game state tensors
        self._init_state_tensors()

        # Initialize competitive feature encoder
        # Note: use_flood_fill=False for speed (avoids CPU bottleneck)
        self.feature_encoder = CompetitiveFeatureEncoder(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            max_length=self.max_length,
            target_food=target_food,
            device=self.device,
            use_flood_fill=False
        )

        # Initialize game state by resetting all environments
        self.reset()

    @property
    def grid_diagonal(self) -> float:
        """Calculate true diagonal distance for normalization"""
        return (self.grid_width ** 2 + self.grid_height ** 2) ** 0.5

    @property
    def total_cells(self) -> int:
        """Total number of cells in grid"""
        return self.grid_width * self.grid_height

    @property
    def max_length(self) -> int:
        """Maximum possible snake length"""
        return self.total_cells

    def set_target_food(self, target_food: int):
        """Dynamically change target food for curriculum learning"""
        self.target_food = target_food
        self.feature_encoder.target_food = target_food

    def _init_state_tensors(self):
        """Initialize all state tensors on GPU"""
        max_length = self.max_length  # Use property

        # Snake 1 (Big network - 256x256)
        self.snakes1 = torch.zeros(
            (self.num_envs, max_length, 2), dtype=torch.long, device=self.device
        )
        self.lengths1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.directions1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.alive1 = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.food_counts1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Snake 2 (Small network - 128x128)
        self.snakes2 = torch.zeros(
            (self.num_envs, max_length, 2), dtype=torch.long, device=self.device
        )
        self.lengths2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.directions2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.alive2 = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.food_counts2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Shared game state
        self.foods = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.round_winners = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Winners: 0=in_progress, 1=snake1_wins, 2=snake2_wins, 3=both_lose

        # Steps since last food (for state features)
        self.steps_since_food1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.steps_since_food2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset all environments

        Returns:
            obs1: (num_envs, 35) observations for snake 1
            obs2: (num_envs, 35) observations for snake 2
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # VECTORIZED: Reset all environments at once
        # For rectangular grids, spawn snakes in left/right quarters
        center_x1 = self.grid_width // 4  # Left quarter for snake 1
        center_x2 = (self.grid_width * 3) // 4  # Right quarter for snake 2
        center_y = self.grid_height // 2  # Vertical center

        # Initialize snake 1 (left side, heading right) - ALL environments at once
        self.snakes1[:, 0] = torch.tensor([center_x1, center_y], device=self.device)
        self.snakes1[:, 1] = torch.tensor([center_x1 - 1, center_y], device=self.device)
        self.snakes1[:, 2] = torch.tensor([center_x1 - 2, center_y], device=self.device)

        # Initialize snake 2 (right side, heading left) - ALL environments at once
        self.snakes2[:, 0] = torch.tensor([center_x2, center_y], device=self.device)
        self.snakes2[:, 1] = torch.tensor([center_x2 + 1, center_y], device=self.device)
        self.snakes2[:, 2] = torch.tensor([center_x2 + 2, center_y], device=self.device)

        self.lengths1[:] = 3
        self.lengths2[:] = 3
        self.directions1[:] = self.RIGHT  # Snake 1 faces right
        self.directions2[:] = self.LEFT   # Snake 2 faces left
        self.alive1[:] = True
        self.alive2[:] = True
        self.food_counts1[:] = 0
        self.food_counts2[:] = 0
        self.steps[:] = 0
        self.round_winners[:] = 0
        self.steps_since_food1[:] = 0
        self.steps_since_food2[:] = 0

        # Spawn food
        self._spawn_food_all()

        # Get initial observations
        return self._get_observations()

    def _spawn_food_all(self):
        """VECTORIZED: Spawn food in all environments at random empty cells"""
        # Generate random positions for all environments (handle rectangular grids)
        food_x = torch.randint(0, self.grid_width, (self.num_envs,), device=self.device)
        food_y = torch.randint(0, self.grid_height, (self.num_envs,), device=self.device)
        candidate_foods = torch.stack([food_x, food_y], dim=1)

        # For simplicity, accept the random position (collision probability is low early in game)
        # A fully vectorized collision check would be too complex for marginal benefit
        self.foods = candidate_foods

    def _spawn_food_single(self, env_idx: int):
        """Spawn food in a single environment (used for per-env respawning)"""
        # Get occupied cells
        snake1 = self.snakes1[env_idx, :self.lengths1[env_idx]]
        snake2 = self.snakes2[env_idx, :self.lengths2[env_idx]]
        occupied = torch.cat([snake1, snake2], dim=0)

        # Find empty cells
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random position for rectangular grid
            food_x = torch.randint(0, self.grid_width, (1,), device=self.device).item()
            food_y = torch.randint(0, self.grid_height, (1,), device=self.device).item()
            food_pos = torch.tensor([food_x, food_y], device=self.device)

            # Check if position is empty
            if not ((occupied == food_pos).all(dim=1).any()):
                self.foods[env_idx] = food_pos
                return

        # Fallback: place food at grid center if no empty cell found
        self.foods[env_idx] = torch.tensor([self.grid_width // 2, self.grid_height // 2], device=self.device)

    def step(
        self, actions1: torch.Tensor, actions2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all environments with actions from both snakes

        Args:
            actions1: (num_envs,) actions for snake 1
            actions2: (num_envs,) actions for snake 2

        Returns:
            obs1: (num_envs, 35) observations for snake 1
            obs2: (num_envs, 35) observations for snake 2
            rewards1: (num_envs,) rewards for snake 1
            rewards2: (num_envs,) rewards for snake 2
            dones: (num_envs,) done flags (True when round ends)
            info: dict with episode statistics
        """
        # Initialize rewards
        rewards1 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        rewards2 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Convert actions to new directions
        new_directions1 = self._actions_to_directions(actions1, self.directions1)
        new_directions2 = self._actions_to_directions(actions2, self.directions2)

        # Update directions for alive snakes only
        self.directions1 = torch.where(self.alive1, new_directions1, self.directions1)
        self.directions2 = torch.where(self.alive2, new_directions2, self.directions2)

        # Move snakes (only if alive)
        self._move_snake(1)
        self._move_snake(2)

        # Check collisions (wall, self, opponent)
        collision1 = self._check_collisions(1)
        collision2 = self._check_collisions(2)

        # Mark dead snakes
        self.alive1 = self.alive1 & ~collision1
        self.alive2 = self.alive2 & ~collision2

        # Death penalties
        rewards1 = torch.where(collision1, torch.full_like(rewards1, self.reward_death), rewards1)
        rewards2 = torch.where(collision2, torch.full_like(rewards2, self.reward_death), rewards2)

        # Check food collection (only for alive snakes)
        food_collected1 = self._check_food_collection(1)
        food_collected2 = self._check_food_collection(2)

        # Food rewards
        rewards1 = torch.where(
            food_collected1,
            rewards1 + self.reward_food,
            rewards1
        )
        rewards1 = torch.where(
            food_collected2,
            rewards1 + self.reward_opponent_food,
            rewards1
        )

        rewards2 = torch.where(
            food_collected2,
            rewards2 + self.reward_food,
            rewards2
        )
        rewards2 = torch.where(
            food_collected1,
            rewards2 + self.reward_opponent_food,
            rewards2
        )

        # Step alive reward
        rewards1 = torch.where(self.alive1, rewards1 + self.reward_step, rewards1)
        rewards2 = torch.where(self.alive2, rewards2 + self.reward_step, rewards2)

        # Update step counter
        self.steps += 1
        self.total_steps += self.num_envs

        # Update steps since last food (reset to 0 in _check_food_collection when food is eaten)
        self.steps_since_food1 += 1
        self.steps_since_food2 += 1

        # Check win conditions
        self._check_win_conditions()

        # Apply win/stalemate rewards
        win_mask1 = self.round_winners == 1
        win_mask2 = self.round_winners == 2
        stalemate_mask = self.round_winners == 3

        rewards1 = torch.where(win_mask1, rewards1 + self.reward_win, rewards1)
        rewards2 = torch.where(win_mask2, rewards2 + self.reward_win, rewards2)
        rewards1 = torch.where(stalemate_mask, rewards1 + self.reward_stalemate, rewards1)
        rewards2 = torch.where(stalemate_mask, rewards2 + self.reward_stalemate, rewards2)

        # Check if rounds are done
        dones = self.round_winners > 0

        # Prepare info dict
        info = self._get_info(dones)

        # Auto-reset finished environments
        if dones.any():
            self._reset_done_envs(dones)

        # Get observations
        obs1, obs2 = self._get_observations()

        return obs1, obs2, rewards1, rewards2, dones, info

    def _actions_to_directions(self, actions: torch.Tensor, current_directions: torch.Tensor) -> torch.Tensor:
        """Convert relative actions to absolute directions"""
        # STRAIGHT = 0, TURN_RIGHT = 1, TURN_LEFT = 2
        # Directions: UP=0, RIGHT=1, DOWN=2, LEFT=3

        new_directions = current_directions.clone()

        # Turn right: (direction + 1) % 4
        turn_right_mask = actions == self.TURN_RIGHT
        new_directions = torch.where(
            turn_right_mask,
            (current_directions + 1) % 4,
            new_directions
        )

        # Turn left: (direction - 1) % 4
        turn_left_mask = actions == self.TURN_LEFT
        new_directions = torch.where(
            turn_left_mask,
            (current_directions - 1) % 4,
            new_directions
        )

        return new_directions

    def _move_snake(self, snake_id: int):
        """Move snake forward in current direction"""
        if snake_id == 1:
            snakes = self.snakes1
            lengths = self.lengths1
            directions = self.directions1
            alive = self.alive1
        else:
            snakes = self.snakes2
            lengths = self.lengths2
            directions = self.directions2
            alive = self.alive2

        # Get head positions
        heads = snakes[:, 0, :]  # (num_envs, 2)

        # Calculate new head positions
        direction_vecs = self.DIRECTION_VECTORS[directions]  # (num_envs, 2)
        new_heads = heads + direction_vecs

        # VECTORIZED: Shift snake bodies (all environments at once)
        # Shift all body segments back by 1 position
        snakes[:, 1:] = snakes[:, :-1].clone()
        # Place new heads
        snakes[:, 0] = new_heads

        # Mask invalid segments for dead snakes and segments beyond length
        segment_idx = torch.arange(self.max_length, device=self.device).unsqueeze(0)  # (1, max_length)
        valid_mask = (segment_idx < lengths.unsqueeze(1)) & alive.unsqueeze(1)  # (num_envs, max_length)
        snakes[~valid_mask] = -1  # Mark invalid positions

    def _check_collisions(self, snake_id: int) -> torch.Tensor:
        """VECTORIZED: Check collisions for snake (wall, self, opponent)"""
        if snake_id == 1:
            snakes = self.snakes1
            lengths = self.lengths1
            alive = self.alive1
            opponent_snakes = self.snakes2
            opponent_lengths = self.lengths2
        else:
            snakes = self.snakes2
            lengths = self.lengths2
            alive = self.alive2
            opponent_snakes = self.snakes1
            opponent_lengths = self.lengths1

        # Get all heads at once (num_envs, 2)
        heads = snakes[:, 0, :]

        # Wall collisions - pure tensor operations
        wall_collision = (
            (heads[:, 0] < 0) |
            (heads[:, 0] >= self.grid_width) |  # X bound
            (heads[:, 1] < 0) |
            (heads[:, 1] >= self.grid_height)   # Y bound
        )

        # Self collisions - broadcast comparison
        # heads: (num_envs, 2) -> (num_envs, 1, 2)
        # body: (num_envs, max_length-1, 2)
        heads_expanded = heads.unsqueeze(1)
        body = snakes[:, 1:, :]

        # Check which segments match the head
        matches = (heads_expanded == body).all(dim=-1)  # (num_envs, max_length-1)

        # Mask by valid segments (segments 1 onwards that are within length)
        segment_idx = torch.arange(1, self.max_length, device=self.device).unsqueeze(0)  # (1, max_length-1)
        valid_segments = segment_idx < lengths.unsqueeze(1)  # (num_envs, max_length-1)

        self_collision = (matches & valid_segments).any(dim=1)

        # Opponent collisions - broadcast comparison
        # Check head against all opponent segments
        opponent_matches = (heads_expanded == opponent_snakes).all(dim=-1)  # (num_envs, max_length)

        # Mask by valid opponent segments
        opponent_segment_idx = torch.arange(self.max_length, device=self.device).unsqueeze(0)
        valid_opponent = opponent_segment_idx < opponent_lengths.unsqueeze(1)

        opponent_collision = (opponent_matches & valid_opponent).any(dim=1)

        # Combine all collision types, but only for alive snakes
        collision = (wall_collision | self_collision | opponent_collision) & alive

        return collision

    def _check_food_collection(self, snake_id: int) -> torch.Tensor:
        """VECTORIZED: Check if snake collected food"""
        if snake_id == 1:
            snakes = self.snakes1
            lengths = self.lengths1
            alive = self.alive1
            food_counts = self.food_counts1
            steps_since_food = self.steps_since_food1
        else:
            snakes = self.snakes2
            lengths = self.lengths2
            alive = self.alive2
            food_counts = self.food_counts2
            steps_since_food = self.steps_since_food2

        # Get all heads at once (num_envs, 2)
        heads = snakes[:, 0, :]

        # Check food collection (vectorized)
        collected = (heads == self.foods).all(dim=1) & alive  # (num_envs,)

        # Update lengths, food counts, and steps_since_food where food was collected
        lengths[collected] += 1
        food_counts[collected] += 1
        steps_since_food[collected] = 0

        # Respawn food for envs that collected (will be vectorized in Phase 1.4)
        if collected.any():
            env_indices = collected.nonzero(as_tuple=True)[0]
            for env_idx in env_indices:
                self._spawn_food_single(env_idx.item())

        return collected

    def _check_win_conditions(self):
        """Check win conditions for all environments"""
        # Win by reaching target food
        wins_by_food1 = self.food_counts1 >= self.target_food
        wins_by_food2 = self.food_counts2 >= self.target_food

        # Win by being last alive
        wins_by_survival1 = self.alive1 & ~self.alive2
        wins_by_survival2 = self.alive2 & ~self.alive1

        # Stalemate (timeout or both dead)
        both_dead = ~self.alive1 & ~self.alive2
        timeout = self.steps >= self.max_steps
        stalemate = both_dead | timeout

        # Set winners (priority: food > survival > stalemate)
        self.round_winners = torch.where(wins_by_food1, torch.ones_like(self.round_winners), self.round_winners)
        self.round_winners = torch.where(wins_by_food2, torch.full_like(self.round_winners, 2), self.round_winners)
        self.round_winners = torch.where(wins_by_survival1 & (self.round_winners == 0), torch.ones_like(self.round_winners), self.round_winners)
        self.round_winners = torch.where(wins_by_survival2 & (self.round_winners == 0), torch.full_like(self.round_winners, 2), self.round_winners)
        self.round_winners = torch.where(stalemate & (self.round_winners == 0), torch.full_like(self.round_winners, 3), self.round_winners)

    def _get_observations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get observations for both snakes using competitive feature encoder"""
        # Encode observations for Snake 1 (agent-centric)
        obs1 = self.feature_encoder.encode_batch(
            snake_self=self.snakes1,
            length_self=self.lengths1,
            direction_self=self.directions1,
            food_count_self=self.food_counts1,
            steps_since_food_self=self.steps_since_food1,
            snake_opponent=self.snakes2,
            length_opponent=self.lengths2,
            direction_opponent=self.directions2,
            food_count_opponent=self.food_counts2,
            food=self.foods
        )

        # Encode observations for Snake 2 (agent-centric, swapped roles)
        obs2 = self.feature_encoder.encode_batch(
            snake_self=self.snakes2,
            length_self=self.lengths2,
            direction_self=self.directions2,
            food_count_self=self.food_counts2,
            steps_since_food_self=self.steps_since_food2,
            snake_opponent=self.snakes1,
            length_opponent=self.lengths1,
            direction_opponent=self.directions1,
            food_count_opponent=self.food_counts1,
            food=self.foods
        )

        return obs1, obs2

    def _get_info(self, dones: torch.Tensor) -> Dict:
        """Get episode info for finished environments"""
        info = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
        }

        if dones.any():
            done_indices = dones.nonzero(as_tuple=True)[0]
            info['done_envs'] = done_indices.cpu().numpy()
            info['winners'] = self.round_winners[done_indices].cpu().numpy()
            info['food_counts1'] = self.food_counts1[done_indices].cpu().numpy()
            info['food_counts2'] = self.food_counts2[done_indices].cpu().numpy()
            info['steps'] = self.steps[done_indices].cpu().numpy()

        return info

    def _reset_done_envs(self, dones: torch.Tensor):
        """Reset environments that finished"""
        done_indices = dones.nonzero(as_tuple=True)[0]

        for env_idx in done_indices:
            env_idx = env_idx.item()
            # For rectangular grids, spawn snakes in left/right quarters
            center_x1 = self.grid_width // 4  # Left quarter for snake 1
            center_x2 = (self.grid_width * 3) // 4  # Right quarter for snake 2
            center_y = self.grid_height // 2  # Vertical center

            # Reset snake 1
            self.snakes1[env_idx, 0] = torch.tensor([center_x1, center_y], device=self.device)
            self.snakes1[env_idx, 1] = torch.tensor([center_x1 - 1, center_y], device=self.device)
            self.snakes1[env_idx, 2] = torch.tensor([center_x1 - 2, center_y], device=self.device)
            self.lengths1[env_idx] = 3
            self.directions1[env_idx] = self.RIGHT
            self.alive1[env_idx] = True
            self.food_counts1[env_idx] = 0

            # Reset snake 2
            self.snakes2[env_idx, 0] = torch.tensor([center_x2, center_y], device=self.device)
            self.snakes2[env_idx, 1] = torch.tensor([center_x2 + 1, center_y], device=self.device)
            self.snakes2[env_idx, 2] = torch.tensor([center_x2 + 2, center_y], device=self.device)
            self.lengths2[env_idx] = 3
            self.directions2[env_idx] = self.LEFT
            self.alive2[env_idx] = True
            self.food_counts2[env_idx] = 0

            # Reset game state
            self.steps[env_idx] = 0
            self.round_winners[env_idx] = 0

            # Respawn food
            self._spawn_food_single(env_idx)

        self.episode_count += len(done_indices)

    def render(self, env_idx: int = 0):
        """Render single environment (placeholder for visualizer)"""
        print(f"Environment {env_idx}:")
        print(f"  Snake 1: Length={self.lengths1[env_idx].item()}, Food={self.food_counts1[env_idx].item()}, Alive={self.alive1[env_idx].item()}")
        print(f"  Snake 2: Length={self.lengths2[env_idx].item()}, Food={self.food_counts2[env_idx].item()}, Alive={self.alive2[env_idx].item()}")
        print(f"  Steps: {self.steps[env_idx].item()}, Winner: {self.round_winners[env_idx].item()}")

    def close(self):
        """Clean up resources"""
        pass
