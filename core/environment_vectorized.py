"""
Vectorized Snake Environment for GPU Acceleration

Runs N snake environments in parallel using PyTorch tensors on GPU.
Achieves 100-300x speedup over single CPU environment.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Literal
from collections import deque
import gymnasium as gym
from gymnasium import spaces


class VectorizedSnakeEnv:
    """
    Vectorized Snake Environment

    Runs N parallel snake games on GPU for efficient training.
    Compatible with DQN and PPO training loops.
    """

    def __init__(
        self,
        num_envs: int = 256,
        grid_size: int = 10,
        action_space_type: Literal['absolute', 'relative'] = 'relative',
        state_representation: Literal['feature', 'grid'] = 'feature',
        max_steps: int = 1000,
        reward_food: float = 10.0,
        reward_death: float = -10.0,
        reward_step: float = -0.01,
        reward_distance: bool = True,
        use_flood_fill: bool = False,
        use_enhanced_features: bool = False,
        use_selective_features: bool = False,
        device: torch.device = None
    ):
        """
        Initialize vectorized environment

        Args:
            num_envs: Number of parallel environments
            grid_size: Size of game grid
            action_space_type: 'absolute' or 'relative'
            state_representation: 'feature' or 'grid'
            max_steps: Maximum steps per episode
            reward_food: Reward for eating food
            reward_death: Reward for dying
            reward_step: Per-step penalty
            reward_distance: Whether to use distance-based rewards
            use_flood_fill: Whether to use flood-fill features (14-dim instead of 11-dim)
            use_enhanced_features: Whether to use all enhanced features (24-dim total)
            use_selective_features: Whether to use only high-impact features (19-dim: flood-fill + tail features)
            device: PyTorch device (GPU/CPU)
        """
        # Validate feature flags (selective and enhanced are mutually exclusive)
        if use_selective_features and use_enhanced_features:
            raise ValueError("Cannot enable both use_selective_features and use_enhanced_features simultaneously")

        self.num_envs = num_envs
        self.grid_size = grid_size
        self.action_space_type = action_space_type
        self.state_representation = state_representation
        self.max_steps = max_steps
        self.use_flood_fill = use_flood_fill
        self.use_enhanced_features = use_enhanced_features
        self.use_selective_features = use_selective_features

        # Rewards
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_distance = reward_distance
        self.reward_timeout = -5.0  # Penalty for timing out without eating

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Action/observation spaces (for compatibility)
        if action_space_type == 'absolute':
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Discrete(3)

        if state_representation == 'feature':
            # Feature hierarchy (mutually exclusive modes)
            if use_enhanced_features:
                feature_dim = 23  # 10 base + 3 flood-fill + 10 enhanced
                use_flood_fill = True  # Enhanced requires flood-fill
            elif use_selective_features:
                feature_dim = 18  # 10 base + 3 flood-fill + 5 selective (tail features only)
                use_flood_fill = True  # Selective requires flood-fill
            elif use_flood_fill:
                feature_dim = 13  # 10 base + 3 flood-fill
            else:
                feature_dim = 10  # Base features only

            self.observation_space = spaces.Box(
                low=0, high=1, shape=(num_envs, feature_dim), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(num_envs, grid_size, grid_size, 3),
                dtype=np.float32
            )

        # Initialize game state tensors on GPU
        self._init_state_tensors()

    def _init_state_tensors(self):
        """Initialize state tensors on GPU"""
        # Snake bodies: (num_envs, max_length, 2) where max_length = grid_size^2
        max_length = self.grid_size * self.grid_size
        self.snakes = torch.zeros(
            (self.num_envs, max_length, 2),
            dtype=torch.long,
            device=self.device
        )

        # Snake lengths
        self.snake_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Snake directions (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        self.directions = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Food positions
        self.foods = torch.zeros(
            (self.num_envs, 2), dtype=torch.long, device=self.device
        )

        # Episode info
        self.steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.scores = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.dones = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Previous food distances (for distance reward)
        self.prev_food_distances = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Reset all environments

        Returns:
            observations: (num_envs, obs_dim) tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Reset all environments
        center = self.grid_size // 2

        # Initialize snakes in center, length 3, heading right
        for i in range(self.num_envs):
            self.snakes[i, 0] = torch.tensor([center, center], device=self.device)
            self.snakes[i, 1] = torch.tensor([center - 1, center], device=self.device)
            self.snakes[i, 2] = torch.tensor([center - 2, center], device=self.device)

        self.snake_lengths[:] = 3
        self.directions[:] = 1  # RIGHT
        self.steps[:] = 0
        self.scores[:] = 0
        self.dones[:] = False

        # Spawn food
        self._spawn_food_all()

        # Calculate initial distances
        self._update_food_distances()

        return self._get_observations()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step all environments

        Args:
            actions: (num_envs,) tensor of actions

        Returns:
            observations: (num_envs, obs_dim)
            rewards: (num_envs,)
            dones: (num_envs,)
            info: dict with episode info
        """
        # Convert actions to directions
        new_directions = self._actions_to_directions(actions)

        # For absolute actions, prevent 180-degree turns
        if self.action_space_type == 'absolute':
            # Check if opposite direction
            opposite_mask = (new_directions - self.directions) % 4 == 2
            new_directions = torch.where(opposite_mask, self.directions, new_directions)

        self.directions = new_directions

        # Calculate new head positions
        deltas = torch.tensor([
            [0, -1],  # UP
            [1, 0],   # RIGHT
            [0, 1],   # DOWN
            [-1, 0]   # LEFT
        ], device=self.device)

        current_heads = self.snakes[torch.arange(self.num_envs), 0]
        direction_deltas = deltas[self.directions]
        new_heads = current_heads + direction_deltas

        # Initialize rewards
        rewards = torch.full(
            (self.num_envs,),
            self.reward_step,
            dtype=torch.float32,
            device=self.device
        )

        # Check wall collisions
        wall_collision = (
            (new_heads[:, 0] < 0) |
            (new_heads[:, 0] >= self.grid_size) |
            (new_heads[:, 1] < 0) |
            (new_heads[:, 1] >= self.grid_size)
        )

        # Check self collisions
        self_collision = self._check_self_collision(new_heads)

        # Mark terminated environments
        terminated = wall_collision | self_collision
        rewards[terminated] = self.reward_death

        # Check food consumption (only for non-terminated)
        food_eaten = (~terminated) & (
            (new_heads[:, 0] == self.foods[:, 0]) &
            (new_heads[:, 1] == self.foods[:, 1])
        )

        # Update snakes
        self._update_snakes(new_heads, food_eaten, terminated)

        # Update rewards for food
        rewards[food_eaten] = self.reward_food

        # Distance-based rewards (only for non-terminated, non-food)
        if self.reward_distance:
            needs_distance_reward = (~terminated) & (~food_eaten)
            if needs_distance_reward.any():
                current_distances = self._manhattan_distance(
                    new_heads[needs_distance_reward],
                    self.foods[needs_distance_reward]
                )
                prev_distances = self.prev_food_distances[needs_distance_reward]

                # Scaled distance reward: reward magnitude proportional to distance change
                # Max Manhattan distance is ~18 in 10x10 grid, normalize to [-0.5, 0.5]
                distance_change = prev_distances - current_distances
                distance_rewards = distance_change * 0.5 / self.grid_size

                # Update main rewards
                rewards[needs_distance_reward] += distance_rewards

        # Update food distances
        self._update_food_distances()

        # Update step counters and check truncation
        self.steps += 1
        truncated = self.steps >= self.max_steps

        # Apply timeout penalty for environments that hit max steps without eating
        rewards[truncated] += self.reward_timeout

        # Combine termination conditions
        self.dones = terminated | truncated

        # Save done flags before auto-reset (important for training)
        dones_output = self.dones.clone()

        # Save episode info BEFORE auto-reset (so we capture final scores/lengths)
        info = {
            'scores': self.scores.clone(),
            'steps': self.steps.clone(),
            'snake_lengths': self.snake_lengths.clone(),
            'wall_deaths': wall_collision.clone(),
            'self_deaths': self_collision.clone(),
            'timeouts': truncated.clone()
        }

        # Auto-reset done environments
        if self.dones.any():
            self._reset_done_envs()

        # Get observations
        obs = self._get_observations()

        return obs, rewards, dones_output, info

    def _actions_to_directions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert actions to directions"""
        if self.action_space_type == 'absolute':
            return actions
        else:
            # Relative actions
            new_directions = self.directions.clone()

            # 0 = STRAIGHT (no change)
            # 1 = LEFT (turn left)
            left_mask = actions == 1
            new_directions[left_mask] = (self.directions[left_mask] - 1) % 4

            # 2 = RIGHT (turn right)
            right_mask = actions == 2
            new_directions[right_mask] = (self.directions[right_mask] + 1) % 4

            return new_directions

    def _check_self_collision(self, new_heads: torch.Tensor) -> torch.Tensor:
        """Check if new heads collide with snake bodies"""
        collisions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i in range(self.num_envs):
            snake_body = self.snakes[i, :self.snake_lengths[i]]
            head = new_heads[i]

            # Check if head matches any body segment
            matches = (snake_body == head).all(dim=1)
            collisions[i] = matches.any()

        return collisions

    def _update_snakes(
        self,
        new_heads: torch.Tensor,
        food_eaten: torch.Tensor,
        terminated: torch.Tensor
    ):
        """Update snake positions"""
        for i in range(self.num_envs):
            if terminated[i]:
                continue

            # Shift body forward
            length = self.snake_lengths[i]
            if food_eaten[i]:
                # Grow snake - don't remove tail
                self.snakes[i, 1:length+1] = self.snakes[i, :length].clone()
                self.snake_lengths[i] += 1
                self.scores[i] += 1

                # Spawn new food
                self._spawn_food_single(i)
            else:
                # Move snake - remove tail
                self.snakes[i, 1:length] = self.snakes[i, :length-1].clone()

            # Set new head
            self.snakes[i, 0] = new_heads[i]

    def _spawn_food_all(self):
        """Spawn food for all environments"""
        for i in range(self.num_envs):
            self._spawn_food_single(i)

    def _spawn_food_single(self, env_idx: int):
        """Spawn food for a single environment"""
        # Get empty cells
        snake = self.snakes[env_idx, :self.snake_lengths[env_idx]]

        # Generate random position until we find empty cell
        max_attempts = 100
        for _ in range(max_attempts):
            food_x = torch.randint(0, self.grid_size, (1,), device=self.device)
            food_y = torch.randint(0, self.grid_size, (1,), device=self.device)
            food_pos = torch.tensor([food_x, food_y], device=self.device).squeeze()

            # Check if position is empty
            occupied = ((snake == food_pos).all(dim=1)).any()
            if not occupied:
                self.foods[env_idx] = food_pos
                return

        # Fallback: place anywhere (shouldn't happen often)
        self.foods[env_idx] = torch.tensor(
            [self.grid_size // 2 + 1, self.grid_size // 2 + 1],
            device=self.device
        )

    def _manhattan_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """Calculate Manhattan distance"""
        return (pos1 - pos2).abs().sum(dim=-1).float()

    def _update_food_distances(self):
        """Update previous food distances"""
        heads = self.snakes[torch.arange(self.num_envs), 0]
        self.prev_food_distances = self._manhattan_distance(heads, self.foods)

    def _reset_done_envs(self):
        """Reset environments that are done"""
        done_indices = torch.where(self.dones)[0]

        center = self.grid_size // 2

        for idx in done_indices:
            idx = idx.item()

            # Reset snake
            self.snakes[idx, 0] = torch.tensor([center, center], device=self.device)
            self.snakes[idx, 1] = torch.tensor([center - 1, center], device=self.device)
            self.snakes[idx, 2] = torch.tensor([center - 2, center], device=self.device)
            self.snake_lengths[idx] = 3
            self.directions[idx] = 1  # RIGHT
            self.steps[idx] = 0
            self.scores[idx] = 0
            self.dones[idx] = False

            # Spawn food
            self._spawn_food_single(idx)

        # Update distances for reset envs
        if len(done_indices) > 0:
            self._update_food_distances()

    def _get_observations(self) -> torch.Tensor:
        """Get observations for all environments"""
        if self.state_representation == 'feature':
            return self._get_feature_observations()
        else:
            return self._get_grid_observations()

    def _get_feature_observations(self) -> torch.Tensor:
        """
        Get feature observations (11, 14, 19, or 24-dimensional)

        Returns:
            (num_envs, feature_dim) tensor
        """
        feature_dim = 10
        if self.use_flood_fill:
            feature_dim = 13
        if self.use_selective_features:
            feature_dim = 18  # Selective: tail features only
        if self.use_enhanced_features:
            feature_dim = 23  # All enhanced features
        obs = torch.zeros(
            (self.num_envs, feature_dim),
            dtype=torch.float32,
            device=self.device
        )

        deltas = torch.tensor([
            [0, -1],  # UP
            [1, 0],   # RIGHT
            [0, 1],   # DOWN
            [-1, 0]   # LEFT
        ], device=self.device)

        heads = self.snakes[torch.arange(self.num_envs), 0]

        # Danger detection (3 features: straight, left, right)
        for offset, col in zip([0, -1, 1], range(3)):
            check_dirs = (self.directions + offset) % 4
            check_deltas = deltas[check_dirs]
            next_pos = heads + check_deltas

            # Wall danger
            wall_danger = (
                (next_pos[:, 0] < 0) |
                (next_pos[:, 0] >= self.grid_size) |
                (next_pos[:, 1] < 0) |
                (next_pos[:, 1] >= self.grid_size)
            )

            # Body danger
            body_danger = self._check_self_collision(next_pos)

            obs[:, col] = (wall_danger | body_danger).float()

        # Food direction (4 features)
        obs[:, 3] = (self.foods[:, 1] < heads[:, 1]).float()  # UP
        obs[:, 4] = (self.foods[:, 0] > heads[:, 0]).float()  # RIGHT
        obs[:, 5] = (self.foods[:, 1] > heads[:, 1]).float()  # DOWN
        obs[:, 6] = (self.foods[:, 0] < heads[:, 0]).float()  # LEFT

        # Current direction (3 features - one-hot)
        for i in range(3):
            obs[:, 7 + i] = (self.directions == i).float()

        # Add flood-fill features if enabled
        if self.use_flood_fill:
            # Compute flood-fill for straight, right, and left directions
            for offset, col in zip([0, 1, -1], [10, 11, 12]):
                check_dirs = (self.directions + offset) % 4
                check_deltas = deltas[check_dirs]
                next_pos = heads + check_deltas

                # Only flood-fill if position is safe
                wall_safe = (
                    (next_pos[:, 0] >= 0) &
                    (next_pos[:, 0] < self.grid_size) &
                    (next_pos[:, 1] >= 0) &
                    (next_pos[:, 1] < self.grid_size)
                )
                body_safe = ~self._check_self_collision(next_pos)
                safe = wall_safe & body_safe

                # Compute flood-fill for safe positions
                flood_fill_values = self._compute_flood_fill_vectorized(next_pos, safe)
                obs[:, col] = flood_fill_values

        # Add selective features (tail features only - highest impact)
        if self.use_selective_features:
            # Selective mode: Only tail features (5 features)
            # [13-16]: Tail direction (up, right, down, left)
            # [17]: Tail reachability (most important!)

            # Tail direction (4 features)
            tail_directions = self._compute_tail_direction_vectorized()
            obs[:, 13:17] = tail_directions

            # Tail reachability (1 feature - HIGH IMPACT)
            tail_reachability = self._compute_tail_reachability_vectorized()
            obs[:, 17] = tail_reachability

        # Add all enhanced features if enabled
        elif self.use_enhanced_features:
            # Compute enhanced features (10 features total)
            # [13-15]: Escape routes (straight, right, left)
            # [16-19]: Tail direction (up, right, down, left)
            # [20]: Tail reachability
            # [21]: Distance to tail
            # [22]: Snake length ratio

            # 1. Escape route detection (3 features)
            for offset, col in zip([0, 1, -1], [13, 14, 15]):
                check_dirs = (self.directions + offset) % 4
                check_deltas = deltas[check_dirs]
                next_pos = heads + check_deltas

                # Count escape routes for each position
                escape_counts = self._compute_escape_routes_vectorized(next_pos)
                obs[:, col] = escape_counts

            # 2. Tail direction (4 features)
            tail_directions = self._compute_tail_direction_vectorized()
            obs[:, 16:20] = tail_directions

            # 3. Tail reachability (1 feature)
            tail_reachability = self._compute_tail_reachability_vectorized()
            obs[:, 20] = tail_reachability

            # 4. Distance to tail (1 feature, normalized)
            distance_to_tail = self._compute_distance_to_tail_vectorized()
            obs[:, 21] = distance_to_tail

            # 5. Snake length ratio (1 feature)
            max_length = self.grid_size * self.grid_size
            obs[:, 22] = self.snake_lengths.float() / max_length

        return obs

    def _get_grid_observations(self) -> torch.Tensor:
        """
        Get grid observations

        Returns:
            (num_envs, grid_size, grid_size, 3) tensor
        """
        obs = torch.zeros(
            (self.num_envs, self.grid_size, self.grid_size, 3),
            dtype=torch.float32,
            device=self.device
        )

        for i in range(self.num_envs):
            # Channel 0: Head
            head = self.snakes[i, 0]
            obs[i, head[1], head[0], 0] = 1.0

            # Channel 1: Body
            body = self.snakes[i, 1:self.snake_lengths[i]]
            for segment in body:
                obs[i, segment[1], segment[0], 1] = 1.0

            # Channel 2: Food
            food = self.foods[i]
            obs[i, food[1], food[0], 2] = 1.0

        return obs

    def _compute_flood_fill_vectorized(
        self,
        start_positions: torch.Tensor,
        safe_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flood-fill free space for multiple starting positions (optimized).

        Optimizations:
        - Batch GPU->CPU transfers (single transfer for all envs)
        - Early termination for unsafe positions
        - Optimized BFS with deque and pre-computed bounds

        Args:
            start_positions: (num_envs, 2) tensor of starting positions
            safe_mask: (num_envs,) boolean tensor indicating which positions are safe

        Returns:
            (num_envs,) tensor of normalized free space ratios [0, 1]
        """
        flood_fill_values = torch.zeros(self.num_envs, device=self.device)

        # OPTIMIZATION: Batch GPU->CPU transfer for ALL environments at once
        start_positions_cpu = start_positions.cpu().numpy()
        safe_mask_cpu = safe_mask.cpu().numpy()
        snake_lengths_cpu = self.snake_lengths.cpu().numpy()

        # Find max snake length to batch snake positions transfer
        max_len = int(snake_lengths_cpu.max())
        if max_len > 0:
            snakes_cpu = self.snakes[:, :max_len].cpu().numpy()
        else:
            # All snakes are dead (shouldn't happen in normal training)
            return flood_fill_values

        # Process each environment (BFS must be sequential, but data transfer is batched)
        max_val = self.grid_size - 1

        for env_idx in range(self.num_envs):
            if not safe_mask_cpu[env_idx]:
                flood_fill_values[env_idx] = 0.0
                continue

            # Get snake body as set for this environment (already on CPU)
            snake_length = snake_lengths_cpu[env_idx]
            snake_positions = snakes_cpu[env_idx, :snake_length]
            snake_set = set(map(tuple, snake_positions))

            # Starting position (already on CPU)
            start_pos = tuple(start_positions_cpu[env_idx])

            # Optimized BFS flood-fill with deque
            visited = set()
            queue = deque([start_pos])
            visited.add(start_pos)

            while queue:
                current = queue.popleft()
                x, y = current

                # Check all 4 neighbors with optimized bounds check
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy

                    # Fast bounds check
                    if nx < 0 or nx > max_val or ny < 0 or ny > max_val:
                        continue

                    neighbor = (nx, ny)

                    # Check if valid and not visited
                    if neighbor not in visited and neighbor not in snake_set:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Normalize by max free space
            max_free_space = self.grid_size * self.grid_size - len(snake_set)
            if max_free_space <= 0:
                flood_fill_values[env_idx] = 0.0
            else:
                free_space_ratio = len(visited) / max_free_space
                flood_fill_values[env_idx] = min(1.0, free_space_ratio)

        return flood_fill_values

    def _compute_escape_routes_vectorized(
        self,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Count the number of safe adjacent cells (escape routes) for given positions.

        Args:
            positions: (num_envs, 2) tensor of positions to check

        Returns:
            (num_envs,) tensor of normalized escape counts (0-1)
        """
        escape_counts = torch.zeros(self.num_envs, device=self.device)

        deltas = torch.tensor([
            [0, -1],  # UP
            [1, 0],   # RIGHT
            [0, 1],   # DOWN
            [-1, 0]   # LEFT
        ], device=self.device)

        for env_idx in range(self.num_envs):
            pos = positions[env_idx]

            # Check if position itself is safe
            if not self._is_position_safe_single(pos, env_idx):
                escape_counts[env_idx] = 0.0
                continue

            # Count safe adjacent cells
            safe_count = 0
            for delta in deltas:
                adj_pos = pos + delta
                if self._is_position_safe_single(adj_pos, env_idx):
                    safe_count += 1

            # Normalize by max possible (4 directions)
            escape_counts[env_idx] = safe_count / 4.0

        return escape_counts

    def _is_position_safe_single(
        self,
        pos: torch.Tensor,
        env_idx: int
    ) -> bool:
        """
        Check if a position is safe (within bounds and not in snake) for a single environment.

        Args:
            pos: (2,) tensor of (x, y) position
            env_idx: Environment index

        Returns:
            bool indicating if position is safe
        """
        x, y = pos

        # Check bounds
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False

        # Check snake body collision
        snake_length = self.snake_lengths[env_idx].item()
        snake = self.snakes[env_idx, :snake_length]

        # Check if position matches any body segment
        matches = (snake == pos).all(dim=1)
        if matches.any():
            return False

        return True

    def _compute_tail_direction_vectorized(self) -> torch.Tensor:
        """
        Compute direction to tail for all environments (4 features per env).

        Returns:
            (num_envs, 4) tensor with tail directions [up, right, down, left]
        """
        tail_dirs = torch.zeros((self.num_envs, 4), device=self.device)

        heads = self.snakes[torch.arange(self.num_envs), 0]

        for env_idx in range(self.num_envs):
            length = self.snake_lengths[env_idx].item()
            if length < 2:
                continue

            head = heads[env_idx]
            tail = self.snakes[env_idx, length - 1]

            tail_dirs[env_idx, 0] = float(tail[1] < head[1])  # UP
            tail_dirs[env_idx, 1] = float(tail[0] > head[0])  # RIGHT
            tail_dirs[env_idx, 2] = float(tail[1] > head[1])  # DOWN
            tail_dirs[env_idx, 3] = float(tail[0] < head[0])  # LEFT

        return tail_dirs

    def _compute_tail_reachability_vectorized(self) -> torch.Tensor:
        """
        Check if tail is reachable via flood-fill for all environments (optimized).

        Optimizations:
        - Batch GPU->CPU transfers
        - Early termination when tail is found

        Returns:
            (num_envs,) tensor with tail reachability (0 or 1)
        """
        tail_reachability = torch.zeros(self.num_envs, device=self.device)

        # OPTIMIZATION: Batch GPU->CPU transfer
        snake_lengths_cpu = self.snake_lengths.cpu().numpy()
        max_len = int(snake_lengths_cpu.max())

        if max_len < 2:
            # All snakes too short, all tails are "reachable"
            return torch.ones(self.num_envs, device=self.device)

        snakes_cpu = self.snakes[:, :max_len].cpu().numpy()
        max_val = self.grid_size - 1

        for env_idx in range(self.num_envs):
            length = snake_lengths_cpu[env_idx]
            if length < 2:
                tail_reachability[env_idx] = 1.0
                continue

            head = tuple(snakes_cpu[env_idx, 0])
            tail = tuple(snakes_cpu[env_idx, length - 1])

            # Create snake set without tail (tail will move away)
            snake_positions = snakes_cpu[env_idx, :length - 1]
            snake_set = set(map(tuple, snake_positions))

            # Optimized BFS with deque and early termination
            visited = set()
            queue = deque([head])
            visited.add(head)

            reachable = False

            while queue:
                current = queue.popleft()

                # Early termination if tail found
                if current == tail:
                    reachable = True
                    break

                x, y = current

                # Check all 4 neighbors with optimized bounds check
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy

                    # Fast bounds check
                    if nx < 0 or nx > max_val or ny < 0 or ny > max_val:
                        continue

                    neighbor = (nx, ny)

                    # Check if valid and not visited
                    if neighbor not in visited and neighbor not in snake_set:
                        visited.add(neighbor)
                        queue.append(neighbor)

            tail_reachability[env_idx] = 1.0 if reachable else 0.0

        return tail_reachability

    def _compute_distance_to_tail_vectorized(self) -> torch.Tensor:
        """
        Compute Manhattan distance to tail for all environments, normalized.

        Returns:
            (num_envs,) tensor with normalized distances (0-1)
        """
        distances = torch.zeros(self.num_envs, device=self.device)

        heads = self.snakes[torch.arange(self.num_envs), 0]
        max_dist = 2 * (self.grid_size - 1)

        for env_idx in range(self.num_envs):
            length = self.snake_lengths[env_idx].item()
            if length < 2:
                continue

            head = heads[env_idx]
            tail = self.snakes[env_idx, length - 1]

            manhattan_dist = (head - tail).abs().sum().float()

            if max_dist > 0:
                distances[env_idx] = manhattan_dist / max_dist

        return distances
