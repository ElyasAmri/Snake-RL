"""
Competitive State Representation Encoder for Two-Snake Environment

Provides 33-dimensional feature vector for competitive Snake gameplay:
- Self-awareness (13 dims): Danger, food direction, direction, flood-fill
- Opponent-awareness (14 dims): Opponent danger, head position, direction, metrics
- Competitive metrics (6 dims): Length diff, score diff, food proximity, space control
"""

import torch
import time
from typing import Tuple


class CompetitiveFeatureEncoder:
    """
    Encodes competitive two-snake game state as a 33-dimensional feature vector.

    Features are agent-centric: each snake sees the world from its own perspective,
    treating itself as "self" and the other snake as "opponent".

    Feature breakdown (33 dims):

    Self-awareness (13 dims):
        [0-2]: Danger from walls/self in 3 directions (straight, left, right)
        [3-6]: Food direction (up, right, down, left)
        [7-9]: Current direction one-hot (3 bits for 4 directions)
        [10-12]: Flood-fill free space (straight, right, left)

    Opponent-awareness (14 dims):
        [13-15]: Danger from opponent body in 3 directions (straight, left, right)
        [16-19]: Opponent head position relative (up, right, down, left)
        [20-22]: Opponent current direction (3 bits)
        [23]: Opponent length normalized (0-1)
        [24]: Manhattan distance to opponent head normalized (0-1)
        [25]: Opponent threat level (0-1) = is_longer + is_closer_to_food
        [26]: Can reach opponent head via flood-fill (0-1)

    Competitive metrics (6 dims):
        [27]: Length difference normalized: (len_self - len_opponent) / max_length
        [28]: Food count difference: (food_self - food_opponent) / target_food
        [29]: Food proximity advantage: (dist_opponent_food - dist_self_food) / grid_diagonal
        [30]: Space control: (flood_self - flood_opponent) / total_cells
        [31]: Steps since last food normalized (0-1)
        [32]: Round progress: food_self / target_food
    """

    def __init__(
        self,
        grid_width: int = None,
        grid_height: int = None,
        grid_size: int = None,  # DEPRECATED: Use grid_width and grid_height instead
        max_length: int = None,
        target_food: int = None,
        device: torch.device = None,
        use_flood_fill: bool = False
    ):
        """
        Initialize competitive feature encoder.

        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            grid_size: DEPRECATED - Use grid_width and grid_height instead
            max_length: Maximum snake length
            target_food: Target food count to win
            device: PyTorch device (CPU or CUDA)
            use_flood_fill: Enable CPU-intensive flood-fill features (default: False for speed)
        """
        # Handle backward compatibility for grid dimensions
        if grid_width is None and grid_height is None:
            if grid_size is None:
                raise ValueError("Must specify either (grid_width, grid_height) or grid_size")
            self.grid_width = grid_size
            self.grid_height = grid_size
        else:
            assert grid_width is not None and grid_height is not None, \
                "Both grid_width and grid_height must be specified"
            self.grid_width = grid_width
            self.grid_height = grid_height

        self.max_length = max_length
        self.target_food = target_food
        self.device = device if device is not None else torch.device('cpu')
        self.use_flood_fill = use_flood_fill

        # Precompute grid diagonal for normalization
        self.grid_diagonal = (self.grid_width ** 2 + self.grid_height ** 2) ** 0.5
        self.total_cells = self.grid_width * self.grid_height

        # Direction deltas: UP=0, RIGHT=1, DOWN=2, LEFT=3
        self.direction_deltas = torch.tensor([
            [0, -1],  # UP
            [1, 0],   # RIGHT
            [0, 1],   # DOWN
            [-1, 0]   # LEFT
        ], dtype=torch.long, device=self.device)

    def encode_batch(
        self,
        snake_self: torch.Tensor,
        length_self: torch.Tensor,
        direction_self: torch.Tensor,
        food_count_self: torch.Tensor,
        steps_since_food_self: torch.Tensor,
        snake_opponent: torch.Tensor,
        length_opponent: torch.Tensor,
        direction_opponent: torch.Tensor,
        food_count_opponent: torch.Tensor,
        food: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode batch of competitive game states.

        Args:
            snake_self: (batch, max_length, 2) - self snake positions
            length_self: (batch,) - self snake lengths
            direction_self: (batch,) - self snake directions
            food_count_self: (batch,) - self food collected
            steps_since_food_self: (batch,) - steps since last food
            snake_opponent: (batch, max_length, 2) - opponent snake positions
            length_opponent: (batch,) - opponent snake lengths
            direction_opponent: (batch,) - opponent snake directions
            food_count_opponent: (batch,) - opponent food collected
            food: (batch, 2) - food positions

        Returns:
            features: (batch, 33) - encoded feature vectors
        """
        batch_size = snake_self.shape[0]
        features = torch.zeros((batch_size, 33), dtype=torch.float32, device=self.device)

        # Profiling: Track feature computation timing (accumulate totals)
        if not hasattr(self, '_encoding_call_count'):
            self._encoding_call_count = 0
            self._total_danger_self_time = 0.0
            self._total_danger_opp_time = 0.0
            self._total_food_dir_time = 0.0
        self._encoding_call_count += 1

        # Extract heads
        head_self = snake_self[:, 0, :]  # (batch, 2)
        head_opponent = snake_opponent[:, 0, :]  # (batch, 2)

        # ========== SELF-AWARENESS (13 dims) ==========

        # [0-2]: Danger from walls/self in 3 directions
        t0 = time.perf_counter()
        features[:, 0:3] = self._compute_danger_self(
            head_self, snake_self, length_self, direction_self
        )
        self._total_danger_self_time += time.perf_counter() - t0

        # [3-6]: Food direction (up, right, down, left)
        t0 = time.perf_counter()
        features[:, 3:7] = self._compute_food_direction(head_self, food)
        self._total_food_dir_time += time.perf_counter() - t0

        # [7-9]: Current direction one-hot
        features[:, 7:10] = self._encode_direction(direction_self)

        # [10-12]: Flood-fill free space (straight, right, left)
        if self.use_flood_fill:
            features[:, 10:13] = self._compute_flood_fill_features(
                head_self, snake_self, length_self, direction_self,
                snake_opponent, length_opponent
            )
        else:
            # Use simple heuristic: free space = 1.0 - danger
            features[:, 10:13] = 1.0 - features[:, 0:3]

        # ========== OPPONENT-AWARENESS (14 dims) ==========

        # [13-15]: Danger from opponent body in 3 directions
        t0 = time.perf_counter()
        features[:, 13:16] = self._compute_danger_opponent(
            head_self, direction_self, snake_opponent, length_opponent
        )
        self._total_danger_opp_time += time.perf_counter() - t0

        # [16-19]: Opponent head position relative (up, right, down, left)
        features[:, 16:20] = self._compute_relative_position(head_self, head_opponent)

        # [20-22]: Opponent direction one-hot
        features[:, 20:23] = self._encode_direction(direction_opponent)

        # [23]: Opponent length normalized
        features[:, 23] = length_opponent.float() / self.max_length

        # [24]: Manhattan distance to opponent head normalized
        features[:, 24] = self._compute_manhattan_distance(
            head_self, head_opponent
        ) / self.grid_diagonal

        # [25]: Opponent threat level
        features[:, 25] = self._compute_threat_level(
            head_self, head_opponent, food, length_self, length_opponent
        )

        # [26]: Can reach opponent head via flood-fill
        if self.use_flood_fill:
            features[:, 26] = self._compute_reachability(
                head_self, head_opponent, snake_self, length_self,
                snake_opponent, length_opponent
            )
        else:
            # Use simple heuristic: reachable if distance < average dimension / 2
            avg_dimension = (self.grid_width + self.grid_height) / 2
            dist = self._compute_manhattan_distance(head_self, head_opponent)
            features[:, 26] = (dist < avg_dimension / 2).float()

        # ========== COMPETITIVE METRICS (6 dims) ==========

        # [27]: Length difference normalized
        features[:, 27] = (length_self.float() - length_opponent.float()) / self.max_length

        # [28]: Food count difference
        features[:, 28] = (food_count_self.float() - food_count_opponent.float()) / self.target_food

        # [29]: Food proximity advantage
        dist_self_food = self._compute_manhattan_distance(head_self, food)
        dist_opponent_food = self._compute_manhattan_distance(head_opponent, food)
        features[:, 29] = (dist_opponent_food - dist_self_food) / self.grid_diagonal

        # [30]: Space control (flood-fill difference)
        if self.use_flood_fill:
            flood_self = self._compute_flood_fill_total(
                head_self, snake_self, length_self, snake_opponent, length_opponent
            )
            flood_opponent = self._compute_flood_fill_total(
                head_opponent, snake_opponent, length_opponent, snake_self, length_self
            )
            features[:, 30] = (flood_self - flood_opponent) / self.total_cells
        else:
            # Use length difference as proxy for space control
            features[:, 30] = (length_self.float() - length_opponent.float()) / self.max_length

        # [31]: Steps since last food normalized
        features[:, 31] = torch.clamp(steps_since_food_self.float() / 100.0, 0.0, 1.0)

        # [32]: Round progress
        features[:, 32] = food_count_self.float() / self.target_food

        return features

    def get_profiling_stats(self):
        """Get accumulated profiling statistics"""
        if not hasattr(self, '_encoding_call_count') or self._encoding_call_count == 0:
            return None

        return {
            'total_encodings': self._encoding_call_count,
            'avg_danger_self_ms': (self._total_danger_self_time / self._encoding_call_count) * 1000,
            'avg_danger_opp_ms': (self._total_danger_opp_time / self._encoding_call_count) * 1000,
            'avg_food_dir_ms': (self._total_food_dir_time / self._encoding_call_count) * 1000,
            'total_time_s': self._total_danger_self_time + self._total_danger_opp_time + self._total_food_dir_time
        }

    def _compute_danger_self(
        self,
        head: torch.Tensor,
        snake: torch.Tensor,
        length: torch.Tensor,
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute danger from walls and self in 3 directions (straight, left, right).

        VECTORIZED VERSION - No Python loops, pure tensor operations.

        Returns: (batch, 3) tensor of binary danger flags
        """
        batch_size = head.shape[0]

        # Compute next positions for all 3 relative directions at once
        # relative_offsets: [straight=0, left=-1, right=+1]
        relative_offsets = torch.tensor([0, -1, 1], device=self.device)  # (3,)
        check_dirs = (direction.unsqueeze(1) + relative_offsets.unsqueeze(0)) % 4  # (batch, 3)

        # Get deltas for all directions: (batch, 3, 2)
        deltas = self.direction_deltas[check_dirs]  # (batch, 3, 2)

        # Compute next positions: (batch, 3, 2)
        next_positions = head.unsqueeze(1) + deltas  # (batch, 1, 2) + (batch, 3, 2) = (batch, 3, 2)

        # Check wall collisions (vectorized): (batch, 3)
        x_coords = next_positions[:, :, 0]  # (batch, 3)
        y_coords = next_positions[:, :, 1]  # (batch, 3)
        is_wall = (
            (x_coords < 0) | (x_coords >= self.grid_width) |   # X bound
            (y_coords < 0) | (y_coords >= self.grid_height)    # Y bound
        ).float()  # (batch, 3)

        # Check self collisions (vectorized)
        # Compare next_positions (batch, 3, 2) with snake body (batch, max_length, 2)
        # Expand dimensions: next_pos (batch, 3, 1, 2) vs snake (batch, 1, max_length, 2)
        next_pos_expanded = next_positions.unsqueeze(2)  # (batch, 3, 1, 2)
        snake_expanded = snake.unsqueeze(1)  # (batch, 1, max_length, 2)

        # Check if next_pos matches any snake segment: (batch, 3, max_length)
        matches = ((next_pos_expanded == snake_expanded).all(dim=-1))  # (batch, 3, max_length)

        # Create mask for valid snake segments based on length
        segment_indices = torch.arange(snake.shape[1], device=self.device)  # (max_length,)
        valid_segments = segment_indices.unsqueeze(0) < length.unsqueeze(1)  # (batch, max_length)
        valid_segments = valid_segments.unsqueeze(1)  # (batch, 1, max_length)

        # Check if any valid segment matches: (batch, 3)
        is_self_collision = (matches & valid_segments).any(dim=2).float()  # (batch, 3)

        # Combine: danger if wall OR self-collision
        dangers = torch.maximum(is_wall, is_self_collision)  # (batch, 3)

        return dangers

    def _compute_food_direction(
        self,
        head: torch.Tensor,
        food: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute food direction as 4 binary features (up, right, down, left).

        Returns: (batch, 4) tensor
        """
        batch_size = head.shape[0]
        food_dir = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)

        # UP: food_y < head_y
        food_dir[:, 0] = (food[:, 1] < head[:, 1]).float()
        # RIGHT: food_x > head_x
        food_dir[:, 1] = (food[:, 0] > head[:, 0]).float()
        # DOWN: food_y > head_y
        food_dir[:, 2] = (food[:, 1] > head[:, 1]).float()
        # LEFT: food_x < head_x
        food_dir[:, 3] = (food[:, 0] < head[:, 0]).float()

        return food_dir

    def _encode_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Encode direction as 3-bit one-hot (4th is implicit).

        Args:
            direction: (batch,) tensor of directions 0-3

        Returns: (batch, 3) tensor
        """
        batch_size = direction.shape[0]
        encoded = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)

        for i in range(3):
            encoded[:, i] = (direction == i).float()

        return encoded

    def _compute_flood_fill_features(
        self,
        head: torch.Tensor,
        snake_self: torch.Tensor,
        length_self: torch.Tensor,
        direction: torch.Tensor,
        snake_opponent: torch.Tensor,
        length_opponent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flood-fill free space for straight, right, and left directions.

        Returns: (batch, 3) tensor of normalized free space values
        """
        batch_size = head.shape[0]
        flood_features = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)

        # For each direction offset: straight=0, right=+1, left=-1
        for i, offset in enumerate([0, 1, -1]):
            check_dir = (direction + offset) % 4

            for b in range(batch_size):
                dir_idx = check_dir[b].item()
                dx, dy = self.direction_deltas[dir_idx]
                next_pos = head[b] + torch.tensor([dx, dy], device=self.device)

                # Check if position is safe
                if (next_pos[0] >= 0 and next_pos[0] < self.grid_width and
                    next_pos[1] >= 0 and next_pos[1] < self.grid_height):

                    # Perform flood-fill from this position
                    free_space = self._flood_fill_from_position(
                        next_pos, snake_self[b], length_self[b],
                        snake_opponent[b], length_opponent[b]
                    )

                    # Normalize by max possible free space
                    max_free = self.total_cells - length_self[b].item() - length_opponent[b].item()
                    if max_free > 0:
                        flood_features[b, i] = min(1.0, free_space / max_free)

        return flood_features

    def _flood_fill_from_position(
        self,
        start_pos: torch.Tensor,
        snake_self: torch.Tensor,
        length_self: torch.Tensor,
        snake_opponent: torch.Tensor,
        length_opponent: torch.Tensor
    ) -> float:
        """
        BFS flood-fill to count reachable free cells from start_pos.

        Returns: Number of reachable cells
        """
        visited = set()
        queue = [tuple(start_pos.cpu().numpy())]
        visited.add(queue[0])

        # Create obstacle set (both snakes)
        obstacles = set()
        for i in range(length_self.item()):
            pos = tuple(snake_self[i].cpu().numpy())
            obstacles.add(pos)
        for i in range(length_opponent.item()):
            pos = tuple(snake_opponent[i].cpu().numpy())
            obstacles.add(pos)

        while queue:
            current = queue.pop(0)
            x, y = current

            # Check all 4 neighbors
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor

                if (0 <= nx < self.grid_width and
                    0 <= ny < self.grid_height and
                    neighbor not in visited and
                    neighbor not in obstacles):
                    visited.add(neighbor)
                    queue.append(neighbor)

        return float(len(visited))

    def _compute_danger_opponent(
        self,
        head: torch.Tensor,
        direction: torch.Tensor,
        snake_opponent: torch.Tensor,
        length_opponent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute danger from opponent body in 3 directions (straight, left, right).

        VECTORIZED VERSION - No Python loops, pure tensor operations.

        Returns: (batch, 3) tensor
        """
        batch_size = head.shape[0]

        # Compute next positions for all 3 relative directions at once
        # relative_offsets: [straight=0, left=-1, right=+1]
        relative_offsets = torch.tensor([0, -1, 1], device=self.device)  # (3,)
        check_dirs = (direction.unsqueeze(1) + relative_offsets.unsqueeze(0)) % 4  # (batch, 3)

        # Get deltas for all directions: (batch, 3, 2)
        deltas = self.direction_deltas[check_dirs]  # (batch, 3, 2)

        # Compute next positions: (batch, 3, 2)
        next_positions = head.unsqueeze(1) + deltas  # (batch, 1, 2) + (batch, 3, 2) = (batch, 3, 2)

        # Check collisions with opponent body (vectorized)
        # Compare next_positions (batch, 3, 2) with opponent snake (batch, max_length, 2)
        # Expand dimensions: next_pos (batch, 3, 1, 2) vs snake_opp (batch, 1, max_length, 2)
        next_pos_expanded = next_positions.unsqueeze(2)  # (batch, 3, 1, 2)
        snake_expanded = snake_opponent.unsqueeze(1)  # (batch, 1, max_length, 2)

        # Check if next_pos matches any opponent segment: (batch, 3, max_length)
        matches = ((next_pos_expanded == snake_expanded).all(dim=-1))  # (batch, 3, max_length)

        # Create mask for valid opponent segments based on length
        segment_indices = torch.arange(snake_opponent.shape[1], device=self.device)  # (max_length,)
        valid_segments = segment_indices.unsqueeze(0) < length_opponent.unsqueeze(1)  # (batch, max_length)
        valid_segments = valid_segments.unsqueeze(1)  # (batch, 1, max_length)

        # Check if any valid segment matches: (batch, 3)
        dangers = (matches & valid_segments).any(dim=2).float()  # (batch, 3)

        return dangers

    def _compute_relative_position(
        self,
        head_self: torch.Tensor,
        head_opponent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute opponent head position relative to self (up, right, down, left).

        Returns: (batch, 4) tensor
        """
        batch_size = head_self.shape[0]
        relative = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)

        # UP: opponent_y < self_y
        relative[:, 0] = (head_opponent[:, 1] < head_self[:, 1]).float()
        # RIGHT: opponent_x > self_x
        relative[:, 1] = (head_opponent[:, 0] > head_self[:, 0]).float()
        # DOWN: opponent_y > self_y
        relative[:, 2] = (head_opponent[:, 1] > head_self[:, 1]).float()
        # LEFT: opponent_x < self_x
        relative[:, 3] = (head_opponent[:, 0] < head_self[:, 0]).float()

        return relative

    def _compute_manhattan_distance(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Manhattan distance between two positions.

        Returns: (batch,) tensor
        """
        return torch.abs(pos1[:, 0] - pos2[:, 0]) + torch.abs(pos1[:, 1] - pos2[:, 1])

    def _compute_threat_level(
        self,
        head_self: torch.Tensor,
        head_opponent: torch.Tensor,
        food: torch.Tensor,
        length_self: torch.Tensor,
        length_opponent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute opponent threat level (0-1) based on length and food proximity.

        Threat = 0.5 * is_longer + 0.5 * is_closer_to_food

        Returns: (batch,) tensor
        """
        batch_size = head_self.shape[0]
        threat = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        # Component 1: Is opponent longer?
        is_longer = (length_opponent > length_self).float() * 0.5

        # Component 2: Is opponent closer to food?
        dist_self_food = self._compute_manhattan_distance(head_self, food)
        dist_opponent_food = self._compute_manhattan_distance(head_opponent, food)
        is_closer = (dist_opponent_food < dist_self_food).float() * 0.5

        threat = is_longer + is_closer
        return threat

    def _compute_reachability(
        self,
        head_self: torch.Tensor,
        head_opponent: torch.Tensor,
        snake_self: torch.Tensor,
        length_self: torch.Tensor,
        snake_opponent: torch.Tensor,
        length_opponent: torch.Tensor
    ) -> torch.Tensor:
        """
        Check if opponent head is reachable via flood-fill.

        Returns: (batch,) tensor of binary flags
        """
        batch_size = head_self.shape[0]
        reachable = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for b in range(batch_size):
            # Perform flood-fill from self head
            visited = set()
            queue = [tuple(head_self[b].cpu().numpy())]
            visited.add(queue[0])

            # Create obstacle set
            obstacles = set()
            for i in range(length_self[b].item()):
                if i > 0:  # Exclude head
                    obstacles.add(tuple(snake_self[b, i].cpu().numpy()))
            for i in range(length_opponent[b].item()):
                if i > 0:  # Exclude opponent head (we want to reach it)
                    obstacles.add(tuple(snake_opponent[b, i].cpu().numpy()))

            target = tuple(head_opponent[b].cpu().numpy())
            found = False

            while queue and not found:
                current = queue.pop(0)

                if current == target:
                    found = True
                    break

                x, y = current
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    neighbor = (x + dx, y + dy)
                    nx, ny = neighbor

                    if (0 <= nx < self.grid_width and
                        0 <= ny < self.grid_height and
                        neighbor not in visited and
                        neighbor not in obstacles):
                        visited.add(neighbor)
                        queue.append(neighbor)

            reachable[b] = float(found)

        return reachable

    def _compute_flood_fill_total(
        self,
        head: torch.Tensor,
        snake_self: torch.Tensor,
        length_self: torch.Tensor,
        snake_opponent: torch.Tensor,
        length_opponent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total reachable free space from head position.

        Returns: (batch,) tensor of reachable cell counts
        """
        batch_size = head.shape[0]
        total_space = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for b in range(batch_size):
            free_space = self._flood_fill_from_position(
                head[b], snake_self[b], length_self[b],
                snake_opponent[b], length_opponent[b]
            )
            total_space[b] = free_space

        return total_space


class CompetitiveGridEncoder:
    """
    Encodes competitive two-snake game state as a multi-channel grid.

    Grid-based CNN representation for competitive Snake gameplay.
    Each snake sees the world from its own perspective (agent-centric).

    Channels (5 total):
        0: Self head position
        1: Self body positions
        2: Opponent head position
        3: Opponent body positions
        4: Food position

    Output shape: (batch_size, grid_height, grid_width, 5)
    """

    def __init__(
        self,
        grid_width: int = None,
        grid_height: int = None,
        grid_size: int = None,  # DEPRECATED: Use grid_width and grid_height instead
        device: torch.device = None
    ):
        """
        Initialize competitive grid encoder.

        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            grid_size: DEPRECATED - Use grid_width and grid_height instead
            device: PyTorch device (CPU or CUDA)
        """
        # Handle backward compatibility for grid dimensions
        if grid_width is None and grid_height is None:
            if grid_size is None:
                raise ValueError("Must specify either (grid_width, grid_height) or grid_size")
            self.grid_width = grid_size
            self.grid_height = grid_size
        else:
            assert grid_width is not None and grid_height is not None, \
                "Both grid_width and grid_height must be specified"
            self.grid_width = grid_width
            self.grid_height = grid_height

        self.device = device if device is not None else torch.device('cpu')

        # Profiling
        self._encoding_call_count = 0
        self._total_encoding_time = 0.0

    def encode_batch(
        self,
        snake_self: torch.Tensor,
        length_self: torch.Tensor,
        snake_opponent: torch.Tensor,
        length_opponent: torch.Tensor,
        food: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode batch of competitive game states as grids (VECTORIZED).

        Args:
            snake_self: (batch, max_length, 2) - self snake positions
            length_self: (batch,) - self snake lengths
            snake_opponent: (batch, max_length, 2) - opponent snake positions
            length_opponent: (batch,) - opponent snake lengths
            food: (batch, 2) - food positions

        Returns:
            (batch, grid_height, grid_width, 5) grid tensor
        """
        import time
        t0 = time.perf_counter()

        batch_size = snake_self.shape[0]
        max_length = snake_self.shape[1]

        grid = torch.zeros(
            (batch_size, self.grid_height, self.grid_width, 5),
            dtype=torch.float32,
            device=self.device
        )

        # Channel 0: Self heads (all at once)
        head_pos_self = snake_self[:, 0, :].long()  # (batch, 2)
        valid_heads_self = (
            (head_pos_self[:, 0] >= 0) &
            (head_pos_self[:, 0] < self.grid_width) &   # X bound
            (head_pos_self[:, 1] >= 0) &
            (head_pos_self[:, 1] < self.grid_height)    # Y bound
        )
        valid_batch_idx = torch.arange(batch_size, device=self.device)[valid_heads_self]
        valid_x = head_pos_self[valid_heads_self, 0]
        valid_y = head_pos_self[valid_heads_self, 1]
        grid[valid_batch_idx, valid_y, valid_x, 0] = 1.0

        # Channel 1: Self bodies (excluding head) - vectorized with masking
        segment_indices = torch.arange(1, max_length, device=self.device).unsqueeze(0)  # (1, max_length-1)
        body_mask_self = segment_indices < length_self.unsqueeze(1)  # (batch, max_length-1)

        body_positions_self = snake_self[:, 1:, :].long()  # (batch, max_length-1, 2)
        valid_body_self = (
            body_mask_self &
            (body_positions_self[..., 0] >= 0) &
            (body_positions_self[..., 0] < self.grid_width) &   # X bound
            (body_positions_self[..., 1] >= 0) &
            (body_positions_self[..., 1] < self.grid_height)    # Y bound
        )

        # Get flattened indices for valid body segments
        batch_idx_body_self = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, max_length-1)[valid_body_self]
        x_body_self = body_positions_self[..., 0][valid_body_self]
        y_body_self = body_positions_self[..., 1][valid_body_self]
        grid[batch_idx_body_self, y_body_self, x_body_self, 1] = 1.0

        # Channel 2: Opponent heads (all at once)
        head_pos_opp = snake_opponent[:, 0, :].long()  # (batch, 2)
        valid_heads_opp = (
            (head_pos_opp[:, 0] >= 0) &
            (head_pos_opp[:, 0] < self.grid_width) &   # X bound
            (head_pos_opp[:, 1] >= 0) &
            (head_pos_opp[:, 1] < self.grid_height)    # Y bound
        )
        valid_batch_idx_opp = torch.arange(batch_size, device=self.device)[valid_heads_opp]
        valid_x_opp = head_pos_opp[valid_heads_opp, 0]
        valid_y_opp = head_pos_opp[valid_heads_opp, 1]
        grid[valid_batch_idx_opp, valid_y_opp, valid_x_opp, 2] = 1.0

        # Channel 3: Opponent bodies (excluding head) - vectorized with masking
        body_mask_opp = segment_indices < length_opponent.unsqueeze(1)  # (batch, max_length-1)

        body_positions_opp = snake_opponent[:, 1:, :].long()  # (batch, max_length-1, 2)
        valid_body_opp = (
            body_mask_opp &
            (body_positions_opp[..., 0] >= 0) &
            (body_positions_opp[..., 0] < self.grid_width) &   # X bound
            (body_positions_opp[..., 1] >= 0) &
            (body_positions_opp[..., 1] < self.grid_height)    # Y bound
        )

        # Get flattened indices for valid body segments
        batch_idx_body_opp = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, max_length-1)[valid_body_opp]
        x_body_opp = body_positions_opp[..., 0][valid_body_opp]
        y_body_opp = body_positions_opp[..., 1][valid_body_opp]
        grid[batch_idx_body_opp, y_body_opp, x_body_opp, 3] = 1.0

        # Channel 4: Food (all at once)
        food_pos = food.long()  # (batch, 2)
        valid_food = (
            (food_pos[:, 0] >= 0) &
            (food_pos[:, 0] < self.grid_width) &   # X bound
            (food_pos[:, 1] >= 0) &
            (food_pos[:, 1] < self.grid_height)    # Y bound
        )
        valid_batch_idx_food = torch.arange(batch_size, device=self.device)[valid_food]
        valid_x_food = food_pos[valid_food, 0]
        valid_y_food = food_pos[valid_food, 1]
        grid[valid_batch_idx_food, valid_y_food, valid_x_food, 4] = 1.0

        # Update profiling
        self._encoding_call_count += 1
        self._total_encoding_time += time.perf_counter() - t0

        return grid

    def get_profiling_stats(self):
        """Get accumulated profiling statistics"""
        if self._encoding_call_count == 0:
            return None

        return {
            'total_encodings': self._encoding_call_count,
            'avg_encoding_ms': (self._total_encoding_time / self._encoding_call_count) * 1000,
            'total_time_s': self._total_encoding_time
        }
