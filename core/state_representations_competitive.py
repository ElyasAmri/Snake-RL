"""
Competitive State Representation Encoder for Two-Snake Environment

Provides 35-dimensional feature vector for competitive Snake gameplay:
- Self-awareness (14 dims): Danger, food direction, direction, flood-fill
- Opponent-awareness (15 dims): Opponent danger, head position, direction, metrics
- Competitive metrics (6 dims): Length diff, score diff, food proximity, space control
"""

import torch
from typing import Tuple


class CompetitiveFeatureEncoder:
    """
    Encodes competitive two-snake game state as a 35-dimensional feature vector.

    Features are agent-centric: each snake sees the world from its own perspective,
    treating itself as "self" and the other snake as "opponent".

    Feature breakdown (35 dims):

    Self-awareness (14 dims):
        [0-3]: Danger from walls/self in 4 directions (straight, left, right, back)
        [4-7]: Food direction (up, right, down, left)
        [8-10]: Current direction one-hot (3 bits for 4 directions)
        [11-13]: Flood-fill free space (straight, right, left)

    Opponent-awareness (15 dims):
        [14-17]: Danger from opponent body in 4 directions (straight, left, right, back)
        [18-21]: Opponent head position relative (up, right, down, left)
        [22-24]: Opponent current direction (3 bits)
        [25]: Opponent length normalized (0-1)
        [26]: Manhattan distance to opponent head normalized (0-1)
        [27]: Opponent threat level (0-1) = is_longer + is_closer_to_food
        [28]: Can reach opponent head via flood-fill (0-1)

    Competitive metrics (6 dims):
        [29]: Length difference normalized: (len_self - len_opponent) / max_length
        [30]: Food count difference: (food_self - food_opponent) / target_food
        [31]: Food proximity advantage: (dist_opponent_food - dist_self_food) / grid_diagonal
        [32]: Space control: (flood_self - flood_opponent) / total_cells
        [33]: Steps since last food normalized (0-1)
        [34]: Round progress: food_self / target_food
    """

    def __init__(
        self,
        grid_size: int,
        max_length: int,
        target_food: int,
        device: torch.device = None
    ):
        """
        Initialize competitive feature encoder.

        Args:
            grid_size: Size of the game grid
            max_length: Maximum snake length
            target_food: Target food count to win
            device: PyTorch device (CPU or CUDA)
        """
        self.grid_size = grid_size
        self.max_length = max_length
        self.target_food = target_food
        self.device = device if device is not None else torch.device('cpu')

        # Precompute grid diagonal for normalization
        self.grid_diagonal = (grid_size ** 2 + grid_size ** 2) ** 0.5
        self.total_cells = grid_size * grid_size

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
            features: (batch, 35) - encoded feature vectors
        """
        batch_size = snake_self.shape[0]
        features = torch.zeros((batch_size, 35), dtype=torch.float32, device=self.device)

        # Extract heads
        head_self = snake_self[:, 0, :]  # (batch, 2)
        head_opponent = snake_opponent[:, 0, :]  # (batch, 2)

        # ========== SELF-AWARENESS (14 dims) ==========

        # [0-3]: Danger from walls/self in 4 directions
        features[:, 0:4] = self._compute_danger_self(
            head_self, snake_self, length_self, direction_self
        )

        # [4-7]: Food direction (up, right, down, left)
        features[:, 4:8] = self._compute_food_direction(head_self, food)

        # [8-10]: Current direction one-hot
        features[:, 8:11] = self._encode_direction(direction_self)

        # [11-13]: Flood-fill free space (straight, right, left)
        features[:, 11:14] = self._compute_flood_fill_features(
            head_self, snake_self, length_self, direction_self,
            snake_opponent, length_opponent
        )

        # ========== OPPONENT-AWARENESS (15 dims) ==========

        # [14-17]: Danger from opponent body in 4 directions
        features[:, 14:18] = self._compute_danger_opponent(
            head_self, direction_self, snake_opponent, length_opponent
        )

        # [18-21]: Opponent head position relative (up, right, down, left)
        features[:, 18:22] = self._compute_relative_position(head_self, head_opponent)

        # [22-24]: Opponent direction one-hot
        features[:, 22:25] = self._encode_direction(direction_opponent)

        # [25]: Opponent length normalized
        features[:, 25] = length_opponent.float() / self.max_length

        # [26]: Manhattan distance to opponent head normalized
        features[:, 26] = self._compute_manhattan_distance(
            head_self, head_opponent
        ) / self.grid_diagonal

        # [27]: Opponent threat level
        features[:, 27] = self._compute_threat_level(
            head_self, head_opponent, food, length_self, length_opponent
        )

        # [28]: Can reach opponent head via flood-fill
        features[:, 28] = self._compute_reachability(
            head_self, head_opponent, snake_self, length_self,
            snake_opponent, length_opponent
        )

        # ========== COMPETITIVE METRICS (6 dims) ==========

        # [29]: Length difference normalized
        features[:, 29] = (length_self.float() - length_opponent.float()) / self.max_length

        # [30]: Food count difference
        features[:, 30] = (food_count_self.float() - food_count_opponent.float()) / self.target_food

        # [31]: Food proximity advantage
        dist_self_food = self._compute_manhattan_distance(head_self, food)
        dist_opponent_food = self._compute_manhattan_distance(head_opponent, food)
        features[:, 31] = (dist_opponent_food - dist_self_food) / self.grid_diagonal

        # [32]: Space control (flood-fill difference)
        flood_self = self._compute_flood_fill_total(
            head_self, snake_self, length_self, snake_opponent, length_opponent
        )
        flood_opponent = self._compute_flood_fill_total(
            head_opponent, snake_opponent, length_opponent, snake_self, length_self
        )
        features[:, 32] = (flood_self - flood_opponent) / self.total_cells

        # [33]: Steps since last food normalized
        features[:, 33] = torch.clamp(steps_since_food_self.float() / 100.0, 0.0, 1.0)

        # [34]: Round progress
        features[:, 34] = food_count_self.float() / self.target_food

        return features

    def _compute_danger_self(
        self,
        head: torch.Tensor,
        snake: torch.Tensor,
        length: torch.Tensor,
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute danger from walls and self in 4 directions (straight, left, right, back).

        Returns: (batch, 4) tensor of binary danger flags
        """
        batch_size = head.shape[0]
        dangers = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)

        # Check each relative direction: straight=0, left=-1, right=+1, back=+2
        for i, offset in enumerate([0, -1, 1, 2]):
            check_dir = (direction + offset) % 4  # (batch,)

            # Get delta for each direction
            for b in range(batch_size):
                dir_idx = check_dir[b].item()
                dx, dy = self.direction_deltas[dir_idx]
                next_pos = head[b] + torch.tensor([dx, dy], device=self.device)

                # Check wall collision
                is_wall = (
                    next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                    next_pos[1] < 0 or next_pos[1] >= self.grid_size
                )

                # Check self collision
                is_self_collision = False
                if not is_wall:
                    for j in range(length[b]):
                        if torch.equal(next_pos, snake[b, j]):
                            is_self_collision = True
                            break

                dangers[b, i] = float(is_wall or is_self_collision)

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
                if (next_pos[0] >= 0 and next_pos[0] < self.grid_size and
                    next_pos[1] >= 0 and next_pos[1] < self.grid_size):

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

                if (0 <= nx < self.grid_size and
                    0 <= ny < self.grid_size and
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
        Compute danger from opponent body in 4 directions (straight, left, right, back).

        Returns: (batch, 4) tensor
        """
        batch_size = head.shape[0]
        dangers = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)

        for i, offset in enumerate([0, -1, 1, 2]):
            check_dir = (direction + offset) % 4

            for b in range(batch_size):
                dir_idx = check_dir[b].item()
                dx, dy = self.direction_deltas[dir_idx]
                next_pos = head[b] + torch.tensor([dx, dy], device=self.device)

                # Check collision with opponent body
                for j in range(length_opponent[b]):
                    if torch.equal(next_pos, snake_opponent[b, j]):
                        dangers[b, i] = 1.0
                        break

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

                    if (0 <= nx < self.grid_size and
                        0 <= ny < self.grid_size and
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
