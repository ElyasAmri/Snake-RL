"""
Scripted Opponents for Curriculum Learning

Provides baseline opponents with increasing difficulty:
- StaticAgent: Always goes straight (Stage 0)
- RandomAgent: Random valid actions (Stage 1)
- GreedyFoodAgent: BFS pathfinding to food (Stage 2)
- DefensiveAgent: Avoids opponent, seeks food when safe (Stage 3)
"""

import torch
from typing import Tuple
from collections import deque


class ScriptedAgent:
    """Base class for scripted opponents"""

    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')

    def select_action(self, env) -> torch.Tensor:
        """
        Select actions for all environments.

        Args:
            env: VectorizedTwoSnakeEnv instance

        Returns:
            actions: (num_envs,) tensor of actions (0=STRAIGHT, 1=RIGHT, 2=LEFT)
        """
        raise NotImplementedError


class StaticAgent(ScriptedAgent):
    """
    Stage 0: Static opponent that always goes straight.

    Easiest opponent - helps agent learn basic movement and food collection.
    """

    def select_action(self, env) -> torch.Tensor:
        """Always return STRAIGHT action (0)"""
        return torch.zeros(env.num_envs, dtype=torch.long, device=self.device)


class RandomAgent(ScriptedAgent):
    """
    Stage 1: Random opponent that takes random valid actions.

    Introduces unpredictability while still being easy to beat.
    """

    def select_action(self, env) -> torch.Tensor:
        """Return random actions for all environments"""
        return torch.randint(0, 3, (env.num_envs,), dtype=torch.long, device=self.device)


class GreedyFoodAgent(ScriptedAgent):
    """
    Stage 2: Greedy opponent that uses BFS to find shortest path to food.

    More challenging - always moves optimally toward food.
    Agent must learn to compete for food and use space control.
    """

    def select_action(self, env) -> torch.Tensor:
        """Use BFS to find shortest path to food and take first step"""
        actions = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)

        for env_idx in range(env.num_envs):
            # Skip if snake is dead
            if not env.alive2[env_idx]:
                actions[env_idx] = 0  # STRAIGHT (doesn't matter)
                continue

            # Get snake head, direction, and food position
            head = env.snakes2[env_idx, 0].cpu().numpy()
            direction = env.directions2[env_idx].item()
            food = env.foods[env_idx].cpu().numpy()

            # Get all snake body positions (obstacles)
            snake1_body = env.snakes1[env_idx, :env.lengths1[env_idx]].cpu().numpy()
            snake2_body = env.snakes2[env_idx, :env.lengths2[env_idx]].cpu().numpy()
            obstacles = set(map(tuple, snake1_body)) | set(map(tuple, snake2_body))

            # BFS to find shortest path
            # Use grid_width (assuming square grid, or use min of both for safety)
            grid_size = getattr(env, 'grid_size', None) or env.grid_width
            action = self._bfs_to_food(
                tuple(head), tuple(food), direction,
                grid_size, obstacles
            )
            actions[env_idx] = action

        return actions

    def _bfs_to_food(
        self,
        start: Tuple[int, int],
        food: Tuple[int, int],
        current_direction: int,
        grid_size: int,
        obstacles: set
    ) -> int:
        """
        BFS to find shortest path to food and return first action.

        Returns:
            action: 0=STRAIGHT, 1=RIGHT, 2=LEFT
        """
        # Direction vectors: UP=0, RIGHT=1, DOWN=2, LEFT=3
        direction_deltas = {
            0: (0, -1),  # UP
            1: (1, 0),   # RIGHT
            2: (0, 1),   # DOWN
            3: (-1, 0)   # LEFT
        }

        # If already at food, go straight
        if start == food:
            return 0

        # BFS
        queue = deque([(start, [])])  # (position, path_of_directions)
        visited = {start}

        while queue:
            current, path = queue.popleft()

            # Check all 4 directions
            for next_dir in range(4):
                dx, dy = direction_deltas[next_dir]
                next_pos = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
                    continue

                # Check if visited or obstacle
                if next_pos in visited or next_pos in obstacles:
                    continue

                visited.add(next_pos)
                new_path = path + [next_dir]

                # Found food!
                if next_pos == food:
                    # Convert first direction in path to relative action
                    if len(new_path) > 0:
                        return self._direction_to_action(current_direction, new_path[0])
                    return 0  # Shouldn't happen, but return straight

                queue.append((next_pos, new_path))

        # No path found - return random safe action
        return self._get_safe_action(start, current_direction, grid_size, obstacles)

    def _direction_to_action(self, current_dir: int, target_dir: int) -> int:
        """
        Convert absolute direction to relative action.

        Args:
            current_dir: Current direction (0-3)
            target_dir: Target direction (0-3)

        Returns:
            action: 0=STRAIGHT, 1=RIGHT, 2=LEFT
        """
        # Calculate turn needed
        turn = (target_dir - current_dir) % 4

        if turn == 0:
            return 0  # STRAIGHT
        elif turn == 1:
            return 1  # RIGHT
        elif turn == 3:
            return 2  # LEFT
        else:  # turn == 2 (180 degree turn, invalid)
            # Should not happen in valid snake movement
            return 0  # STRAIGHT

    def _get_safe_action(
        self,
        pos: Tuple[int, int],
        direction: int,
        grid_size: int,
        obstacles: set
    ) -> int:
        """Find a safe action (no immediate collision)"""
        direction_deltas = {
            0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)
        }

        # Try straight, right, left in order
        for action_offset in [0, 1, 2]:
            test_dir = (direction + (action_offset if action_offset <= 1 else -1)) % 4
            dx, dy = direction_deltas[test_dir]
            next_pos = (pos[0] + dx, pos[1] + dy)

            # Check if safe
            if (0 <= next_pos[0] < grid_size and
                0 <= next_pos[1] < grid_size and
                next_pos not in obstacles):
                return action_offset if action_offset <= 1 else 2

        # No safe action, return straight
        return 0


class DefensiveAgent(ScriptedAgent):
    """
    Stage 3: Defensive opponent that avoids the player and seeks food when safe.

    Most challenging scripted opponent:
    - Uses flood-fill to assess space control
    - Avoids head-to-head confrontations when smaller
    - Seeks food aggressively when ahead
    """

    def __init__(self, device: torch.device = None, safety_threshold: float = 0.3):
        super().__init__(device)
        self.safety_threshold = safety_threshold  # Minimum space ratio to be aggressive

    def select_action(self, env) -> torch.Tensor:
        """Select defensive/aggressive action based on game state"""
        actions = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)

        for env_idx in range(env.num_envs):
            # Skip if snake is dead
            if not env.alive2[env_idx]:
                actions[env_idx] = 0
                continue

            # Get state
            head = env.snakes2[env_idx, 0].cpu().numpy()
            direction = env.directions2[env_idx].item()
            food = env.foods[env_idx].cpu().numpy()
            opponent_head = env.snakes1[env_idx, 0].cpu().numpy()

            # Get snake bodies
            snake1_body = env.snakes1[env_idx, :env.lengths1[env_idx]].cpu().numpy()
            snake2_body = env.snakes2[env_idx, :env.lengths2[env_idx]].cpu().numpy()
            obstacles = set(map(tuple, snake1_body)) | set(map(tuple, snake2_body))

            # Calculate space control (use grid_width for square grids)
            grid_size = getattr(env, 'grid_size', None) or env.grid_width
            my_space = self._flood_fill(tuple(head), grid_size, obstacles - {tuple(head)})
            opponent_space = self._flood_fill(tuple(opponent_head), grid_size, obstacles - {tuple(opponent_head)})

            total_free = my_space + opponent_space
            my_ratio = my_space / max(total_free, 1)

            # Decide strategy
            if my_ratio < self.safety_threshold:
                # Defensive: seek maximum space
                action = self._seek_space(tuple(head), direction, grid_size, obstacles)
            else:
                # Aggressive: go for food
                action = self._seek_food_cautiously(
                    tuple(head), tuple(food), tuple(opponent_head),
                    direction, grid_size, obstacles
                )

            actions[env_idx] = action

        return actions

    def _flood_fill(self, start: Tuple[int, int], grid_size: int, obstacles: set) -> int:
        """Count reachable cells from start position"""
        if start in obstacles:
            return 0

        visited = {start}
        queue = deque([start])

        direction_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        while queue:
            x, y = queue.popleft()

            for dx, dy in direction_deltas:
                next_pos = (x + dx, y + dy)

                if (0 <= next_pos[0] < grid_size and
                    0 <= next_pos[1] < grid_size and
                    next_pos not in visited and
                    next_pos not in obstacles):
                    visited.add(next_pos)
                    queue.append(next_pos)

        return len(visited)

    def _seek_space(
        self,
        pos: Tuple[int, int],
        direction: int,
        grid_size: int,
        obstacles: set
    ) -> int:
        """Choose action that leads to maximum reachable space"""
        direction_deltas = {
            0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)
        }

        best_action = 0
        best_space = -1

        # Try each action
        for action_offset in [0, 1, 2]:
            test_dir = (direction + (action_offset if action_offset <= 1 else -1)) % 4
            dx, dy = direction_deltas[test_dir]
            next_pos = (pos[0] + dx, pos[1] + dy)

            # Check if valid
            if not (0 <= next_pos[0] < grid_size and
                    0 <= next_pos[1] < grid_size):
                continue

            if next_pos in obstacles:
                continue

            # Count space from this position
            space = self._flood_fill(next_pos, grid_size, obstacles)

            if space > best_space:
                best_space = space
                best_action = action_offset if action_offset <= 1 else 2

        return best_action

    def _seek_food_cautiously(
        self,
        pos: Tuple[int, int],
        food: Tuple[int, int],
        opponent_head: Tuple[int, int],
        direction: int,
        grid_size: int,
        obstacles: set
    ) -> int:
        """Seek food while avoiding head-to-head collisions"""
        direction_deltas = {
            0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)
        }

        # Calculate Manhattan distances
        my_dist = abs(pos[0] - food[0]) + abs(pos[1] - food[1])
        opp_dist = abs(opponent_head[0] - food[0]) + abs(opponent_head[1] - food[1])

        # If opponent is closer, seek space instead
        if opp_dist < my_dist:
            return self._seek_space(pos, direction, grid_size, obstacles)

        # BFS to food (reuse greedy agent logic)
        greedy = GreedyFoodAgent(self.device)
        return greedy._bfs_to_food(pos, food, direction, grid_size, obstacles)


def get_scripted_agent(agent_type: str, device: torch.device = None) -> ScriptedAgent:
    """
    Factory function to create scripted agents.

    Args:
        agent_type: One of 'static', 'random', 'greedy', 'defensive'
        device: PyTorch device

    Returns:
        ScriptedAgent instance
    """
    agents = {
        'static': StaticAgent,
        'random': RandomAgent,
        'greedy': GreedyFoodAgent,
        'defensive': DefensiveAgent
    }

    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from {list(agents.keys())}")

    return agents[agent_type](device)
