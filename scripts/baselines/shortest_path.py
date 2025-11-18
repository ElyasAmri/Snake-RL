"""
Shortest Path Agent using A* Algorithm

Deterministic agent that finds optimal path to food while avoiding obstacles
"""

import numpy as np
from typing import Tuple, List, Optional
from heapq import heappush, heappop


class ShortestPathAgent:
    """
    A* pathfinding agent for Snake

    Finds shortest path to food, considering:
    - Wall boundaries
    - Snake body as obstacles
    - Tie-breaking for equal-cost paths
    """

    def __init__(self, action_space_type: str = 'absolute'):
        """
        Initialize shortest path agent

        Args:
            action_space_type: 'absolute' or 'relative'
        """
        self.action_space_type = action_space_type

    def get_action(self, env) -> int:
        """
        Get action using A* pathfinding

        Args:
            env: SnakeEnv instance

        Returns:
            Action to take
        """
        # Extract game state
        snake = env.snake
        food = env.food
        grid_size = env.grid_size
        current_direction = env.direction

        # Find path using A*
        path = self._astar(snake, food, grid_size)

        if path and len(path) > 1:
            # Get next position in path
            next_pos = path[1]
            head = snake[0]

            # Determine direction to move
            dx = next_pos[0] - head[0]
            dy = next_pos[1] - head[1]

            if self.action_space_type == 'absolute':
                # Convert delta to absolute direction
                return self._delta_to_action_absolute(dx, dy)
            else:
                # Convert delta to relative action
                return self._delta_to_action_relative(dx, dy, current_direction)
        else:
            # No path found, return safe action
            return self._get_safe_action(env)

    def _astar(
        self,
        snake: List[Tuple[int, int]],
        food: Tuple[int, int],
        grid_size: int
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm

        Args:
            snake: List of snake positions
            food: Food position
            grid_size: Grid size

        Returns:
            List of positions from head to food, or None if no path
        """
        start = snake[0]
        goal = food

        # Priority queue: (f_score, g_score, position, path)
        heap = [(0, 0, start, [start])]
        visited = set()

        while heap:
            f_score, g_score, current, path = heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            # Goal reached
            if current == goal:
                return path

            # Explore neighbors
            for neighbor in self._get_neighbors(current, grid_size):
                if neighbor in visited:
                    continue

                # Check if neighbor is obstacle (snake body, but not tail)
                # Tail will move, so we can occupy it
                if neighbor in snake[:-1]:
                    continue

                new_g = g_score + 1
                h = self._manhattan_distance(neighbor, goal)
                new_f = new_g + h

                new_path = path + [neighbor]
                heappush(heap, (new_f, new_g, neighbor, new_path))

        # No path found
        return None

    def _get_neighbors(
        self,
        pos: Tuple[int, int],
        grid_size: int
    ) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []

        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # UP, RIGHT, DOWN, LEFT
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbors.append((nx, ny))

        return neighbors

    def _manhattan_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> int:
        """Calculate Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _delta_to_action_absolute(self, dx: int, dy: int) -> int:
        """
        Convert movement delta to absolute action

        Returns:
            0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        """
        if dy == -1:
            return 0  # UP
        elif dx == 1:
            return 1  # RIGHT
        elif dy == 1:
            return 2  # DOWN
        else:  # dx == -1
            return 3  # LEFT

    def _delta_to_action_relative(
        self,
        dx: int,
        dy: int,
        current_direction: int
    ) -> int:
        """
        Convert movement delta to relative action

        Args:
            dx, dy: Movement delta
            current_direction: Current direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)

        Returns:
            0=STRAIGHT, 1=LEFT, 2=RIGHT
        """
        # Determine target direction
        if dy == -1:
            target_direction = 0  # UP
        elif dx == 1:
            target_direction = 1  # RIGHT
        elif dy == 1:
            target_direction = 2  # DOWN
        else:  # dx == -1
            target_direction = 3  # LEFT

        # Calculate relative turn
        turn = (target_direction - current_direction) % 4

        if turn == 0:
            return 0  # STRAIGHT
        elif turn == 3 or turn == -1:
            return 1  # LEFT
        else:  # turn == 1
            return 2  # RIGHT

    def _get_safe_action(self, env) -> int:
        """
        Get safe action when no path to food exists

        Try to find any safe move, or straight if all blocked
        """
        head = env.snake[0]
        grid_size = env.grid_size
        current_direction = env.direction

        # Try each direction
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT

        if self.action_space_type == 'absolute':
            for action, (dx, dy) in enumerate(directions):
                next_pos = (head[0] + dx, head[1] + dy)
                if self._is_safe(next_pos, env.snake, grid_size):
                    return action
        else:
            # Relative: try straight, left, right
            for action in [0, 1, 2]:
                if action == 0:
                    check_dir = current_direction
                elif action == 1:
                    check_dir = (current_direction - 1) % 4
                else:
                    check_dir = (current_direction + 1) % 4

                dx, dy = directions[check_dir]
                next_pos = (head[0] + dx, head[1] + dy)
                if self._is_safe(next_pos, env.snake, grid_size):
                    return action

        # All blocked, return straight (will likely die)
        return 0

    def _is_safe(
        self,
        pos: Tuple[int, int],
        snake: List[Tuple[int, int]],
        grid_size: int
    ) -> bool:
        """Check if position is safe (not wall or snake body)"""
        x, y = pos
        return (
            0 <= x < grid_size and
            0 <= y < grid_size and
            pos not in snake
        )
