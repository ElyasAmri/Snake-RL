"""
Hamiltonian Cycle Agent Baseline

Agent that follows a pre-computed Hamiltonian cycle through the grid.
Guarantees visiting all cells and can theoretically achieve maximum score,
but is very slow and inefficient.
"""

from typing import List, Tuple


class HamiltonianAgent:
    """
    Hamiltonian Cycle agent that follows a pre-computed cycle.

    A Hamiltonian cycle visits every cell exactly once before returning
    to the start. Following this cycle guarantees:
    - No self-collision
    - Eventually eating all food
    - Ability to fill the entire grid

    Limitations:
    - Very slow (takes grid_size^2 steps per food on average)
    - Not optimal path-wise
    """

    def __init__(self, action_space_type: str = 'absolute', grid_size: int = 10):
        """
        Initialize Hamiltonian cycle agent.

        Args:
            action_space_type: 'absolute' (4 actions) or 'relative' (3 actions)
            grid_size: Size of the grid
        """
        self.action_space_type = action_space_type
        self.grid_size = grid_size

        # Pre-compute Hamiltonian cycle
        self.cycle = self._compute_hamiltonian_cycle(grid_size)

        # Create position to index mapping for fast lookup
        self.pos_to_index = {pos: i for i, pos in enumerate(self.cycle)}

    def _compute_hamiltonian_cycle(self, size: int) -> List[Tuple[int, int]]:
        """
        Compute a Hamiltonian cycle using a zig-zag pattern that forms a proper cycle.

        Pattern for 10x10 grid:
        - Row 0: go right from (0,0) to (9,0)
        - Then down to (9,1)
        - Row 1: go left from (9,1) to (1,1) (stop at x=1, not x=0)
        - Then down to (1,2)
        - Row 2: go right from (1,2) to (9,2)
        - Continue this pattern...
        - Column 0: reserved for the return path going up

        This creates a cycle: right across top, snake down through interior,
        then up the left column back to start.

        Args:
            size: Grid size

        Returns:
            List of (x, y) positions forming a cycle
        """
        cycle = []

        # First row: go all the way right
        for x in range(size):
            cycle.append((x, 0))

        # Snake through the interior (rows 1 to size-1)
        for y in range(1, size):
            if y % 2 == 1:
                # Odd rows: go left from (size-1) to (1)
                for x in range(size - 1, 0, -1):
                    cycle.append((x, y))
            else:
                # Even rows: go right from (1) to (size-1)
                for x in range(1, size):
                    cycle.append((x, y))

        # Return up the left column (from bottom to row 1)
        for y in range(size - 1, 0, -1):
            cycle.append((0, y))

        return cycle

    def get_action(self, env) -> int:
        """
        Get action by following Hamiltonian cycle.

        The agent follows the cycle, but if the next position would require
        a 180-degree turn (which is invalid in Snake), it finds a safe
        alternative direction that won't result in immediate death.

        Args:
            env: SnakeEnv instance

        Returns:
            Action to follow cycle
        """
        head = env.snake[0]
        current_direction = int(env.direction)
        snake_body = set(env.snake[:-1])  # Exclude tail (it will move)

        # Find current position in cycle
        if head in self.pos_to_index:
            current_index = self.pos_to_index[head]
        else:
            # Position not in cycle (shouldn't happen), find safe move
            return self._get_safe_action(env)

        # Get next position in cycle
        next_index = (current_index + 1) % len(self.cycle)
        next_pos = self.cycle[next_index]

        # Calculate direction to next position
        dx = next_pos[0] - head[0]
        dy = next_pos[1] - head[1]

        # Convert delta to target direction
        target_direction = self._delta_to_direction(dx, dy)

        # Check if this would be a 180-degree turn (invalid in Snake)
        if self._is_opposite(target_direction, current_direction):
            # Can't go backwards, find the best safe alternative
            # Priority: turn that gets us closer to cycle flow
            return self._get_safe_action(env)

        if self.action_space_type == 'absolute':
            return target_direction
        else:
            return self._direction_to_relative_action(target_direction, current_direction)

    def _get_safe_action(self, env) -> int:
        """
        Get a safe action when we can't follow the cycle directly.

        Prioritizes directions that are safe (no wall/body collision).

        Args:
            env: SnakeEnv instance

        Returns:
            Safe action
        """
        head = env.snake[0]
        current_direction = int(env.direction)
        snake_body = set(env.snake[:-1])  # Exclude tail

        # Direction deltas: UP, RIGHT, DOWN, LEFT
        direction_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # Try directions in order: straight, right turn, left turn
        # (Never try backwards as it's invalid)
        directions_to_try = [
            current_direction,  # Straight
            (current_direction + 1) % 4,  # Right turn
            (current_direction - 1) % 4,  # Left turn
        ]

        for direction in directions_to_try:
            dx, dy = direction_deltas[direction]
            next_pos = (head[0] + dx, head[1] + dy)

            # Check if safe (within bounds and not hitting body)
            if (0 <= next_pos[0] < self.grid_size and
                0 <= next_pos[1] < self.grid_size and
                next_pos not in snake_body):

                if self.action_space_type == 'absolute':
                    return direction
                else:
                    return self._direction_to_relative_action(direction, current_direction)

        # All directions blocked, just go straight (will die)
        return 0 if self.action_space_type == 'relative' else current_direction

    def _is_opposite(self, dir1: int, dir2: int) -> bool:
        """Check if two directions are opposite (180 degrees apart)."""
        return (dir1 - dir2) % 4 == 2

    def _delta_to_direction(self, dx: int, dy: int) -> int:
        """
        Convert movement delta to absolute direction.

        Args:
            dx, dy: Movement delta

        Returns:
            Direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        if dy == -1:
            return 0  # UP
        elif dx == 1:
            return 1  # RIGHT
        elif dy == 1:
            return 2  # DOWN
        else:  # dx == -1
            return 3  # LEFT

    def _direction_to_relative_action(self, target_direction: int, current_direction: int) -> int:
        """
        Convert target direction to relative action.

        Args:
            target_direction: Target absolute direction
            current_direction: Current absolute direction

        Returns:
            Relative action (0=STRAIGHT, 1=LEFT, 2=RIGHT)
        """
        turn = (target_direction - current_direction) % 4

        if turn == 0:
            return 0  # STRAIGHT
        elif turn == 3:  # -1 mod 4
            return 1  # LEFT
        elif turn == 1:
            return 2  # RIGHT
        else:  # turn == 2
            # Need 180-degree turn, turn right twice (return right for now)
            return 2  # RIGHT
