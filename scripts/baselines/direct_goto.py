"""
Direct Go-To Agent Baseline

Agent that moves directly toward food using Manhattan-style movement:
first moves horizontally until aligned, then moves vertically.
Does not avoid obstacles - will die on self-collision.
"""

from typing import Tuple


class DirectGoToAgent:
    """
    Direct path agent for Snake using Manhattan-style movement.

    Moves horizontally first until x-aligned with food,
    then moves vertically to reach food.
    Simple deterministic behavior, no obstacle avoidance.
    """

    def __init__(self, action_space_type: str = 'absolute'):
        """
        Initialize direct go-to agent.

        Args:
            action_space_type: 'absolute' (4 actions) or 'relative' (3 actions)
        """
        self.action_space_type = action_space_type

    def get_action(self, env) -> int:
        """
        Get action to move directly toward food.

        First moves horizontally until aligned with food,
        then moves vertically.

        Args:
            env: SnakeEnv instance

        Returns:
            Action to take
        """
        head = env.snake[0]
        food = env.food
        current_direction = env.direction

        # Calculate delta to food
        dx = food[0] - head[0]
        dy = food[1] - head[1]

        # Determine target direction (horizontal first, then vertical)
        if dx != 0:
            # Move horizontally first
            target_direction = 1 if dx > 0 else 3  # RIGHT or LEFT
        elif dy != 0:
            # Then move vertically
            target_direction = 2 if dy > 0 else 0  # DOWN or UP
        else:
            # Already at food (shouldn't happen), go straight
            target_direction = current_direction

        if self.action_space_type == 'absolute':
            # Check if target would be a 180-degree turn (invalid)
            if self._is_opposite(target_direction, current_direction):
                # Can't go backwards, try perpendicular direction
                if dx != 0:
                    # Was trying horizontal, try vertical instead
                    target_direction = 2 if dy > 0 else 0 if dy < 0 else current_direction
                else:
                    # Was trying vertical, try horizontal instead
                    target_direction = 1 if dx > 0 else 3 if dx < 0 else current_direction
            return target_direction
        else:
            # Relative action space
            return self._direction_to_relative_action(target_direction, current_direction)

    def _is_opposite(self, dir1: int, dir2: int) -> bool:
        """Check if two directions are opposite (180 degrees apart)."""
        return (dir1 - dir2) % 4 == 2

    def _direction_to_relative_action(self, target_direction: int, current_direction: int) -> int:
        """
        Convert target direction to relative action.

        Args:
            target_direction: Target absolute direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            current_direction: Current absolute direction

        Returns:
            Relative action (0=STRAIGHT, 1=LEFT, 2=RIGHT)
        """
        # Calculate relative turn
        turn = (target_direction - current_direction) % 4

        if turn == 0:
            return 0  # STRAIGHT
        elif turn == 3:  # -1 mod 4
            return 1  # LEFT
        elif turn == 1:
            return 2  # RIGHT
        else:  # turn == 2, need to turn around
            # Can't turn 180, so turn right (or left, arbitrary choice)
            return 2  # RIGHT
