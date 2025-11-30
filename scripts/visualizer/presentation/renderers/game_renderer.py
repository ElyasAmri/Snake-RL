"""
Game Renderer

Pygame-based game visualization for presentation videos.
Renders snake, food, grid, and score overlays.
"""

import os
import numpy as np
from typing import List, Tuple, Optional

# Set SDL to use dummy video driver (headless)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
DARK_GREEN = (30, 180, 30)
YELLOW = (255, 255, 50)
GRAY = (60, 60, 60)
LIGHT_GRAY = (120, 120, 120)
BLUE = (50, 150, 255)


class GameRenderer:
    """
    Renders Snake game state to pygame surface.

    Produces high-quality 1080p game visualization.
    """

    def __init__(
        self,
        grid_size: int = 10,
        game_area_size: int = 1080
    ):
        """
        Initialize game renderer.

        Args:
            grid_size: Number of cells in grid (default 10)
            game_area_size: Pixel size of game area (default 1080 for HD)
        """
        self.grid_size = grid_size
        self.game_area_size = game_area_size
        self.cell_size = game_area_size // grid_size  # 108px for 10x10 at 1080p

        # Initialize pygame
        pygame.init()
        self.surface = pygame.display.set_mode((game_area_size, game_area_size))

        # Fonts - scaled for 1080p
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)

    def render(
        self,
        snake: List[Tuple[int, int]],
        food: Tuple[int, int],
        direction: int,
        score: int,
        episode: int,
        steps: int = 0,
        show_grid: bool = True
    ) -> np.ndarray:
        """
        Render game state to numpy array.

        Args:
            snake: List of (x, y) positions, head first
            food: (x, y) position of food
            direction: Current direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            score: Current score
            episode: Current episode number
            steps: Current step count
            show_grid: Whether to draw grid lines

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        # Clear surface
        self.surface.fill(BLACK)

        # Draw grid
        if show_grid:
            self._draw_grid()

        # Draw snake
        self._draw_snake(snake, direction)

        # Draw food
        self._draw_food(food)

        # Draw score overlay
        self._draw_score_overlay(score, episode, steps)

        # Convert to numpy array
        return self._surface_to_array()

    def render_with_path(
        self,
        snake: List[Tuple[int, int]],
        food: Tuple[int, int],
        direction: int,
        score: int,
        episode: int,
        path: Optional[List[Tuple[int, int]]] = None,
        steps: int = 0
    ) -> np.ndarray:
        """
        Render game state with A* path visualization.

        Args:
            snake: List of (x, y) positions
            food: (x, y) position of food
            direction: Current direction
            score: Current score
            episode: Current episode number
            path: Optional list of (x, y) positions showing planned path
            steps: Current step count

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        # Clear surface
        self.surface.fill(BLACK)

        # Draw grid
        self._draw_grid()

        # Draw path first (so it appears behind snake)
        if path:
            self._draw_path(path)

        # Draw snake
        self._draw_snake(snake, direction)

        # Draw food
        self._draw_food(food)

        # Draw score overlay
        self._draw_score_overlay(score, episode, steps)

        return self._surface_to_array()

    def _draw_grid(self):
        """Draw grid lines."""
        for x in range(0, self.game_area_size + 1, self.cell_size):
            pygame.draw.line(self.surface, GRAY, (x, 0), (x, self.game_area_size), 2)
        for y in range(0, self.game_area_size + 1, self.cell_size):
            pygame.draw.line(self.surface, GRAY, (0, y), (self.game_area_size, y), 2)

    def _draw_snake(self, snake: List[Tuple[int, int]], direction: int):
        """Draw snake with head and body segments."""
        for i, (x, y) in enumerate(snake):
            px = x * self.cell_size
            py = y * self.cell_size

            if i == 0:
                # Head - bright green with eyes
                pygame.draw.rect(
                    self.surface, GREEN,
                    (px + 2, py + 2, self.cell_size - 4, self.cell_size - 4),
                    border_radius=8
                )

                # Draw eyes based on direction
                eye_size = self.cell_size // 8
                eye_offset = self.cell_size // 4

                if direction == 0:  # UP
                    left_eye = (px + eye_offset, py + eye_offset)
                    right_eye = (px + self.cell_size - eye_offset, py + eye_offset)
                elif direction == 1:  # RIGHT
                    left_eye = (px + self.cell_size - eye_offset, py + eye_offset)
                    right_eye = (px + self.cell_size - eye_offset, py + self.cell_size - eye_offset)
                elif direction == 2:  # DOWN
                    left_eye = (px + self.cell_size - eye_offset, py + self.cell_size - eye_offset)
                    right_eye = (px + eye_offset, py + self.cell_size - eye_offset)
                else:  # LEFT
                    left_eye = (px + eye_offset, py + self.cell_size - eye_offset)
                    right_eye = (px + eye_offset, py + eye_offset)

                pygame.draw.circle(self.surface, BLACK, left_eye, eye_size)
                pygame.draw.circle(self.surface, BLACK, right_eye, eye_size)
            else:
                # Body - darker green with rounded corners
                pygame.draw.rect(
                    self.surface, DARK_GREEN,
                    (px + 4, py + 4, self.cell_size - 8, self.cell_size - 8),
                    border_radius=6
                )

    def _draw_food(self, food: Tuple[int, int]):
        """Draw food as red circle."""
        if food is None:
            return

        x, y = food
        px = x * self.cell_size
        py = y * self.cell_size

        center = (px + self.cell_size // 2, py + self.cell_size // 2)
        radius = self.cell_size // 3

        # Draw food with glow effect
        pygame.draw.circle(self.surface, (150, 30, 30), center, radius + 4)  # Outer glow
        pygame.draw.circle(self.surface, RED, center, radius)

    def _draw_path(self, path: List[Tuple[int, int]]):
        """Draw A* path as dotted line."""
        if not path or len(path) < 2:
            return

        for i, (x, y) in enumerate(path):
            px = x * self.cell_size + self.cell_size // 2
            py = y * self.cell_size + self.cell_size // 2

            # Draw path dots
            alpha = 1.0 - (i / len(path)) * 0.5  # Fade towards end
            color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
            pygame.draw.circle(self.surface, color, (px, py), self.cell_size // 6)

    def _draw_score_overlay(self, score: int, episode: int, steps: int):
        """Draw score and episode info overlay."""
        # Semi-transparent background
        overlay_height = 80
        overlay_surface = pygame.Surface((self.game_area_size, overlay_height))
        overlay_surface.set_alpha(180)
        overlay_surface.fill(BLACK)
        self.surface.blit(overlay_surface, (0, self.game_area_size - overlay_height))

        # Score text
        score_text = self.font_large.render(f"Score: {score}", True, WHITE)
        self.surface.blit(score_text, (20, self.game_area_size - overlay_height + 15))

        # Episode and steps
        info_text = self.font_medium.render(f"Episode: {episode}  Steps: {steps}", True, LIGHT_GRAY)
        self.surface.blit(info_text, (self.game_area_size - 400, self.game_area_size - overlay_height + 25))

    def _surface_to_array(self) -> np.ndarray:
        """Convert pygame surface to numpy RGB array."""
        # Get raw pixel data (pygame uses RGB)
        frame = pygame.surfarray.array3d(self.surface)
        # Transpose from (W, H, C) to (H, W, C)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def draw_cell_value(
        self,
        x: int,
        y: int,
        value: float,
        color: Tuple[int, int, int] = WHITE
    ):
        """
        Draw numeric value on a grid cell.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            value: Value to display
            color: Text color
        """
        px = x * self.cell_size + self.cell_size // 2
        py = y * self.cell_size + self.cell_size // 2

        # Format value
        if abs(value) < 0.01:
            text = "0"
        elif abs(value) < 1:
            text = f"{value:.2f}"
        else:
            text = f"{value:.1f}"

        text_surface = self.font_small.render(text, True, color)
        text_rect = text_surface.get_rect(center=(px, py))
        self.surface.blit(text_surface, text_rect)

    def draw_direction_arrow(
        self,
        x: int,
        y: int,
        direction: int,
        color: Tuple[int, int, int] = YELLOW,
        scale: float = 0.6
    ):
        """
        Draw direction arrow on a grid cell.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            direction: Direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            color: Arrow color
            scale: Arrow size relative to cell
        """
        px = x * self.cell_size + self.cell_size // 2
        py = y * self.cell_size + self.cell_size // 2
        size = int(self.cell_size * scale // 2)

        # Arrow points based on direction
        if direction == 0:  # UP
            points = [(px, py - size), (px - size//2, py + size//2), (px + size//2, py + size//2)]
        elif direction == 1:  # RIGHT
            points = [(px + size, py), (px - size//2, py - size//2), (px - size//2, py + size//2)]
        elif direction == 2:  # DOWN
            points = [(px, py + size), (px - size//2, py - size//2), (px + size//2, py - size//2)]
        else:  # LEFT
            points = [(px - size, py), (px + size//2, py - size//2), (px + size//2, py + size//2)]

        pygame.draw.polygon(self.surface, color, points)

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()
