"""
Snake Animation Recorder

Records snake gameplay with different agents to MP4 files for presentations.

Supported agents:
- random: Random action selection
- direct: Manhattan-style direct path to food
- greedy: A* shortest path
- hamiltonian: Space-filling cycle

Usage:
    python scripts/visualizer/record_animations.py --output presentation/animations/
    python scripts/visualizer/record_animations.py --agent random --duration 30 --output output.mp4
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import cv2

# Set SDL to use dummy video driver (headless)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import pygame

from core.environment import SnakeEnv
from scripts.baselines.random_agent import RandomAgent
from scripts.baselines.direct_goto import DirectGoToAgent
from scripts.baselines.shortest_path import ShortestPathAgent
from scripts.baselines.hamiltonian import HamiltonianAgent


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
GRAY = (50, 50, 50)
LIGHT_GRAY = (100, 100, 100)


class AnimationRecorder:
    """Records snake gameplay to MP4 files."""

    def __init__(self, grid_size: int = 10, cell_size: int = 50, fps: int = 10):
        """
        Initialize the animation recorder.

        Args:
            grid_size: Size of the game grid
            cell_size: Pixel size of each cell
            fps: Frames per second for the video
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 24)

    def _draw_grid(self):
        """Draw the grid lines."""
        for x in range(0, self.width + 1, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.height), 1)
        for y in range(0, self.height + 1, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)

    def _draw_snake(self, snake):
        """Draw the snake on the screen."""
        for i, (x, y) in enumerate(snake):
            px = x * self.cell_size
            py = y * self.cell_size

            if i == 0:
                # Head - bright green with eyes
                pygame.draw.rect(self.screen, GREEN,
                               (px + 1, py + 1, self.cell_size - 2, self.cell_size - 2))
                # Eyes
                eye_size = self.cell_size // 6
                pygame.draw.circle(self.screen, BLACK,
                                 (px + self.cell_size // 3, py + self.cell_size // 3), eye_size)
                pygame.draw.circle(self.screen, BLACK,
                                 (px + 2 * self.cell_size // 3, py + self.cell_size // 3), eye_size)
            else:
                # Body - darker green with slight gap
                pygame.draw.rect(self.screen, DARK_GREEN,
                               (px + 2, py + 2, self.cell_size - 4, self.cell_size - 4))

    def _draw_food(self, food):
        """Draw the food on the screen."""
        if food is None:
            return
        x, y = food
        px = x * self.cell_size
        py = y * self.cell_size
        center = (px + self.cell_size // 2, py + self.cell_size // 2)
        radius = self.cell_size // 3
        pygame.draw.circle(self.screen, RED, center, radius)

    def _draw_score(self, score, episode):
        """Draw score overlay."""
        text = self.font.render(f"Score: {score}  Episode: {episode}", True, WHITE)
        # Background for text
        text_rect = text.get_rect()
        text_rect.topleft = (5, 5)
        bg_rect = text_rect.inflate(10, 6)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
        self.screen.blit(text, text_rect)

    def _capture_frame(self) -> np.ndarray:
        """Capture current pygame screen as numpy array."""
        # Get the raw pixel data
        frame = pygame.surfarray.array3d(self.screen)
        # Pygame uses (width, height, channels), OpenCV uses (height, width, channels)
        frame = np.transpose(frame, (1, 0, 2))
        # Pygame uses RGB, OpenCV uses BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def record_agent(self, agent, output_path: str, min_duration_seconds: int = 30,
                     agent_name: str = "Agent", max_duration_seconds: int = 120,
                     fps_override: int = None):
        """
        Record gameplay with the given agent.

        Records for at least min_duration_seconds, then continues until the snake dies.
        The video always ends with a death (or max duration for agents that don't die).

        Args:
            agent: Agent instance with get_action(env) method
            output_path: Path to save the MP4 file
            min_duration_seconds: Minimum duration in seconds (continues until death after this)
            agent_name: Name to display (for logging)
            max_duration_seconds: Maximum duration (safety limit for agents that rarely die)
            fps_override: Override FPS for this recording (for faster playback)
        """
        fps = fps_override if fps_override else self.fps
        print(f"Recording {agent_name} (min {min_duration_seconds}s, ends on death, {fps} fps)...")

        # Create environment
        env = SnakeEnv(
            grid_size=self.grid_size,
            action_space_type='absolute',
            state_representation='feature',
            max_steps=10000,  # High limit to allow long runs
            reward_distance=False
        )

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))

        min_frames = min_duration_seconds * fps
        max_frames = max_duration_seconds * fps
        frame_count = 0
        episode = 1
        total_score = 0

        # Reset environment
        obs, info = env.reset(seed=67)

        while True:
            # Get action from agent
            action = agent.get_action(env)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Draw frame
            self.screen.fill(BLACK)
            self._draw_grid()
            self._draw_snake(env.snake)
            self._draw_food(env.food)
            self._draw_score(env.score, episode)
            pygame.display.flip()

            # Capture and write frame
            frame = self._capture_frame()
            out.write(frame)
            frame_count += 1

            if done:
                total_score += env.score
                print(f"  Episode {episode}: Score {env.score}, Steps {env.steps}")

                # If we've reached minimum duration, end on this death
                if frame_count >= min_frames:
                    break

                # Otherwise restart for another episode
                episode += 1
                obs, info = env.reset()

            # Safety limit for agents that rarely die (like Hamiltonian)
            if frame_count >= max_frames:
                print(f"  (Reached max duration of {max_duration_seconds}s)")
                break

            # Handle pygame events (required for headless mode)
            for event in pygame.event.get():
                pass

        out.release()
        actual_duration = frame_count / self.fps
        avg_score = total_score / episode if episode > 0 else 0
        print(f"  Saved to {output_path}")
        print(f"  Duration: {actual_duration:.1f}s, Episodes: {episode}, Avg score: {avg_score:.1f}")

    def record_all(self, output_dir: str, duration_seconds: int = 30):
        """
        Record all 4 agent types.

        Args:
            output_dir: Directory to save MP4 files
            duration_seconds: Duration of each video in seconds
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        agents = [
            ("random", RandomAgent(action_space_type='absolute'), "Random Agent", self.fps),
            ("direct_goto", DirectGoToAgent(action_space_type='absolute'), "Direct Go-To Agent", self.fps),
            ("greedy_astar", ShortestPathAgent(action_space_type='absolute'), "Greedy (A*) Agent", self.fps),
            ("hamiltonian", HamiltonianAgent(action_space_type='absolute', grid_size=self.grid_size),
             "Hamiltonian Agent", 30),  # Faster FPS for Hamiltonian to show full cycle
        ]

        for filename, agent, name, fps in agents:
            output_path = str(Path(output_dir) / f"{filename}.mp4")
            self.record_agent(agent, output_path, duration_seconds, name, fps_override=fps)
            print()

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Record Snake agent animations to MP4')
    parser.add_argument('--agent', type=str, choices=['random', 'direct', 'greedy', 'hamiltonian', 'all'],
                        default='all', help='Agent type to record (default: all)')
    parser.add_argument('--output', type=str, default='presentation/animations/',
                        help='Output path (directory for all, file for single agent)')
    parser.add_argument('--duration', type=int, default=30,
                        help='Duration in seconds (default: 30)')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='Grid size (default: 10)')
    parser.add_argument('--cell-size', type=int, default=50,
                        help='Cell size in pixels (default: 50)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second (default: 10)')

    args = parser.parse_args()

    recorder = AnimationRecorder(
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        fps=args.fps
    )

    try:
        if args.agent == 'all':
            recorder.record_all(args.output, args.duration)
        else:
            # Single agent
            if args.agent == 'random':
                agent = RandomAgent(action_space_type='absolute')
                name = "Random Agent"
            elif args.agent == 'direct':
                agent = DirectGoToAgent(action_space_type='absolute')
                name = "Direct Go-To Agent"
            elif args.agent == 'greedy':
                agent = ShortestPathAgent(action_space_type='absolute')
                name = "Greedy (A*) Agent"
            else:  # hamiltonian
                agent = HamiltonianAgent(action_space_type='absolute', grid_size=args.grid_size)
                name = "Hamiltonian Agent"

            # If output looks like a directory, add filename
            if not args.output.endswith('.mp4'):
                Path(args.output).mkdir(parents=True, exist_ok=True)
                output_path = str(Path(args.output) / f"{args.agent}.mp4")
            else:
                output_path = args.output

            recorder.record_agent(agent, output_path, args.duration, name)
    finally:
        recorder.cleanup()

    print("Done!")


if __name__ == '__main__':
    main()
