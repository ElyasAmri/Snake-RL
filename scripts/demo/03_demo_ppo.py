"""
Demo: PPO Agent
Slide: PPO Agent Demo
Narrative: "PPO agent trained with basic features"

Shows a PPO (Proximal Policy Optimization) agent playing Snake.
Demonstrates Actor-Critic architecture with policy gradient methods.

Run with: ./venv/Scripts/python.exe scripts/demo/demo_ppo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pygame
import torch
import torch.nn.functional as F
import random
import numpy as np

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import PPO_Actor_MLP, PPO_Critic_MLP

# Demo parameters
SEED = 67
FPS = 10
GRID_SIZE = 10
CELL_SIZE = 50
WEIGHTS_PATH = "results/weights/demo_ppo.pt"

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
YELLOW = (255, 255, 0)
PURPLE = (180, 0, 255)
GRAY = (128, 128, 128)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PPODemo:
    """Demo showing trained PPO agent."""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Pygame setup
        pygame.init()
        self.width = GRID_SIZE * CELL_SIZE
        self.height = GRID_SIZE * CELL_SIZE + 100
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Demo: PPO Agent (Actor-Critic)')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Environment
        self.env = VectorizedSnakeEnv(
            num_envs=1,
            grid_size=GRID_SIZE,
            action_space_type='relative',
            state_representation='feature',
            max_steps=1000,
            device=self.device
        )

        # Stats
        self.episode = 0
        self.total_score = 0
        self.current_score = 0
        self.current_steps = 0

    def draw_grid(self):
        """Draw grid lines."""
        for x in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.width), 1)
        for y in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)

    def draw_snake(self, snake, length):
        """Draw snake."""
        for i in range(length):
            x = int(snake[i, 0].item()) * CELL_SIZE
            y = int(snake[i, 1].item()) * CELL_SIZE

            if i == 0:
                # Head
                pygame.draw.rect(self.screen, GREEN, (x, y, CELL_SIZE, CELL_SIZE))
                # Eyes
                eye_size = CELL_SIZE // 6
                pygame.draw.circle(self.screen, BLACK,
                                   (x + CELL_SIZE//3, y + CELL_SIZE//3), eye_size)
                pygame.draw.circle(self.screen, BLACK,
                                   (x + 2*CELL_SIZE//3, y + CELL_SIZE//3), eye_size)
            else:
                # Body
                pygame.draw.rect(self.screen, DARK_GREEN,
                                 (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))

    def draw_food(self, food):
        """Draw food."""
        x = int(food[0].item()) * CELL_SIZE
        y = int(food[1].item()) * CELL_SIZE
        pygame.draw.circle(self.screen, RED,
                           (x + CELL_SIZE//2, y + CELL_SIZE//2),
                           CELL_SIZE//3)

    def draw_stats(self, probs, value):
        """Draw stats panel."""
        stats_y = self.width
        pygame.draw.rect(self.screen, BLACK, (0, stats_y, self.width, 100))

        # Title (purple for PPO)
        title = self.font_large.render("PPO Agent (Actor-Critic)", True, PURPLE)
        self.screen.blit(title, (10, stats_y + 5))

        # Stats with policy probabilities
        action_names = ['Str', 'L', 'R']
        probs_str = " | ".join([f"{action_names[i]}:{probs[i]:.0%}" for i in range(3)])
        avg_score = self.total_score / max(1, self.episode)

        lines = [
            f"Episode: {self.episode}  Score: {self.current_score}  Steps: {self.current_steps}",
            f"Avg Score: {avg_score:.1f}  Policy: {probs_str}  V(s): {value:.2f}",
        ]

        for i, line in enumerate(lines):
            surface = self.font.render(line, True, WHITE)
            self.screen.blit(surface, (10, stats_y + 40 + i * 20))

    def load_networks(self):
        """Load pre-trained PPO networks."""
        weights_path = Path(WEIGHTS_PATH)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}. Train first with train_ppo.py")

        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        actor = PPO_Actor_MLP(input_dim=10, output_dim=3, hidden_dims=(128, 128)).to(self.device)
        critic = PPO_Critic_MLP(input_dim=10, hidden_dims=(128, 128)).to(self.device)

        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])

        actor.eval()
        critic.eval()

        print("Weights loaded successfully!")
        return actor, critic

    def run(self, actor=None, critic=None, num_episodes=100):
        """Run the demo."""
        # Load if no networks provided
        if actor is None or critic is None:
            actor, critic = self.load_networks()

        actor.to(self.device)
        critic.to(self.device)
        actor.eval()
        critic.eval()

        print(f"\nRunning demo for {num_episodes} episodes...")
        print("Press ESC or close window to exit")

        state = self.env.reset()
        running = True

        with torch.no_grad():
            while running and self.episode < num_episodes:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False

                # Get action from policy
                logits = actor(state)
                probs = F.softmax(logits, dim=-1)
                value = critic(state)

                # Sample from policy (or use argmax for deterministic)
                action = probs.argmax(dim=1)  # Greedy for demo

                next_state, reward, done, info = self.env.step(action)

                self.current_score = int(info['scores'][0].item())
                self.current_steps += 1

                # Draw
                self.screen.fill(BLACK)
                self.draw_grid()
                self.draw_snake(self.env.snakes[0], self.env.snake_lengths[0])
                self.draw_food(self.env.foods[0])
                self.draw_stats(probs[0].cpu().numpy(), value[0].item())

                pygame.display.flip()
                self.clock.tick(FPS)

                if done[0]:
                    self.episode += 1
                    self.total_score += self.current_score
                    print(f"Episode {self.episode}: Score {self.current_score}, Steps {self.current_steps}")
                    self.current_steps = 0
                    state = self.env.reset()
                else:
                    state = next_state

        pygame.quit()
        print(f"\nFinished {self.episode} episodes. Avg Score: {self.total_score / max(1, self.episode):.2f}")


def main():
    set_seed(SEED)
    demo = PPODemo()
    demo.run(num_episodes=100)


if __name__ == '__main__':
    main()
