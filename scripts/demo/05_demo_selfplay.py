"""
Demo: Naive Self-Play Results
Slide: Self-Play Results
Narrative: "Both agents struggle to learn meaningful behavior"

Shows two-snake competition using EARLY/POORLY trained checkpoint.
Demonstrates the problem with naive self-play:
- Non-stationary environment (both agents changing)
- Both start untrained
- Unstable learning dynamics

Run with: ./venv/Scripts/python.exe scripts/demo/demo_selfplay.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pygame
import torch
import random
import numpy as np

from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from core.networks import DQN_MLP, PPO_Actor_MLP

# Demo parameters
SEED = 67
FPS = 10
GRID_SIZE = 10
CELL_SIZE = 50
TARGET_FOOD = 10

# Early checkpoint (poorly trained - demonstrates self-play struggles)
WEIGHTS_PATH = "archive/experiments/checkpoints/two_snake/dqn/agent1_256n_ep10.pt"

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
BLUE = (0, 100, 255)
LIGHT_BLUE = (100, 150, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)


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


class SelfPlayDemo:
    """Demo showing naive self-play struggles."""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Pygame setup
        pygame.init()
        self.width = GRID_SIZE * CELL_SIZE
        self.height = GRID_SIZE * CELL_SIZE + 150
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Demo: Naive Self-Play (Both Struggle)')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 18)

        # Environment
        self.env = VectorizedTwoSnakeEnv(
            num_envs=1,
            grid_size=GRID_SIZE,
            target_food=TARGET_FOOD,
            max_steps=1000,
            device=self.device
        )

        # Stats
        self.round = 0
        self.snake1_wins = 0
        self.snake2_wins = 0
        self.ties = 0

    def draw_grid(self):
        """Draw grid lines."""
        for x in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.width), 1)
        for y in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)

    def draw_snake(self, snake, length, color_head, color_body):
        """Draw snake with specified colors."""
        for i in range(length):
            x = int(snake[i, 0].item()) * CELL_SIZE
            y = int(snake[i, 1].item()) * CELL_SIZE

            if i == 0:
                pygame.draw.rect(self.screen, color_head, (x, y, CELL_SIZE, CELL_SIZE))
                eye_size = CELL_SIZE // 6
                pygame.draw.circle(self.screen, BLACK,
                                   (x + CELL_SIZE//3, y + CELL_SIZE//3), eye_size)
                pygame.draw.circle(self.screen, BLACK,
                                   (x + 2*CELL_SIZE//3, y + CELL_SIZE//3), eye_size)
            else:
                pygame.draw.rect(self.screen, color_body,
                                 (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))

    def draw_food(self, food):
        """Draw food."""
        x = int(food[0].item()) * CELL_SIZE
        y = int(food[1].item()) * CELL_SIZE
        pygame.draw.circle(self.screen, RED,
                           (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3)

    def draw_stats(self):
        """Draw stats panel with warning about naive self-play."""
        stats_y = self.width + 10
        pygame.draw.rect(self.screen, DARK_GRAY, (0, self.width, self.width, 150))

        # Title with warning color
        title = self.font_large.render("Naive Self-Play (Both Struggle)", True, ORANGE)
        self.screen.blit(title, (10, stats_y))

        # Warning indicator
        warning = self.font_small.render("Early training - erratic behavior!", True, ORANGE)
        self.screen.blit(warning, (10, stats_y + 30))

        # Round and scores
        score1 = self.env.food_counts1[0].item()
        score2 = self.env.food_counts2[0].item()

        round_text = self.font.render(f"Round: {self.round}", True, WHITE)
        self.screen.blit(round_text, (self.width - 100, stats_y))

        # Snake 1 (Green)
        pygame.draw.rect(self.screen, GREEN, (10, stats_y + 55, 20, 20))
        s1_text = self.font.render(f"Snake 1: {score1}/{TARGET_FOOD}  Wins: {self.snake1_wins}", True, WHITE)
        self.screen.blit(s1_text, (35, stats_y + 55))

        # Snake 2 (Blue)
        pygame.draw.rect(self.screen, BLUE, (10, stats_y + 85, 20, 20))
        s2_text = self.font.render(f"Snake 2: {score2}/{TARGET_FOOD}  Wins: {self.snake2_wins}", True, WHITE)
        self.screen.blit(s2_text, (35, stats_y + 85))

        # Ties
        if self.ties > 0:
            ties_text = self.font_small.render(f"Ties: {self.ties}", True, YELLOW)
            self.screen.blit(ties_text, (self.width - 80, stats_y + 30))

        # Winner status
        winner = self.env.round_winners[0].item()
        if winner == 1:
            self.screen.blit(self.font.render("Snake 1 Won!", True, GREEN), (self.width - 120, stats_y + 55))
        elif winner == 2:
            self.screen.blit(self.font.render("Snake 2 Won!", True, BLUE), (self.width - 120, stats_y + 85))
        elif winner == 3:
            self.screen.blit(self.font.render("TIE!", True, YELLOW), (self.width - 80, stats_y + 70))

    def load_early_checkpoint(self):
        """Load early/poorly trained checkpoint to show self-play struggles."""
        weights_path = Path(WEIGHTS_PATH)

        if weights_path.exists():
            print(f"Loading early checkpoint: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

            # Detect type and create network
            if 'actor' in checkpoint:
                network = PPO_Actor_MLP(input_dim=35, output_dim=3, hidden_dims=(256, 256)).to(self.device)
                network.load_state_dict(checkpoint['actor'])
            elif 'policy_net' in checkpoint:
                network = DQN_MLP(input_dim=35, output_dim=3, hidden_dims=(256, 256)).to(self.device)
                network.load_state_dict(checkpoint['policy_net'])
            else:
                print(f"Unknown checkpoint format, using random agent")
                return None

            network.eval()
            return network
        else:
            print(f"Early checkpoint not found at {weights_path}")
            print("Using random actions to simulate untrained self-play")
            return None

    def run(self, num_rounds=50):
        """Run the naive self-play demo."""
        print("=" * 60)
        print("DEMO: Naive Self-Play Results")
        print("=" * 60)
        print("Both agents trained via naive self-play (from scratch together).")
        print("Watch how they struggle to learn meaningful behavior.")
        print("=" * 60)

        # Try to load early checkpoint, otherwise use random
        network = self.load_early_checkpoint()

        running = True

        for round_num in range(num_rounds):
            if not running:
                break

            self.round = round_num + 1
            obs1, obs2 = self.env.reset()
            done = False

            while not done and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                        break

                # Get actions (from network or random)
                if network is not None:
                    with torch.no_grad():
                        output1 = network(obs1)
                        output2 = network(obs2)
                        actions1 = output1.argmax(dim=1)
                        actions2 = output2.argmax(dim=1)
                else:
                    # Random actions for untrained demo
                    actions1 = torch.randint(0, 3, (1,), device=self.device)
                    actions2 = torch.randint(0, 3, (1,), device=self.device)

                # Step
                obs1, obs2, r1, r2, dones, info = self.env.step(actions1, actions2)
                done = dones[0].item()

                # Draw
                self.screen.fill(BLACK)
                self.draw_grid()
                self.draw_food(self.env.foods[0])

                # Draw snakes (snake 2 first so snake 1 appears on top)
                length2 = self.env.lengths2[0].item()
                if self.env.alive2[0].item():
                    self.draw_snake(self.env.snakes2[0], length2, BLUE, LIGHT_BLUE)

                length1 = self.env.lengths1[0].item()
                if self.env.alive1[0].item():
                    self.draw_snake(self.env.snakes1[0], length1, GREEN, DARK_GREEN)

                self.draw_stats()
                pygame.display.flip()
                self.clock.tick(FPS)

            # Track winner
            winner = self.env.round_winners[0].item()
            if winner == 1:
                self.snake1_wins += 1
            elif winner == 2:
                self.snake2_wins += 1
            elif winner == 3:
                self.ties += 1

            print(f"Round {self.round}: Winner = {'Snake 1' if winner == 1 else 'Snake 2' if winner == 2 else 'Tie'}")

            # Brief pause to show result
            pygame.time.wait(500)

        pygame.quit()
        print("\n" + "=" * 60)
        print("RESULTS: Naive Self-Play")
        print("=" * 60)
        print(f"Snake 1 Wins: {self.snake1_wins}")
        print(f"Snake 2 Wins: {self.snake2_wins}")
        print(f"Ties: {self.ties}")
        print("\nNotice: Results are often chaotic with naive self-play!")


def main():
    set_seed(SEED)
    demo = SelfPlayDemo()
    demo.run(num_rounds=50)


if __name__ == '__main__':
    main()
