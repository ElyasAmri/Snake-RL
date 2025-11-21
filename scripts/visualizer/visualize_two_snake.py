"""
Two-Snake Competitive Visualizer

Visualize competitive two-snake RL in multiple modes:
1. Random: Both snakes use random actions
2. Trained: Load saved weights for both networks
3. Scripted: Trained agent vs scripted opponent
4. Training: Watch live curriculum training

Usage:
    # Random vs Random
    python scripts/visualizer/visualize_two_snake.py --mode random --fps 10

    # Trained models
    python scripts/visualizer/visualize_two_snake.py --mode trained \
        --weights1 results/weights/competitive/Stage4_CoEvolution/big_256x256_latest.pt \
        --weights2 results/weights/competitive/Stage4_CoEvolution/small_128x128_latest.pt \
        --episodes 100

    # Trained vs Scripted
    python scripts/visualizer/visualize_two_snake.py --mode scripted \
        --weights1 results/weights/competitive/Stage2_Greedy/big_256x256_latest.pt \
        --opponent greedy
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pygame
import torch
import argparse
import numpy as np

from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from core.networks import DQN_MLP
from scripts.baselines.scripted_opponents import get_scripted_agent

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
LIGHT_BLUE = (100, 150, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)


class TwoSnakeVisualizer:
    """Pygame visualizer for competitive two-snake RL"""

    def __init__(self, grid_size=10, cell_size=50, fps=10, target_food=10, device='cuda'):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.target_food = target_food
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Pygame
        pygame.init()
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size + 150  # Extra space for stats
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Two-Snake Competitive RL')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 18)

        # Environment
        self.env = VectorizedTwoSnakeEnv(
            num_envs=1,
            grid_size=grid_size,
            target_food=target_food,
            max_steps=1000,
            device=self.device
        )

        # Stats
        self.round = 0
        self.snake1_wins = 0
        self.snake2_wins = 0
        self.ties = 0

    def draw_grid(self):
        """Draw grid lines"""
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.width), 1)
        for y in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)

    def draw_snake(self, snake, length, color_head, color_body, thickness=0):
        """Draw snake with specified colors

        Args:
            snake: Snake positions tensor
            length: Snake length
            color_head: Color for head
            color_body: Color for body
            thickness: Border thickness (0 for head, >0 for body)
        """
        for i in range(length):
            x = int(snake[i, 0].item()) * self.cell_size
            y = int(snake[i, 1].item()) * self.cell_size

            if i == 0:
                # Head (filled)
                pygame.draw.rect(self.screen, color_head,
                               (x, y, self.cell_size, self.cell_size))
                # Eyes
                eye_size = self.cell_size // 6
                pygame.draw.circle(self.screen, BLACK,
                                 (x + self.cell_size//3, y + self.cell_size//3), eye_size)
                pygame.draw.circle(self.screen, BLACK,
                                 (x + 2*self.cell_size//3, y + self.cell_size//3), eye_size)
            else:
                # Body (filled with border)
                pygame.draw.rect(self.screen, color_body,
                               (x + 2, y + 2, self.cell_size - 4, self.cell_size - 4))
                if thickness > 0:
                    pygame.draw.rect(self.screen, BLACK,
                                   (x + 2, y + 2, self.cell_size - 4, self.cell_size - 4),
                                   thickness)

    def draw_food(self, food):
        """Draw food"""
        x = int(food[0].item()) * self.cell_size
        y = int(food[1].item()) * self.cell_size

        # Draw apple
        pygame.draw.circle(self.screen, RED,
                          (x + self.cell_size//2, y + self.cell_size//2),
                          self.cell_size//3)
        # Stem
        stem_x = x + self.cell_size//2
        stem_y = y + self.cell_size//4
        pygame.draw.line(self.screen, DARK_GREEN,
                        (stem_x, stem_y), (stem_x, stem_y + self.cell_size//6), 2)

    def draw_stats(self, mode_name, agent1_name="Snake 1", agent2_name="Snake 2"):
        """Draw stats panel"""
        stats_y = self.width + 10

        # Background for stats
        pygame.draw.rect(self.screen, DARK_GRAY,
                        (0, self.width, self.width, 150))

        # Mode title
        title = self.font_large.render(f"Mode: {mode_name}", True, WHITE)
        self.screen.blit(title, (10, stats_y))

        # Round info
        round_text = self.font.render(f"Round: {self.round}", True, WHITE)
        self.screen.blit(round_text, (10, stats_y + 35))

        # Score displays
        score1 = self.env.food_counts1[0].item()
        score2 = self.env.food_counts2[0].item()

        # Snake 1 (Green)
        pygame.draw.rect(self.screen, GREEN, (10, stats_y + 65, 20, 20))
        snake1_text = self.font.render(
            f"{agent1_name}: {score1}/{self.target_food}  (Wins: {self.snake1_wins})",
            True, WHITE
        )
        self.screen.blit(snake1_text, (35, stats_y + 65))

        # Snake 2 (Blue)
        pygame.draw.rect(self.screen, BLUE, (10, stats_y + 95, 20, 20))
        snake2_text = self.font.render(
            f"{agent2_name}: {score2}/{self.target_food}  (Wins: {self.snake2_wins})",
            True, WHITE
        )
        self.screen.blit(snake2_text, (35, stats_y + 95))

        # Winner status
        winner = self.env.round_winners[0].item()
        if winner == 1:
            winner_text = self.font.render("WINNER: Snake 1 (Green)!", True, GREEN)
            self.screen.blit(winner_text, (self.width - 220, stats_y + 65))
        elif winner == 2:
            winner_text = self.font.render("WINNER: Snake 2 (Blue)!", True, BLUE)
            self.screen.blit(winner_text, (self.width - 220, stats_y + 95))
        elif winner == 3:
            tie_text = self.font.render("TIE - Both Lost!", True, YELLOW)
            self.screen.blit(tie_text, (self.width - 180, stats_y + 80))

        # Ties
        if self.ties > 0:
            ties_text = self.font_small.render(f"Ties: {self.ties}", True, YELLOW)
            self.screen.blit(ties_text, (self.width - 80, stats_y + 35))

    def render(self, mode_name="Competitive", agent1_name="Snake 1", agent2_name="Snake 2"):
        """Render current game state"""
        self.screen.fill(BLACK)
        self.draw_grid()

        # Draw food
        self.draw_food(self.env.foods[0])

        # Draw snake 2 (blue) first so snake 1 appears on top
        length2 = self.env.lengths2[0].item()
        if self.env.alive2[0].item():
            self.draw_snake(
                self.env.snakes2[0],
                length2,
                BLUE,
                LIGHT_BLUE,
                thickness=1
            )

        # Draw snake 1 (green)
        length1 = self.env.lengths1[0].item()
        if self.env.alive1[0].item():
            self.draw_snake(
                self.env.snakes1[0],
                length1,
                GREEN,
                DARK_GREEN,
                thickness=2
            )

        # Draw stats
        self.draw_stats(mode_name, agent1_name, agent2_name)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def run_random(self, num_rounds=100):
        """Run random vs random"""
        print(f"Running Random vs Random for {num_rounds} rounds...")

        for round_num in range(num_rounds):
            self.round = round_num + 1
            obs1, obs2 = self.env.reset()
            done = False

            while not done:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Random actions
                actions1 = torch.randint(0, 3, (1,), device=self.device)
                actions2 = torch.randint(0, 3, (1,), device=self.device)

                # Step
                obs1, obs2, r1, r2, dones, info = self.env.step(actions1, actions2)
                done = dones[0].item()

                # Render
                self.render(mode_name="Random vs Random")

            # Track winner
            winner = self.env.round_winners[0].item()
            if winner == 1:
                self.snake1_wins += 1
            elif winner == 2:
                self.snake2_wins += 1
            elif winner == 3:
                self.ties += 1

            # Pause to show winner
            pygame.time.wait(1000)

        print(f"\nFinal Stats:")
        print(f"  Snake 1 Wins: {self.snake1_wins}")
        print(f"  Snake 2 Wins: {self.snake2_wins}")
        print(f"  Ties: {self.ties}")

    def run_trained(self, weights1_path, weights2_path, num_rounds=100):
        """Run trained models"""
        print(f"Loading trained models...")
        print(f"  Agent 1: {weights1_path}")
        print(f"  Agent 2: {weights2_path}")

        # Load networks
        agent1_net = DQN_MLP(input_dim=35, output_dim=3, hidden_dims=(256, 256)).to(self.device)
        agent2_net = DQN_MLP(input_dim=35, output_dim=3, hidden_dims=(128, 128)).to(self.device)

        checkpoint1 = torch.load(weights1_path, map_location=self.device)
        checkpoint2 = torch.load(weights2_path, map_location=self.device)

        agent1_net.load_state_dict(checkpoint1['policy_net'])
        agent2_net.load_state_dict(checkpoint2['policy_net'])

        agent1_net.eval()
        agent2_net.eval()

        print(f"Running {num_rounds} rounds...")

        for round_num in range(num_rounds):
            self.round = round_num + 1
            obs1, obs2 = self.env.reset()
            done = False

            while not done:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Get actions from networks
                with torch.no_grad():
                    q1 = agent1_net(obs1)
                    q2 = agent2_net(obs2)
                    actions1 = q1.argmax(dim=1)
                    actions2 = q2.argmax(dim=1)

                # Step
                obs1, obs2, r1, r2, dones, info = self.env.step(actions1, actions2)
                done = dones[0].item()

                # Render
                self.render(
                    mode_name="Trained Models",
                    agent1_name="Big (256x256)",
                    agent2_name="Small (128x128)"
                )

            # Track winner
            winner = self.env.round_winners[0].item()
            if winner == 1:
                self.snake1_wins += 1
            elif winner == 2:
                self.snake2_wins += 1
            elif winner == 3:
                self.ties += 1

            # Pause to show winner
            pygame.time.wait(1000)

        print(f"\nFinal Stats:")
        print(f"  Big Network Wins: {self.snake1_wins}")
        print(f"  Small Network Wins: {self.snake2_wins}")
        print(f"  Ties: {self.ties}")
        win_rate1 = self.snake1_wins / (self.snake1_wins + self.snake2_wins + self.ties)
        print(f"  Big Network Win Rate: {win_rate1:.2%}")

    def run_scripted(self, weights1_path, opponent_type='greedy', num_rounds=100):
        """Run trained agent vs scripted opponent"""
        print(f"Loading trained agent: {weights1_path}")
        print(f"Opponent: {opponent_type}")

        # Load agent1 network
        agent1_net = DQN_MLP(input_dim=35, output_dim=3, hidden_dims=(256, 256)).to(self.device)
        checkpoint1 = torch.load(weights1_path, map_location=self.device)
        agent1_net.load_state_dict(checkpoint1['policy_net'])
        agent1_net.eval()

        # Load scripted opponent
        try:
            opponent = get_scripted_agent(opponent_type, grid_size=self.grid_size, device=self.device)
        except Exception as e:
            print(f"Error loading scripted agent: {e}")
            return

        print(f"Running {num_rounds} rounds...")

        for round_num in range(num_rounds):
            self.round = round_num + 1
            obs1, obs2 = self.env.reset()
            done = False

            while not done:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Get actions
                with torch.no_grad():
                    q1 = agent1_net(obs1)
                    actions1 = q1.argmax(dim=1)

                actions2 = opponent.select_action(self.env)

                # Step
                obs1, obs2, r1, r2, dones, info = self.env.step(actions1, actions2)
                done = dones[0].item()

                # Render
                self.render(
                    mode_name="Trained vs Scripted",
                    agent1_name="Trained (256x256)",
                    agent2_name=f"Scripted ({opponent_type})"
                )

            # Track winner
            winner = self.env.round_winners[0].item()
            if winner == 1:
                self.snake1_wins += 1
            elif winner == 2:
                self.snake2_wins += 1
            elif winner == 3:
                self.ties += 1

            # Pause to show winner
            pygame.time.wait(1000)

        print(f"\nFinal Stats:")
        print(f"  Trained Agent Wins: {self.snake1_wins}")
        print(f"  Scripted Opponent Wins: {self.snake2_wins}")
        print(f"  Ties: {self.ties}")
        win_rate1 = self.snake1_wins / (self.snake1_wins + self.snake2_wins + self.ties)
        print(f"  Trained Agent Win Rate: {win_rate1:.2%}")


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description='Two-Snake Competitive Visualizer')
    parser.add_argument('--mode', type=str, default='random',
                       choices=['random', 'trained', 'scripted'],
                       help='Visualization mode')
    parser.add_argument('--weights1', type=str, default=None,
                       help='Path to agent1 (big) weights')
    parser.add_argument('--weights2', type=str, default=None,
                       help='Path to agent2 (small) weights')
    parser.add_argument('--opponent', type=str, default='greedy',
                       choices=['static', 'random', 'greedy', 'defensive'],
                       help='Scripted opponent type')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of rounds to run')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size')
    parser.add_argument('--target-food', type=int, default=10,
                       help='Food needed to win')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Create visualizer
    visualizer = TwoSnakeVisualizer(
        grid_size=args.grid_size,
        cell_size=50,
        fps=args.fps,
        target_food=args.target_food,
        device=args.device
    )

    # Run based on mode
    if args.mode == 'random':
        visualizer.run_random(num_rounds=args.episodes)

    elif args.mode == 'trained':
        if not args.weights1 or not args.weights2:
            print("Error: --weights1 and --weights2 required for trained mode")
            return
        visualizer.run_trained(args.weights1, args.weights2, num_rounds=args.episodes)

    elif args.mode == 'scripted':
        if not args.weights1:
            print("Error: --weights1 required for scripted mode")
            return
        visualizer.run_scripted(args.weights1, args.opponent, num_rounds=args.episodes)

    pygame.quit()


if __name__ == '__main__':
    main()
