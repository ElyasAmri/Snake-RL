"""
Demo: Competitive Match
Slide: Competitive Match Demo
Narrative: "Trained agents competing head-to-head"

Shows two-snake competition with two modes:
1. PPO Vectorized: Big Brain (256n) vs Small Brain (128n) using GPU
2. DQN Classic: Two DQN agents using CPU-based environment

Run with:
  PPO mode (default): ./venv/Scripts/python.exe scripts/demo/06_demo_competitive.py
  DQN mode: ./venv/Scripts/python.exe scripts/demo/06_demo_competitive.py --mode dqn
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import pygame
import torch
import torch.nn.functional as F
import random
import numpy as np
import glob

# Demo parameters
SEED = 67
FPS = 10
TARGET_FOOD = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
BLUE = (0, 100, 255)
LIGHT_BLUE = (100, 150, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
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


def find_latest_checkpoint(pattern):
    """Find the latest checkpoint matching pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return files[0]


def get_action_from_actor(actor, obs, device):
    """Get greedy action from PPO actor."""
    with torch.no_grad():
        logits = actor(obs)
        probs = F.softmax(logits, dim=-1)
        action = probs.argmax(dim=-1)
    return action.item()


class PPOCompetitiveDemo:
    """Demo using PPO agents with vectorized environment."""

    def __init__(self, device='cuda'):
        from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
        from core.networks import PPO_Actor_MLP

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mode_name = "PPO (Vectorized)"
        self.grid_size = 20
        self.cell_size = 25
        self.weights_dir = "results/weights/ppo_two_snake_mlp"

        # Pygame setup
        pygame.init()
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size + 150
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Demo: Big Brain vs Small Brain (PPO)')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 18)

        # Environment
        self.env = VectorizedTwoSnakeEnv(
            num_envs=1,
            grid_size=self.grid_size,
            target_food=TARGET_FOOD,
            max_steps=1000,
            device=self.device
        )

        # Load trained actors
        self.actor1 = None
        self.actor2 = None
        self.weights_loaded = self._load_weights(PPO_Actor_MLP)

        # Stats
        self.round = 0
        self.snake1_wins = 0
        self.snake2_wins = 0
        self.ties = 0
        self.last_winner = 0  # Track winner from step() before reset clears it
        self.last_score1 = 0  # Track final scores before reset
        self.last_score2 = 0

    def _load_weights(self, PPO_Actor_MLP):
        """Load trained PPO weights."""
        big_pattern = str(Path(self.weights_dir) / "big_256x256_*.pt")
        small_pattern = str(Path(self.weights_dir) / "small_128x128_*.pt")

        big_checkpoint = find_latest_checkpoint(big_pattern)
        small_checkpoint = find_latest_checkpoint(small_pattern)

        if big_checkpoint and small_checkpoint:
            print(f"Loading Big Brain weights from: {big_checkpoint}")
            print(f"Loading Small Brain weights from: {small_checkpoint}")

            self.actor1 = PPO_Actor_MLP(
                input_dim=33, output_dim=3, hidden_dims=(256, 256)
            ).to(self.device)
            self.actor2 = PPO_Actor_MLP(
                input_dim=33, output_dim=3, hidden_dims=(128, 128)
            ).to(self.device)

            checkpoint1 = torch.load(big_checkpoint, map_location=self.device, weights_only=False)
            checkpoint2 = torch.load(small_checkpoint, map_location=self.device, weights_only=False)

            self.actor1.load_state_dict(checkpoint1['actor'])
            self.actor2.load_state_dict(checkpoint2['actor'])

            self.actor1.eval()
            self.actor2.eval()

            print("Weights loaded successfully!")
            return True
        else:
            print("WARNING: Could not find trained PPO weights!")
            print(f"Searched for: {big_pattern}")
            print(f"Searched for: {small_pattern}")
            print("Using random actions instead.")
            return False

    def get_snakes_and_food(self):
        """Get snake positions and food from environment."""
        snake1 = self.env.snakes1[0]
        snake2 = self.env.snakes2[0]
        length1 = self.env.lengths1[0].item()
        length2 = self.env.lengths2[0].item()
        alive1 = self.env.alive1[0].item()
        alive2 = self.env.alive2[0].item()
        food = self.env.foods[0]
        score1 = self.env.food_counts1[0].item()
        score2 = self.env.food_counts2[0].item()
        winner = self.env.round_winners[0].item()
        return snake1, snake2, length1, length2, alive1, alive2, food, score1, score2, winner

    def step(self, obs1, obs2):
        """Execute one step."""
        if self.weights_loaded:
            action1 = get_action_from_actor(self.actor1, obs1, self.device)
            action2 = get_action_from_actor(self.actor2, obs2, self.device)
        else:
            action1 = random.randint(0, 2)
            action2 = random.randint(0, 2)

        actions1 = torch.tensor([action1], device=self.device)
        actions2 = torch.tensor([action2], device=self.device)
        obs1, obs2, r1, r2, dones, info = self.env.step(actions1, actions2)
        done = dones[0].item()

        # Capture winner and scores from info BEFORE reset clears them
        if done and 'winners' in info:
            self.last_winner = info['winners'][0].item()
            self.last_score1 = info['food_counts1'][0].item()
            self.last_score2 = info['food_counts2'][0].item()
        else:
            self.last_winner = 0

        return obs1, obs2, done

    def reset(self):
        """Reset environment."""
        return self.env.reset()


class DQNCompetitiveDemo:
    """Demo using DQN agents with classic environment."""

    def __init__(self, device='cpu'):
        from core.environment_two_snake_classic import TwoSnakeCompetitiveEnv
        from agents.vanilla_dqn import VanillaDQNAgent

        self.device = torch.device(device)
        self.mode_name = "DQN (Classic)"
        self.grid_size = 10
        self.cell_size = 50
        self.weights_dir = "results/weights/competitive/classic"

        # Pygame setup
        pygame.init()
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size + 150
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Demo: DQN Agent 1 vs DQN Agent 2 (Classic)')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 20)

        # Environment
        self.env = TwoSnakeCompetitiveEnv(
            grid_size=self.grid_size,
            target_score=TARGET_FOOD,
            max_steps=500
        )

        # Load trained agents
        self.agent1 = VanillaDQNAgent(
            state_size=20,
            action_size=3,
            hidden_size=128
        )
        self.agent2 = VanillaDQNAgent(
            state_size=20,
            action_size=3,
            hidden_size=128
        )
        self.weights_loaded = self._load_weights()

        # Stats
        self.round = 0
        self.snake1_wins = 0
        self.snake2_wins = 0
        self.ties = 0

        # Current game state
        self.winner = 0
        self.score1 = 0
        self.score2 = 0

    def _load_weights(self):
        """Load trained DQN weights."""
        agent1_pattern = str(Path(self.weights_dir) / "agent1_dqn_*.pt")
        agent2_pattern = str(Path(self.weights_dir) / "agent2_dqn_*.pt")

        agent1_checkpoint = find_latest_checkpoint(agent1_pattern)
        agent2_checkpoint = find_latest_checkpoint(agent2_pattern)

        if agent1_checkpoint and agent2_checkpoint:
            print(f"Loading Agent 1 weights from: {agent1_checkpoint}")
            print(f"Loading Agent 2 weights from: {agent2_checkpoint}")

            self.agent1.load(agent1_checkpoint)
            self.agent2.load(agent2_checkpoint)

            # Set to greedy mode (no exploration)
            self.agent1.epsilon = 0
            self.agent2.epsilon = 0

            print("DQN weights loaded successfully!")
            return True
        else:
            print("WARNING: Could not find trained DQN weights!")
            print(f"Searched for: {agent1_pattern}")
            print(f"Searched for: {agent2_pattern}")
            print("Using random actions instead.")
            return False

    def get_snakes_and_food(self):
        """Get snake positions and food from classic environment."""
        # Convert classic env positions to tensors for uniform API
        snake1_positions = self.env.snake1_positions
        snake2_positions = self.env.snake2_positions
        food = self.env.food_position

        length1 = len(snake1_positions)
        length2 = len(snake2_positions)
        alive1 = self.env.snake1_alive
        alive2 = self.env.snake2_alive
        score1 = self.env.score1
        score2 = self.env.score2

        # Store for drawing
        self._snake1_positions = snake1_positions
        self._snake2_positions = snake2_positions
        self._food_position = food
        self.score1 = score1
        self.score2 = score2

        return (snake1_positions, snake2_positions, length1, length2,
                alive1, alive2, food, score1, score2, self.winner)

    def step(self, obs1, obs2):
        """Execute one step."""
        if self.weights_loaded:
            action1 = self.agent1.select_action(obs1, training=False)
            action2 = self.agent2.select_action(obs2, training=False)
        else:
            action1 = random.randint(0, 2)
            action2 = random.randint(0, 2)

        actions = {'agent1': action1, 'agent2': action2}
        next_obs, rewards, terminated, truncated, info = self.env.step(actions)

        obs1 = next_obs['agent1']
        obs2 = next_obs['agent2']
        done = terminated['agent1'] or truncated['agent1']

        # Track winner
        self.winner = info.get('winner', 0)

        return obs1, obs2, done

    def reset(self):
        """Reset environment."""
        self.winner = 0
        obs, info = self.env.reset()
        return obs['agent1'], obs['agent2']


class CompetitiveDemo:
    """Unified demo supporting both PPO and DQN modes."""

    def __init__(self, mode='ppo'):
        if mode == 'ppo':
            self.backend = PPOCompetitiveDemo()
        else:
            self.backend = DQNCompetitiveDemo()

        self.screen = self.backend.screen
        self.clock = self.backend.clock
        self.font = self.backend.font
        self.font_large = self.backend.font_large
        self.font_small = self.backend.font_small
        self.grid_size = self.backend.grid_size
        self.cell_size = self.backend.cell_size
        self.width = self.backend.width

    def draw_grid(self):
        """Draw grid lines."""
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.width), 1)
        for y in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)

    def draw_snake_ppo(self, snake, length, color_head, color_body):
        """Draw snake from tensor positions (PPO mode)."""
        for i in range(length):
            x = int(snake[i, 0].item()) * self.cell_size
            y = int(snake[i, 1].item()) * self.cell_size

            if i == 0:
                pygame.draw.rect(self.screen, color_head,
                                 (x, y, self.cell_size, self.cell_size))
                eye_size = max(2, self.cell_size // 6)
                pygame.draw.circle(self.screen, BLACK,
                                   (x + self.cell_size//3, y + self.cell_size//3), eye_size)
                pygame.draw.circle(self.screen, BLACK,
                                   (x + 2*self.cell_size//3, y + self.cell_size//3), eye_size)
            else:
                pygame.draw.rect(self.screen, color_body,
                                 (x + 1, y + 1, self.cell_size - 2, self.cell_size - 2))

    def draw_snake_classic(self, positions, color_head, color_body):
        """Draw snake from list positions (DQN classic mode)."""
        for i, pos in enumerate(positions):
            x = pos[0] * self.cell_size
            y = pos[1] * self.cell_size

            if i == 0:
                pygame.draw.rect(self.screen, color_head,
                                 (x, y, self.cell_size, self.cell_size))
                eye_size = max(2, self.cell_size // 6)
                pygame.draw.circle(self.screen, BLACK,
                                   (x + self.cell_size//3, y + self.cell_size//3), eye_size)
                pygame.draw.circle(self.screen, BLACK,
                                   (x + 2*self.cell_size//3, y + self.cell_size//3), eye_size)
            else:
                pygame.draw.rect(self.screen, color_body,
                                 (x + 1, y + 1, self.cell_size - 2, self.cell_size - 2))

    def draw_food_ppo(self, food):
        """Draw food from tensor (PPO mode)."""
        x = int(food[0].item()) * self.cell_size
        y = int(food[1].item()) * self.cell_size
        pygame.draw.circle(self.screen, RED,
                           (x + self.cell_size//2, y + self.cell_size//2),
                           self.cell_size//3)

    def draw_food_classic(self, food):
        """Draw food from tuple (DQN classic mode)."""
        if food:
            x = food[0] * self.cell_size
            y = food[1] * self.cell_size
            pygame.draw.circle(self.screen, RED,
                               (x + self.cell_size//2, y + self.cell_size//2),
                               self.cell_size//3)

    def draw_stats(self, score1, score2, winner):
        """Draw stats panel."""
        stats_y = self.width + 10
        pygame.draw.rect(self.screen, DARK_GRAY, (0, self.width, self.width, 150))

        # Title
        title = self.font_large.render(f"Mode: {self.backend.mode_name}", True, CYAN)
        self.screen.blit(title, (10, stats_y))

        # Weights status
        status = "TRAINED" if self.backend.weights_loaded else "RANDOM"
        status_color = GREEN if self.backend.weights_loaded else YELLOW
        status_text = self.font_small.render(f"Status: {status}", True, status_color)
        self.screen.blit(status_text, (self.width - 110, stats_y))

        # Round
        round_text = self.font.render(f"Round: {self.backend.round}", True, WHITE)
        self.screen.blit(round_text, (self.width - 110, stats_y + 25))

        # Snake 1 (Green)
        pygame.draw.rect(self.screen, GREEN, (10, stats_y + 55, 20, 20))
        s1_label = "Big Brain" if isinstance(self.backend, PPOCompetitiveDemo) else "Agent 1"
        s1_text = self.font.render(f"{s1_label}: {score1}/{TARGET_FOOD}  Wins: {self.backend.snake1_wins}", True, WHITE)
        self.screen.blit(s1_text, (35, stats_y + 55))

        # Snake 2 (Blue)
        pygame.draw.rect(self.screen, BLUE, (10, stats_y + 85, 20, 20))
        s2_label = "Small Brain" if isinstance(self.backend, PPOCompetitiveDemo) else "Agent 2"
        s2_text = self.font.render(f"{s2_label}: {score2}/{TARGET_FOOD}  Wins: {self.backend.snake2_wins}", True, WHITE)
        self.screen.blit(s2_text, (35, stats_y + 85))

        # Ties
        if self.backend.ties > 0:
            ties_text = self.font_small.render(f"Ties: {self.backend.ties}", True, YELLOW)
            self.screen.blit(ties_text, (self.width - 80, stats_y + 50))

        # Winner status
        if winner == 1:
            self.screen.blit(self.font.render(f"{s1_label} Won!", True, GREEN),
                             (self.width - 130, stats_y + 75))
        elif winner == 2:
            self.screen.blit(self.font.render(f"{s2_label} Won!", True, BLUE),
                             (self.width - 140, stats_y + 95))
        elif winner == 3:
            self.screen.blit(self.font.render("TIE!", True, YELLOW),
                             (self.width - 80, stats_y + 85))

    def run(self, num_rounds=50):
        """Run the competitive match demo."""
        print("=" * 60)
        print(f"DEMO: {self.backend.mode_name}")
        print("=" * 60)
        if self.backend.weights_loaded:
            print("Using TRAINED agents")
        else:
            print("Using RANDOM actions (weights not found)")
        print("=" * 60)

        running = True

        for round_num in range(num_rounds):
            if not running:
                break

            self.backend.round = round_num + 1
            obs1, obs2 = self.backend.reset()
            done = False

            while not done and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                        break

                # Get state info
                result = self.backend.get_snakes_and_food()
                (snake1, snake2, length1, length2, alive1, alive2,
                 food, score1, score2, winner) = result

                # Draw
                self.screen.fill(BLACK)
                self.draw_grid()

                # Draw based on mode
                if isinstance(self.backend, PPOCompetitiveDemo):
                    self.draw_food_ppo(food)
                    if alive2:
                        self.draw_snake_ppo(snake2, length2, BLUE, LIGHT_BLUE)
                    if alive1:
                        self.draw_snake_ppo(snake1, length1, GREEN, DARK_GREEN)
                else:
                    self.draw_food_classic(food)
                    if alive2:
                        self.draw_snake_classic(snake2, BLUE, LIGHT_BLUE)
                    if alive1:
                        self.draw_snake_classic(snake1, GREEN, DARK_GREEN)

                self.draw_stats(score1, score2, winner)
                pygame.display.flip()
                self.clock.tick(FPS)

                # Step
                obs1, obs2, done = self.backend.step(obs1, obs2)

            # Track winner - use last_winner captured from step() before reset
            # For PPO, this comes from info['winners'] captured in step()
            # For DQN, this comes from info['winner'] in step()
            if hasattr(self.backend, 'last_winner'):
                winner = self.backend.last_winner
                final_score1 = self.backend.last_score1
                final_score2 = self.backend.last_score2
            else:
                winner = self.backend.winner  # DQN classic mode
                final_score1 = self.backend.score1
                final_score2 = self.backend.score2

            # Get state for last frame drawing (positions may be reset but we still draw)
            result = self.backend.get_snakes_and_food()
            (snake1, snake2, length1, length2, alive1, alive2, food, _, _, _) = result

            if winner == 1:
                self.backend.snake1_wins += 1
            elif winner == 2:
                self.backend.snake2_wins += 1
            else:
                self.backend.ties += 1

            # Draw the final frame first (so we see the winning state)
            self.screen.fill(BLACK)
            self.draw_grid()
            if isinstance(self.backend, PPOCompetitiveDemo):
                self.draw_food_ppo(food)
                if alive2:
                    self.draw_snake_ppo(snake2, length2, BLUE, LIGHT_BLUE)
                if alive1:
                    self.draw_snake_ppo(snake1, length1, GREEN, DARK_GREEN)
            else:
                self.draw_food_classic(food)
                if alive2:
                    self.draw_snake_classic(snake2, BLUE, LIGHT_BLUE)
                if alive1:
                    self.draw_snake_classic(snake1, GREEN, DARK_GREEN)
            self.draw_stats(final_score1, final_score2, winner)

            # Draw round-end overlay with winner and scores
            s1_label = "Big Brain" if isinstance(self.backend, PPOCompetitiveDemo) else "Agent 1"
            s2_label = "Small Brain" if isinstance(self.backend, PPOCompetitiveDemo) else "Agent 2"

            # Semi-transparent overlay
            overlay = pygame.Surface((self.width, 80))
            overlay.set_alpha(200)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, self.width // 2 - 40))

            # Winner text
            if winner == 1:
                winner_text = self.font_large.render(f"{s1_label} WINS!", True, GREEN)
            elif winner == 2:
                winner_text = self.font_large.render(f"{s2_label} WINS!", True, BLUE)
            else:
                winner_text = self.font_large.render("TIE!", True, YELLOW)
            text_rect = winner_text.get_rect(center=(self.width // 2, self.width // 2 - 15))
            self.screen.blit(winner_text, text_rect)

            # Score text
            score_text = self.font.render(f"Score: {final_score1} - {final_score2}", True, WHITE)
            score_rect = score_text.get_rect(center=(self.width // 2, self.width // 2 + 15))
            self.screen.blit(score_text, score_rect)

            pygame.display.flip()

            winner_name = (s1_label if winner == 1 else s2_label if winner == 2 else 'Tie')
            print(f"Round {self.backend.round}: {winner_name} ({final_score1}-{final_score2})")

            # Pause to show result
            pygame.time.wait(800)

        pygame.quit()
        print("\n" + "=" * 60)
        print(f"RESULTS: {self.backend.mode_name}")
        print("=" * 60)
        s1_label = "Big Brain" if isinstance(self.backend, PPOCompetitiveDemo) else "Agent 1"
        s2_label = "Small Brain" if isinstance(self.backend, PPOCompetitiveDemo) else "Agent 2"
        print(f"{s1_label} Wins: {self.backend.snake1_wins}")
        print(f"{s2_label} Wins: {self.backend.snake2_wins}")
        print(f"Ties: {self.backend.ties}")
        total = self.backend.snake1_wins + self.backend.snake2_wins + self.backend.ties
        if total > 0:
            print(f"{s1_label} Win Rate: {self.backend.snake1_wins/total:.1%}")
            print(f"{s2_label} Win Rate: {self.backend.snake2_wins/total:.1%}")


def main():
    parser = argparse.ArgumentParser(description='Competitive Snake Demo')
    parser.add_argument('--mode', type=str, default='ppo', choices=['ppo', 'dqn'],
                        help='Mode: ppo (vectorized GPU) or dqn (classic CPU)')
    parser.add_argument('--rounds', type=int, default=50, help='Number of rounds')
    args = parser.parse_args()

    set_seed(SEED)
    demo = CompetitiveDemo(mode=args.mode)
    demo.run(num_rounds=args.rounds)


if __name__ == '__main__':
    main()
