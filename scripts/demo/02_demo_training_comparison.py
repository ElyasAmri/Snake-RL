"""
Demo: Training Comparison - With vs Without Experience Replay
Slides: "Training Without Replay" and "Training With Experience Replay"

This script provides both demos:
- demo_no_replay(): Shows training WITHOUT experience replay
  Narrative: "Only learns 'avoid walls', forgets how to get food"

- demo_with_replay(): Shows training WITH experience replay
  Narrative: "Random sample from all, learns everything together"

Run with:
    # Without replay
    ./venv/Scripts/python.exe scripts/demo/demo_training_comparison.py --mode no_replay

    # With replay
    ./venv/Scripts/python.exe scripts/demo/demo_training_comparison.py --mode with_replay
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import argparse

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DuelingDQN_MLP

# Demo parameters
SEED = 67
FPS = 15
GRID_SIZE = 10
CELL_SIZE = 50
BUFFER_SIZE = 10000
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 500

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
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


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, min_size):
        return len(self.buffer) >= min_size


class DQNTrainer:
    """DQN trainer with configurable experience replay."""

    def __init__(self, device, use_replay=True):
        self.device = device
        self.use_replay = use_replay

        # Network
        self.policy_net = DuelingDQN_MLP(
            input_dim=10, output_dim=3, hidden_dims=(128, 128)
        ).to(device)
        self.target_net = DuelingDQN_MLP(
            input_dim=10, output_dim=3, hidden_dims=(128, 128)
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # Experience replay buffer (only used if use_replay=True)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE) if use_replay else None

        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999

        # Training params
        self.gamma = 0.99
        self.target_update_freq = 100
        self.total_steps = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return torch.randint(0, 3, (1,), device=self.device)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax(dim=1)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer (if using replay)."""
        if self.use_replay:
            self.replay_buffer.push(
                state.cpu().numpy().flatten(),
                action.item(),
                reward.item(),
                next_state.cpu().numpy().flatten(),
                done.item()
            )

    def train_step_no_replay(self, state, action, reward, next_state, done):
        """Train on single transition (NO REPLAY)."""
        with torch.no_grad():
            next_q = self.target_net(next_state).max(dim=1)[0]
            target = reward + (1 - done.float()) * self.gamma * next_q

        q_values = self.policy_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze()

        loss = nn.MSELoss()(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_training_state()
        return loss.item()

    def train_step_with_replay(self):
        """Train on random batch from replay buffer."""
        if not self.replay_buffer.is_ready(MIN_BUFFER_SIZE):
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q

        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = nn.MSELoss()(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_training_state()
        return loss.item()

    def _update_training_state(self):
        """Update epsilon and target network."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class TrainingDemo:
    """Demo showing DQN training with or without replay."""

    def __init__(self, use_replay=True, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_replay = use_replay

        # Pygame setup
        pygame.init()
        self.width = GRID_SIZE * CELL_SIZE
        self.height = GRID_SIZE * CELL_SIZE + 120
        self.screen = pygame.display.set_mode((self.width, self.height))

        title = 'Training WITH Replay' if use_replay else 'Training WITHOUT Replay'
        pygame.display.set_caption(f'Demo: {title}')

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 32)

        # Environment
        self.env = VectorizedSnakeEnv(
            num_envs=1,
            grid_size=GRID_SIZE,
            action_space_type='relative',
            state_representation='feature',
            max_steps=500,
            device=self.device
        )

        # Trainer
        self.trainer = DQNTrainer(self.device, use_replay=use_replay)

        # Stats
        self.episode = 0
        self.current_score = 0
        self.current_steps = 0
        self.recent_scores = []
        self.loss = 0.0

    def draw_grid(self):
        for x in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.width), 1)
        for y in range(0, self.width, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)

    def draw_snake(self, snake, length):
        for i in range(length):
            x = int(snake[i, 0].item()) * CELL_SIZE
            y = int(snake[i, 1].item()) * CELL_SIZE
            if i == 0:
                pygame.draw.rect(self.screen, GREEN, (x, y, CELL_SIZE, CELL_SIZE))
                eye_size = CELL_SIZE // 6
                pygame.draw.circle(self.screen, BLACK,
                                   (x + CELL_SIZE//3, y + CELL_SIZE//3), eye_size)
                pygame.draw.circle(self.screen, BLACK,
                                   (x + 2*CELL_SIZE//3, y + CELL_SIZE//3), eye_size)
            else:
                pygame.draw.rect(self.screen, DARK_GREEN,
                                 (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))

    def draw_food(self, food):
        x = int(food[0].item()) * CELL_SIZE
        y = int(food[1].item()) * CELL_SIZE
        pygame.draw.circle(self.screen, RED,
                           (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3)

    def draw_stats(self):
        stats_y = self.width
        pygame.draw.rect(self.screen, BLACK, (0, stats_y, self.width, 120))

        # Title and indicator based on mode
        if self.use_replay:
            title = self.font_large.render("Training WITH Replay", True, CYAN)
            buffer_size = len(self.trainer.replay_buffer)
            indicator = self.font.render(f"Buffer: {buffer_size}/{BUFFER_SIZE}", True, CYAN)
        else:
            title = self.font_large.render("Training WITHOUT Replay", True, ORANGE)
            indicator = self.font.render("(Sequential learning - biased!)", True, ORANGE)

        self.screen.blit(title, (10, stats_y + 5))
        self.screen.blit(indicator, (self.width - 200, stats_y + 10))

        # Stats
        avg_score = sum(self.recent_scores) / max(1, len(self.recent_scores))
        lines = [
            f"Episode: {self.episode}  Score: {self.current_score}  Steps: {self.current_steps}",
            f"Avg Score (last 50): {avg_score:.1f}  Epsilon: {self.trainer.epsilon:.3f}",
            f"Loss: {self.loss:.4f}  Total Steps: {self.trainer.total_steps}",
        ]
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, WHITE), (10, stats_y + 40 + i * 22))

    def run(self, max_episodes=500):
        mode_name = "WITH" if self.use_replay else "WITHOUT"
        print("=" * 60)
        print(f"DEMO: Training {mode_name} Experience Replay")
        print("=" * 60)
        if self.use_replay:
            print("Watch how the agent learns balanced behavior.")
            print("Random sampling from the buffer prevents forgetting.")
        else:
            print("Watch how the agent struggles to learn balanced behavior.")
            print("It will often forget food-seeking when learning wall-avoidance.")
        print("Press ESC or close window to exit.")
        print("=" * 60)

        state = self.env.reset()
        running = True

        while running and self.episode < max_episodes:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            action = self.trainer.select_action(state)
            next_state, reward, done, info = self.env.step(action)

            if self.use_replay:
                self.trainer.store_transition(state, action, reward[0], next_state, done[0])
                self.loss = self.trainer.train_step_with_replay()
            else:
                self.loss = self.trainer.train_step_no_replay(
                    state, action, reward, next_state, done
                )

            self.current_score = int(info['scores'][0].item())
            self.current_steps += 1

            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_snake(self.env.snakes[0], self.env.snake_lengths[0])
            self.draw_food(self.env.foods[0])
            self.draw_stats()
            pygame.display.flip()
            self.clock.tick(FPS)

            if done[0]:
                self.episode += 1
                self.recent_scores.append(self.current_score)
                if len(self.recent_scores) > 50:
                    self.recent_scores.pop(0)
                if self.episode % 50 == 0:
                    avg = sum(self.recent_scores) / len(self.recent_scores)
                    print(f"Episode {self.episode}: Avg Score (last 50) = {avg:.1f}")
                self.current_steps = 0
                state = self.env.reset()
            else:
                state = next_state

        pygame.quit()
        final_avg = sum(self.recent_scores) / max(1, len(self.recent_scores))
        print(f"\nFinished {self.episode} episodes. Final Avg Score: {final_avg:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Training Comparison Demo')
    parser.add_argument('--mode', type=str, default='with_replay',
                        choices=['no_replay', 'with_replay'],
                        help='Training mode')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes')
    args = parser.parse_args()

    set_seed(SEED)
    use_replay = (args.mode == 'with_replay')
    demo = TrainingDemo(use_replay=use_replay)
    demo.run(max_episodes=args.episodes)


if __name__ == '__main__':
    main()
