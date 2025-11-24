"""
PPO Two-Snake CNN Training Script

CNN-based variant using grid representation instead of features.
Uses 5-channel grids: self_head, self_body, opp_head, opp_body, food.

Expected to be 5-10x faster than MLP according to docs.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
import argparse
from dataclasses import dataclass
from typing import Tuple
from datetime import datetime

from train_ppo_two_snake_mlp import PPOConfig, PPOBuffer, TwoSnakePPOTrainer as BaseTrainer
from core.networks import PPO_Actor_CNN, PPO_Critic_CNN
from core.state_representations_competitive import CompetitiveGridEncoder
from core.utils import set_seed, get_device


class CNNAgent:
    """PPO agent using CNN networks for grid-based state"""

    def __init__(self, grid_size, input_channels, output_dim, actor_lr, critic_lr, device, name="Agent"):
        self.device = device
        self.name = name

        # Use CNN networks from core
        self.actor = PPO_Actor_CNN(grid_size, input_channels, output_dim).to(device)
        self.critic = PPO_Critic_CNN(grid_size, input_channels).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_scaler = GradScaler('cuda')
        self.critic_scaler = GradScaler('cuda')
        self.total_steps = 0

    def select_actions(self, states):
        with torch.no_grad():
            logits = self.actor(states)
            values = self.critic(states).squeeze()
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def select_greedy_actions(self, states):
        with torch.no_grad():
            logits = self.actor(states)
            probs = F.softmax(logits, dim=-1)
            actions = probs.argmax(dim=-1)
        return actions

    def evaluate_actions(self, states, actions):
        logits = self.actor(states)
        values = self.critic(states).squeeze()
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


class CNNTrainer(BaseTrainer):
    """
    Extends MLP trainer to use CNN networks with grid encoding.

    Key differences:
    - Uses CompetitiveGridEncoder instead of feature vectors
    - Uses CNNAgent with CNN networks
    - Grid has 5 channels: self_head, self_body, opp_head, opp_body, food
    """

    def __init__(self, config: PPOConfig):
        # Don't call super().__init__() - we'll override everything
        self.config = config
        set_seed(config.seed)
        self.device = get_device()

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Import environment here to avoid circular dependency
        from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv

        self.env = VectorizedTwoSnakeEnv(
            num_envs=config.num_envs,
            grid_size=config.grid_size,
            max_steps=config.max_steps,
            target_food=config.target_food,
            device=self.device
        )

        # Grid encoder (5 channels)
        self.grid_encoder = CompetitiveGridEncoder(
            grid_size=config.grid_size,
            device=self.device
        )

        # CNN agents
        self.agent1 = CNNAgent(
            grid_size=config.grid_size,
            input_channels=5,
            output_dim=3,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            device=self.device,
            name="Big-CNN"
        )

        self.agent2 = CNNAgent(
            grid_size=config.grid_size,
            input_channels=5,
            output_dim=3,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            device=self.device,
            name="Small-CNN"
        )

        self.buffer1 = PPOBuffer(config.rollout_steps, self.device)
        self.buffer2 = PPOBuffer(config.rollout_steps, self.device)

        self.total_steps = 0
        self.total_rounds = 0
        self.round_winners = []
        self.scores1 = []
        self.scores2 = []
        self.losses1 = []
        self.losses2 = []

    def get_observations(self):
        """Override to return grid states instead of feature vectors"""
        grid1 = self.grid_encoder.encode_batch(
            self.env.snakes1, self.env.lengths1,
            self.env.snakes2, self.env.lengths2,
            self.env.foods
        )

        grid2 = self.grid_encoder.encode_batch(
            self.env.snakes2, self.env.lengths2,
            self.env.snakes1, self.env.lengths1,
            self.env.foods
        )

        return grid1, grid2

    def train(self):
        """Training loop with grid-based observations"""
        import time
        import numpy as np

        print("\n" + "="*70, flush=True)
        print("PPO TWO-SNAKE CNN TRAINING", flush=True)
        print("="*70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num envs: {self.config.num_envs}", flush=True)
        print(f"Grid: {self.config.grid_size}x{self.config.grid_size} (5 channels)", flush=True)
        print(f"Total steps: {self.config.total_steps}", flush=True)
        print("="*70 + "\n", flush=True)

        start_time = time.time()
        self.env.reset()

        while self.total_steps < self.config.total_steps:
            self.buffer1.clear()
            self.buffer2.clear()

            # Rollout
            for _ in range(max(1, self.config.rollout_steps // self.config.num_envs)):
                obs1, obs2 = self.get_observations()

                actions1, log_probs1, values1 = self.agent1.select_actions(obs1)
                actions2, log_probs2, values2 = self.agent2.select_actions(obs2)

                _, _, r1, r2, dones, info = self.env.step(actions1, actions2)

                self.buffer1.add(obs1, actions1, r1, dones, log_probs1, values1)
                self.buffer2.add(obs2, actions2, r2, dones, log_probs2, values2)

                if dones.any():
                    # Use info dict to get winners (saved before auto-reset)
                    num_done = len(info['done_envs'])
                    for i in range(num_done):
                        self.round_winners.append(int(info['winners'][i]))
                        self.total_rounds += 1
                        self.scores1.append(info['food_counts1'][i])
                        self.scores2.append(info['food_counts2'][i])

                self.total_steps += 1
                if self.total_steps >= self.config.total_steps:
                    break

            if self.total_steps >= self.config.total_steps:
                break

            # Get next values
            obs1, obs2 = self.get_observations()
            with torch.no_grad():
                next_value1 = self.agent1.critic(obs1).squeeze()
                next_value2 = self.agent2.critic(obs2).squeeze()

            # Process buffers and update (use parent class methods)
            states1, actions1, rewards1, dones1, log_probs1, values1 = self.buffer1.get()
            states2, actions2, rewards2, dones2, log_probs2, values2 = self.buffer2.get()

            steps_per_env = len(self.buffer1.states)
            rewards1 = rewards1.view(steps_per_env, self.config.num_envs)
            values1 = values1.view(steps_per_env, self.config.num_envs)
            dones1 = dones1.view(steps_per_env, self.config.num_envs)
            rewards2 = rewards2.view(steps_per_env, self.config.num_envs)
            values2 = values2.view(steps_per_env, self.config.num_envs)
            dones2 = dones2.view(steps_per_env, self.config.num_envs)

            # Compute advantages and returns (VECTORIZED - all envs at once)
            advantages1, returns1 = self.compute_gae(
                rewards1, values1, dones1, next_value1
            )
            advantages2, returns2 = self.compute_gae(
                rewards2, values2, dones2, next_value2
            )

            # Flatten for update
            advantages1 = advantages1.view(-1)
            returns1 = returns1.view(-1)
            advantages2 = advantages2.view(-1)
            returns2 = returns2.view(-1)

            actor_loss1, critic_loss1, _ = self.update_agent(
                self.agent1, states1, actions1, log_probs1, advantages1, returns1
            )
            actor_loss2, critic_loss2, _ = self.update_agent(
                self.agent2, states2, actions2, log_probs2, advantages2, returns2
            )

            self.losses1.append(actor_loss1 + critic_loss1)
            self.losses2.append(actor_loss2 + critic_loss2)

            # Logging
            if self.total_steps % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_steps / elapsed if elapsed > 0 else 0
                win_rate = self.calculate_win_rate()

                avg_score1 = np.mean(self.scores1[-100:]) if self.scores1 else 0
                avg_score2 = np.mean(self.scores2[-100:]) if self.scores2 else 0
                avg_loss1 = np.mean(self.losses1[-10:]) if self.losses1 else 0

                print(f"[Step {self.total_steps:>6}/{self.config.total_steps}] "
                      f"Win: {win_rate:.2%} | Scores: {avg_score1:.1f}v{avg_score2:.1f} | "
                      f"Loss: {avg_loss1:.4f} | FPS: {fps:.0f}", flush=True)

            if self.total_steps % self.config.save_interval == 0:
                self.save_checkpoint()

        total_time = time.time() - start_time
        print(f"\nTraining complete! Time: {total_time/60:.1f}min", flush=True)

        # Print profiling stats
        stats = self.grid_encoder.get_profiling_stats()
        if stats:
            print(f"Grid Encoding Performance:", flush=True)
            print(f"  Total encodings: {stats['total_encodings']}", flush=True)
            print(f"  Avg encoding time: {stats['avg_encoding_ms']:.2f} ms", flush=True)
            print(f"  Total encoding time: {stats['total_time_s']:.1f} s ({stats['total_time_s']/total_time*100:.1f}% of total)", flush=True)

        self.save_checkpoint(final=True)


def main():
    parser = argparse.ArgumentParser(description='PPO Two-Snake CNN Training')
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--total-steps', type=int, default=20000)  # Optimized: was 250K, reduced 92% (CNN slower than MLP)
    parser.add_argument('--save-dir', type=str, default='results/weights/ppo_two_snake_cnn')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = PPOConfig(
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        save_dir=args.save_dir,
        seed=args.seed
    )

    trainer = CNNTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
