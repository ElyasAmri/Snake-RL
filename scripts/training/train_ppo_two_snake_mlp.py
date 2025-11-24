"""
PPO Training for Two-Snake Competitive Environment (MLP-based)

Uses:
- Vectorized two-snake environment (128 parallel games)
- 35-dimensional feature vector state representation
- Separate actor-critic networks for each snake
- Direct co-evolution (both agents learning simultaneously)
- Rollout-based training with GAE
- Mixed precision (FP16) for faster training

Network sizes:
- Big snake (Agent 1): 256x256 hidden layers
- Small snake (Agent 2): 128x128 hidden layers
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import time
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
from datetime import datetime

from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from core.networks import PPO_Actor_MLP, PPO_Critic_MLP
from core.utils import set_seed, get_device


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Environment
    num_envs: int = 128
    grid_size: int = 20  # Increased from 10 for more exploration space
    target_food: int = 10
    max_steps: int = 1000

    # Network sizes
    big_hidden_dims: Tuple[int, int] = (256, 256)
    small_hidden_dims: Tuple[int, int] = (128, 128)

    # PPO hyperparameters
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003  # Changed from 0.001 to balance with actor
    rollout_steps: int = 2048
    batch_size: int = 64
    epochs_per_rollout: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.1  # Increased from 0.01 for more exploration
    max_grad_norm: float = 0.5

    # Training
    total_steps: int = 250000
    log_interval: int = 100
    save_interval: int = 10000

    # Other
    seed: int = 42
    save_dir: str = 'results/weights/ppo_two_snake_mlp'
    max_time: Optional[int] = None  # Maximum training time in seconds


class PPOBuffer:
    """GPU-based rollout buffer for PPO (OPTIMIZED - no CPU transfers)"""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.clear()

    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.size = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor
    ):
        """Add experience to buffer (kept on GPU)"""
        # Keep everything on GPU - no .cpu() calls!
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.size += state.shape[0]

    def get(self) -> Tuple:
        """Get all experiences (already on GPU)"""
        states = torch.cat(self.states, dim=0)
        actions = torch.cat(self.actions, dim=0)
        rewards = torch.cat(self.rewards, dim=0)
        dones = torch.cat(self.dones, dim=0)
        log_probs = torch.cat(self.log_probs, dim=0)
        values = torch.cat(self.values, dim=0)

        return states, actions, rewards, dones, log_probs, values

    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.size >= self.capacity


class TwoSnakePPOAgent:
    """PPO agent for competitive two-snake training"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, int],
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
        name: str = "Agent"
    ):
        self.device = device
        self.name = name

        # Networks
        self.actor = PPO_Actor_MLP(input_dim, output_dim, hidden_dims).to(device)
        self.critic = PPO_Critic_MLP(input_dim, hidden_dims).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Mixed precision scalers
        self.actor_scaler = GradScaler('cuda')
        self.critic_scaler = GradScaler('cuda')

        # Stats
        self.total_steps = 0

    def select_actions(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select actions using current policy

        Returns:
            actions, log_probs, values
        """
        with torch.no_grad():
            logits = self.actor(states)
            values = self.critic(states).squeeze()

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions, log_probs, values

    def select_greedy_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Select greedy actions (for frozen policy)"""
        with torch.no_grad():
            logits = self.actor(states)
            probs = F.softmax(logits, dim=-1)
            actions = probs.argmax(dim=-1)
        return actions

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update

        Returns:
            log_probs, values, entropy
        """
        logits = self.actor(states)
        values = self.critic(states).squeeze()

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


class TwoSnakePPOTrainer:
    """PPO trainer for competitive two-snake environment"""

    def __init__(self, config: PPOConfig):
        self.config = config
        set_seed(config.seed)
        self.device = get_device()

        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Environment
        self.env = VectorizedTwoSnakeEnv(
            num_envs=config.num_envs,
            grid_size=config.grid_size,
            max_steps=config.max_steps,
            target_food=config.target_food,
            device=self.device
        )

        # Agents
        self.agent1 = TwoSnakePPOAgent(
            input_dim=35,
            output_dim=3,
            hidden_dims=config.big_hidden_dims,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            device=self.device,
            name="Big-256x256"
        )

        self.agent2 = TwoSnakePPOAgent(
            input_dim=35,
            output_dim=3,
            hidden_dims=config.small_hidden_dims,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            device=self.device,
            name="Small-128x128"
        )

        # Buffers
        self.buffer1 = PPOBuffer(config.rollout_steps, self.device)
        self.buffer2 = PPOBuffer(config.rollout_steps, self.device)

        # Metrics
        self.total_steps = 0
        self.total_rounds = 0
        self.round_winners = []
        self.scores1 = []
        self.scores2 = []
        self.losses1 = []
        self.losses2 = []

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (VECTORIZED for all envs)"""
        # rewards, values, dones: (T, num_envs)
        # next_value: (num_envs,)

        T = rewards.shape[0]
        num_envs = rewards.shape[1] if rewards.dim() > 1 else 1

        # Ensure 2D tensors
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
            values = values.unsqueeze(1)
            dones = dones.unsqueeze(1)
            next_value = next_value.unsqueeze(0)

        dones = dones.float()
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(num_envs, device=rewards.device)

        # Vectorized backward pass over time
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values

        return advantages, returns

    def update_agent(
        self,
        agent: TwoSnakePPOAgent,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Update agent networks"""

        # Robust advantage normalization (avoid division by near-zero std)
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 0.1:  # Only normalize if std is significant
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = torch.clamp(advantages, -10.0, 10.0)  # Prevent extreme values
        else:
            advantages = advantages - adv_mean  # Just center if variance is low

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.config.epochs_per_rollout):
            # Random minibatch sampling
            indices = torch.randperm(states.size(0))

            for start in range(0, states.size(0), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                with autocast('cuda'):
                    log_probs, values, entropy = agent.evaluate_actions(batch_states, batch_actions)

                    # Actor loss (clipped surrogate objective)
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss
                    critic_loss = F.mse_loss(values, batch_returns)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                # Update actor
                agent.actor_optimizer.zero_grad()
                agent.actor_scaler.scale(actor_loss).backward()
                agent.actor_scaler.unscale_(agent.actor_optimizer)
                nn.utils.clip_grad_norm_(agent.actor.parameters(), self.config.max_grad_norm)
                agent.actor_scaler.step(agent.actor_optimizer)
                agent.actor_scaler.update()

                # Update critic
                agent.critic_optimizer.zero_grad()
                agent.critic_scaler.scale(critic_loss).backward()
                agent.critic_scaler.unscale_(agent.critic_optimizer)
                nn.utils.clip_grad_norm_(agent.critic.parameters(), self.config.max_grad_norm)
                agent.critic_scaler.step(agent.critic_optimizer)
                agent.critic_scaler.update()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return (
            total_actor_loss / n_updates,
            total_critic_loss / n_updates,
            total_entropy / n_updates
        )

    def calculate_win_rate(self, window: int = 100) -> float:
        """Calculate agent1 win rate over last N rounds"""
        if len(self.round_winners) < window:
            window = len(self.round_winners)

        if window == 0:
            return 0.0

        recent_winners = self.round_winners[-window:]
        snake1_wins = sum(1 for w in recent_winners if w == 1)
        return snake1_wins / window

    def train(self):
        """Main training loop"""
        print("\n" + "="*70, flush=True)
        print("PPO TWO-SNAKE COMPETITIVE TRAINING (MLP)", flush=True)
        print("="*70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num environments: {self.config.num_envs}", flush=True)
        print(f"Grid size: {self.config.grid_size}", flush=True)
        print(f"Target food: {self.config.target_food}", flush=True)
        print(f"Agent1 (Big): {self.config.big_hidden_dims} network", flush=True)
        print(f"Agent2 (Small): {self.config.small_hidden_dims} network", flush=True)
        print(f"Total steps: {self.config.total_steps}", flush=True)
        print(f"Rollout steps: {self.config.rollout_steps}", flush=True)
        print("="*70 + "\n", flush=True)

        start_time = time.time()

        # Reset environment
        obs1, obs2 = self.env.reset()

        while self.total_steps < self.config.total_steps:
            # Collect rollout
            self.buffer1.clear()
            self.buffer2.clear()

            # Collect rollout_steps // num_envs steps
            for _ in range(max(1, self.config.rollout_steps // self.config.num_envs)):
                # Select actions
                actions1, log_probs1, values1 = self.agent1.select_actions(obs1)
                actions2, log_probs2, values2 = self.agent2.select_actions(obs2)

                # Environment step
                next_obs1, next_obs2, r1, r2, dones, info = self.env.step(actions1, actions2)

                # Store transitions
                self.buffer1.add(obs1, actions1, r1, dones, log_probs1, values1)
                self.buffer2.add(obs2, actions2, r2, dones, log_probs2, values2)

                # Track completed rounds
                if dones.any():
                    # Use info dict to get winners (saved before auto-reset)
                    num_done = len(info['done_envs'])
                    for i in range(num_done):
                        winner = info['winners'][i]
                        self.round_winners.append(int(winner))
                        self.total_rounds += 1
                        self.scores1.append(info['food_counts1'][i])
                        self.scores2.append(info['food_counts2'][i])

                obs1 = next_obs1
                obs2 = next_obs2
                # FIX: Count all parallel environment steps, not just iterations
                self.total_steps += self.config.num_envs

                if self.total_steps >= self.config.total_steps:
                    break

                # Early exit if max_time reached
                if self.config.max_time and (time.time() - start_time) >= self.config.max_time:
                    print(f"\n[TIMEOUT] Max time {self.config.max_time}s reached. Stopping training...", flush=True)
                    break

            if self.total_steps >= self.config.total_steps:
                break
            if self.config.max_time and (time.time() - start_time) >= self.config.max_time:
                break

            # Compute next values for bootstrapping
            with torch.no_grad():
                next_value1 = self.agent1.critic(obs1).squeeze()
                next_value2 = self.agent2.critic(obs2).squeeze()

            # Get buffer data
            states1, actions1, rewards1, dones1, log_probs1, values1 = self.buffer1.get()
            states2, actions2, rewards2, dones2, log_probs2, values2 = self.buffer2.get()

            # Reshape for GAE computation
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

            # Update both agents
            actor_loss1, critic_loss1, entropy1 = self.update_agent(
                self.agent1, states1, actions1, log_probs1, advantages1, returns1
            )
            actor_loss2, critic_loss2, entropy2 = self.update_agent(
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
                avg_loss2 = np.mean(self.losses2[-10:]) if self.losses2 else 0

                print(f"[Step {self.total_steps:>6}/{self.config.total_steps}] "
                      f"Win Rate: {win_rate:.2%} | "
                      f"Scores: {avg_score1:.1f} vs {avg_score2:.1f} | "
                      f"Loss: {avg_loss1:.4f}, {avg_loss2:.4f} | "
                      f"FPS: {fps:.0f}", flush=True)

            # Save checkpoint
            if self.total_steps % self.config.save_interval == 0:
                self.save_checkpoint()

        total_time = time.time() - start_time

        print("\n" + "="*70, flush=True)
        print("TRAINING COMPLETE!", flush=True)
        print("="*70, flush=True)
        print(f"Total time: {total_time/60:.1f} minutes", flush=True)
        print(f"Total steps: {self.total_steps}", flush=True)
        print(f"Total rounds: {self.total_rounds}", flush=True)
        print(f"Final win rate: {self.calculate_win_rate():.2%}", flush=True)

        # Print profiling stats
        stats = self.env.feature_encoder.get_profiling_stats()
        if stats:
            print(f"\nEncoding Performance:", flush=True)
            print(f"  Total encodings: {stats['total_encodings']}", flush=True)
            print(f"  Avg danger_self: {stats['avg_danger_self_ms']:.2f} ms", flush=True)
            print(f"  Avg danger_opp: {stats['avg_danger_opp_ms']:.2f} ms", flush=True)
            print(f"  Avg food_dir: {stats['avg_food_dir_ms']:.2f} ms", flush=True)
            print(f"  Total encoding time: {stats['total_time_s']:.1f} s ({stats['total_time_s']/total_time*100:.1f}% of total)", flush=True)

        print("="*70 + "\n", flush=True)

        # Save final checkpoint
        self.save_checkpoint(final=True)

    def save_checkpoint(self, final: bool = False):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = "final" if final else f"step{self.total_steps}"

        # Save agent1
        agent1_path = self.save_dir / f"big_256x256_{suffix}_{timestamp}.pt"
        torch.save({
            'actor': self.agent1.actor.state_dict(),
            'critic': self.agent1.critic.state_dict(),
            'actor_optimizer': self.agent1.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent1.critic_optimizer.state_dict(),
            'total_steps': self.total_steps
        }, agent1_path)

        # Save agent2
        agent2_path = self.save_dir / f"small_128x128_{suffix}_{timestamp}.pt"
        torch.save({
            'actor': self.agent2.actor.state_dict(),
            'critic': self.agent2.critic.state_dict(),
            'actor_optimizer': self.agent2.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent2.critic_optimizer.state_dict(),
            'total_steps': self.total_steps
        }, agent2_path)

        # Save metrics
        metrics_path = self.save_dir / f"metrics_{suffix}_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'total_steps': self.total_steps,
                'total_rounds': self.total_rounds,
                'final_win_rate': self.calculate_win_rate(),
                'avg_score1': float(np.mean(self.scores1[-100:])) if self.scores1 else 0,
                'avg_score2': float(np.mean(self.scores2[-100:])) if self.scores2 else 0,
                'config': asdict(self.config)
            }, f, indent=2)

        print(f"Saved checkpoint: {suffix}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='PPO Two-Snake Training (MLP)')
    parser.add_argument('--num-envs', type=int, default=128, help='Number of parallel environments')
    parser.add_argument('--total-steps', type=int, default=10000, help='Total training steps (optimized: was 250K, reduced 96% based on empirical testing)')
    parser.add_argument('--rollout-steps', type=int, default=2048, help='Rollout steps before update')
    parser.add_argument('--actor-lr', type=float, default=0.0003, help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=0.0003, help='Critic learning rate')
    parser.add_argument('--save-dir', type=str, default='results/weights/ppo_two_snake_mlp', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-time', type=int, default=None, help='Maximum training time in seconds')

    args = parser.parse_args()

    config = PPOConfig(
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        save_dir=args.save_dir,
        seed=args.seed,
        max_time=args.max_time
    )

    trainer = TwoSnakePPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
