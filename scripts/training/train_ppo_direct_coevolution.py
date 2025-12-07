"""
PPO Direct Co-evolution Training (No Curriculum)

Both 128x128 and 256x256 networks learn simultaneously from random initialization.
No curriculum stages - direct competition from the start.

This serves as a baseline comparison against curriculum-based training.

Usage:
    # Quick baseline (2M steps)
    ./venv/Scripts/python.exe scripts/training/train_ppo_direct_coevolution.py \
        --total-steps 2000000 --save-dir results/weights/ppo_direct_coevolution_2M

    # Full run matching curriculum (14M steps)
    ./venv/Scripts/python.exe scripts/training/train_ppo_direct_coevolution.py \
        --total-steps 14000000 --save-dir results/weights/ppo_direct_coevolution_14M
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
class DirectCoEvolutionConfig:
    """Configuration for direct co-evolution training"""
    # Environment
    num_envs: int = 128
    grid_size: int = 20
    target_food: int = 10
    max_steps: int = 1000

    # Network sizes
    big_hidden_dims: Tuple[int, int] = (256, 256)
    small_hidden_dims: Tuple[int, int] = (128, 128)

    # PPO hyperparameters
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003
    rollout_steps: int = 2048
    batch_size: int = 64
    epochs_per_rollout: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.1
    max_grad_norm: float = 0.5

    # Training
    total_steps: int = 14_000_000
    log_interval: int = 10000
    save_interval: int = 1000000

    # Output
    save_dir: str = "results/weights/ppo_direct_coevolution"
    seed: int = 67
    device: str = 'auto'  # 'cpu', 'cuda', or 'auto'


class PPOBuffer:
    """GPU-based rollout buffer for PPO"""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.size = 0

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.size += state.shape[0]

    def get(self):
        return (
            torch.cat(self.states, dim=0),
            torch.cat(self.actions, dim=0),
            torch.cat(self.rewards, dim=0),
            torch.cat(self.dones, dim=0),
            torch.cat(self.log_probs, dim=0),
            torch.cat(self.values, dim=0)
        )


class PPOAgent:
    """PPO agent wrapper"""

    def __init__(self, hidden_dims: Tuple[int, int], actor_lr: float, critic_lr: float,
                 device: torch.device, name: str = "Agent"):
        self.hidden_dims = hidden_dims
        self.device = device
        self.name = name

        self.actor = PPO_Actor_MLP(33, 3, hidden_dims).to(device)
        self.critic = PPO_Critic_MLP(33, hidden_dims).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_scaler = GradScaler()
        self.critic_scaler = GradScaler()

    def select_actions(self, states: torch.Tensor):
        with torch.no_grad():
            logits = self.actor(states)
            values = self.critic(states).squeeze()

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions, log_probs, values


class DirectCoEvolutionTrainer:
    """Direct co-evolution trainer (no curriculum)"""

    def __init__(self, config: DirectCoEvolutionConfig):
        self.config = config
        set_seed(config.seed)

        # Device selection
        if config.device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(config.device)
            print(f"Using device: {self.device}")

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

        # Create agents
        self.agent_256 = PPOAgent(
            config.big_hidden_dims, config.actor_lr, config.critic_lr,
            self.device, "Big-256x256"
        )
        self.agent_128 = PPOAgent(
            config.small_hidden_dims, config.actor_lr, config.critic_lr,
            self.device, "Small-128x128"
        )

        # Buffers
        self.buffer_256 = PPOBuffer(config.rollout_steps, self.device)
        self.buffer_128 = PPOBuffer(config.rollout_steps, self.device)

        # Metrics
        self.total_steps = 0
        self.round_winners = []
        self.scores_256 = []
        self.scores_128 = []
        self.losses_256 = []
        self.losses_128 = []

        # History for plotting
        self.history = []

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        T = rewards.shape[0]
        num_envs = rewards.shape[1] if rewards.dim() > 1 else 1

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
            values = values.unsqueeze(1)
            dones = dones.unsqueeze(1)
            next_value = next_value.unsqueeze(0)

        dones = dones.float()
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(num_envs, device=rewards.device)

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

    def update_agent(self, agent: PPOAgent, states, actions, old_log_probs, advantages, returns):
        """Update a single agent's networks"""
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 0.1:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = torch.clamp(advantages, -10.0, 10.0)
        else:
            advantages = advantages - adv_mean

        total_actor_loss = 0
        total_critic_loss = 0
        n_updates = 0

        for _ in range(self.config.epochs_per_rollout):
            indices = torch.randperm(states.size(0))

            for start in range(0, states.size(0), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                with autocast(device_type=self.device.type):
                    logits = agent.actor(batch_states)
                    values = agent.critic(batch_states).squeeze()

                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()

                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                       1 + self.config.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    critic_loss = F.mse_loss(values, batch_returns)

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
                n_updates += 1

        return total_actor_loss / n_updates, total_critic_loss / n_updates

    def calculate_win_rates(self, window: int = 100):
        """Calculate win rates for both networks"""
        if len(self.round_winners) < window:
            window = len(self.round_winners)
        if window == 0:
            return 0.0, 0.0, 0.0

        recent = self.round_winners[-window:]
        # 256x256 is snake 1, 128x128 is snake 2
        wins_256 = sum(1 for w in recent if w == 1)
        wins_128 = sum(1 for w in recent if w == 2)
        draws = sum(1 for w in recent if w == 3)

        return wins_256 / window, wins_128 / window, draws / window

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70, flush=True)
        print("PPO DIRECT CO-EVOLUTION (NO CURRICULUM)", flush=True)
        print("=" * 70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num environments: {self.config.num_envs}", flush=True)
        print(f"Grid size: {self.config.grid_size}", flush=True)
        print(f"Target food: {self.config.target_food}", flush=True)
        print(f"Agent 1 (Snake 1): 256x256", flush=True)
        print(f"Agent 2 (Snake 2): 128x128", flush=True)
        print(f"Total steps: {self.config.total_steps:,}", flush=True)
        print("=" * 70 + "\n", flush=True)

        start_time = time.time()

        # Reset environment
        obs_256, obs_128 = self.env.reset()

        while self.total_steps < self.config.total_steps:
            self.buffer_256.clear()
            self.buffer_128.clear()

            # Collect rollout
            for _ in range(max(1, self.config.rollout_steps // self.config.num_envs)):
                # Both agents select actions
                actions_256, log_probs_256, values_256 = self.agent_256.select_actions(obs_256)
                actions_128, log_probs_128, values_128 = self.agent_128.select_actions(obs_128)

                # Environment step
                next_obs_256, next_obs_128, r_256, r_128, dones, info = \
                    self.env.step(actions_256, actions_128)

                # Store transitions
                self.buffer_256.add(obs_256, actions_256, r_256, dones, log_probs_256, values_256)
                self.buffer_128.add(obs_128, actions_128, r_128, dones, log_probs_128, values_128)

                # Track completed rounds
                if dones.any():
                    num_done = len(info['done_envs'])
                    for i in range(num_done):
                        winner = info['winners'][i]
                        self.round_winners.append(int(winner))
                        self.scores_256.append(info['food_counts1'][i])
                        self.scores_128.append(info['food_counts2'][i])

                obs_256 = next_obs_256
                obs_128 = next_obs_128
                self.total_steps += self.config.num_envs

                if self.total_steps >= self.config.total_steps:
                    break

            if self.total_steps >= self.config.total_steps:
                break

            # Update both agents
            with torch.no_grad():
                next_value_256 = self.agent_256.critic(obs_256).squeeze()
                next_value_128 = self.agent_128.critic(obs_128).squeeze()

            # Update 256x256 agent
            states_256, actions_256, rewards_256, dones_256, log_probs_256, values_256 = self.buffer_256.get()
            steps_per_env = len(self.buffer_256.states)
            rewards_256 = rewards_256.view(steps_per_env, self.config.num_envs)
            values_256 = values_256.view(steps_per_env, self.config.num_envs)
            dones_256 = dones_256.view(steps_per_env, self.config.num_envs)

            advantages_256, returns_256 = self.compute_gae(rewards_256, values_256, dones_256, next_value_256)
            advantages_256 = advantages_256.view(-1)
            returns_256 = returns_256.view(-1)

            actor_loss_256, critic_loss_256 = self.update_agent(
                self.agent_256, states_256, actions_256, log_probs_256, advantages_256, returns_256
            )
            self.losses_256.append(actor_loss_256 + critic_loss_256)

            # Update 128x128 agent
            states_128, actions_128, rewards_128, dones_128, log_probs_128, values_128 = self.buffer_128.get()
            rewards_128 = rewards_128.view(steps_per_env, self.config.num_envs)
            values_128 = values_128.view(steps_per_env, self.config.num_envs)
            dones_128 = dones_128.view(steps_per_env, self.config.num_envs)

            advantages_128, returns_128 = self.compute_gae(rewards_128, values_128, dones_128, next_value_128)
            advantages_128 = advantages_128.view(-1)
            returns_128 = returns_128.view(-1)

            actor_loss_128, critic_loss_128 = self.update_agent(
                self.agent_128, states_128, actions_128, log_probs_128, advantages_128, returns_128
            )
            self.losses_128.append(actor_loss_128 + critic_loss_128)

            # Logging - check if we crossed the log interval boundary
            steps_this_rollout = steps_per_env * self.config.num_envs
            if (self.total_steps // self.config.log_interval) > ((self.total_steps - steps_this_rollout) // self.config.log_interval):
                win_rate_256, win_rate_128, draw_rate = self.calculate_win_rates(window=100)
                avg_score_256 = np.mean(self.scores_256[-100:]) if self.scores_256 else 0
                avg_score_128 = np.mean(self.scores_128[-100:]) if self.scores_128 else 0
                avg_loss_256 = np.mean(self.losses_256[-10:]) if self.losses_256 else 0
                avg_loss_128 = np.mean(self.losses_128[-10:]) if self.losses_128 else 0

                # Record history for plotting
                self.history.append({
                    'step': self.total_steps,
                    'win_rate_256': win_rate_256,
                    'win_rate_128': win_rate_128,
                    'draw_rate': draw_rate,
                    'avg_score_256': float(avg_score_256),
                    'avg_score_128': float(avg_score_128),
                    'loss_256': float(avg_loss_256),
                    'loss_128': float(avg_loss_128),
                    'episodes_completed': len(self.round_winners)
                })

                elapsed = time.time() - start_time
                fps = self.total_steps / elapsed if elapsed > 0 else 0

                print(f"[Step {self.total_steps:>10,} / {self.config.total_steps:,}] "
                      f"Win: 256={win_rate_256:.1%} 128={win_rate_128:.1%} Draw={draw_rate:.1%} | "
                      f"Score: {avg_score_256:.1f} vs {avg_score_128:.1f} | "
                      f"FPS: {fps:.0f}", flush=True)

            # Save checkpoint
            if self.total_steps % self.config.save_interval == 0:
                self.save_checkpoint()

        total_time = time.time() - start_time

        # Final save
        self.save_checkpoint(is_last=True)
        self.save_history()

        # Final summary
        win_rate_256, win_rate_128, draw_rate = self.calculate_win_rates(window=1000)

        print("\n" + "=" * 70, flush=True)
        print("DIRECT CO-EVOLUTION TRAINING COMPLETE!", flush=True)
        print("=" * 70, flush=True)
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)", flush=True)
        print(f"Total steps: {self.total_steps:,}", flush=True)
        print(f"Total rounds: {len(self.round_winners):,}", flush=True)
        print(f"Final win rates (last 1000):", flush=True)
        print(f"  256x256: {win_rate_256:.2%}", flush=True)
        print(f"  128x128: {win_rate_128:.2%}", flush=True)
        print(f"  Draws: {draw_rate:.2%}", flush=True)
        print(f"Saved to: {self.save_dir}", flush=True)
        print("=" * 70 + "\n", flush=True)

        return {
            'total_steps': self.total_steps,
            'total_time': total_time,
            'final_win_rate_256': win_rate_256,
            'final_win_rate_128': win_rate_128,
            'final_draw_rate': draw_rate,
            'history': self.history
        }

    def save_checkpoint(self, is_last: bool = False):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = "coevo" if is_last else f"step{self.total_steps}"

        # Save 256x256 agent
        path_256 = self.save_dir / f"256x256_{suffix}_{timestamp}.pt"
        torch.save({
            'actor': self.agent_256.actor.state_dict(),
            'critic': self.agent_256.critic.state_dict(),
            'actor_optimizer': self.agent_256.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent_256.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'hidden_dims': (256, 256)
        }, path_256)

        # Save 128x128 agent
        path_128 = self.save_dir / f"128x128_{suffix}_{timestamp}.pt"
        torch.save({
            'actor': self.agent_128.actor.state_dict(),
            'critic': self.agent_128.critic.state_dict(),
            'actor_optimizer': self.agent_128.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent_128.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'hidden_dims': (128, 128)
        }, path_128)

        print(f"Saved checkpoint: {suffix}", flush=True)

    def save_history(self):
        """Save training history to JSON with comprehensive metrics"""
        data_dir = Path('results/data')
        data_dir.mkdir(parents=True, exist_ok=True)

        # Calculate final statistics
        win_rate_256, win_rate_128, draw_rate = self.calculate_win_rates(window=min(1000, len(self.round_winners)))
        avg_score_256 = float(np.mean(self.scores_256[-1000:])) if self.scores_256 else 0
        avg_score_128 = float(np.mean(self.scores_128[-1000:])) if self.scores_128 else 0
        std_score_256 = float(np.std(self.scores_256[-1000:])) if self.scores_256 else 0
        std_score_128 = float(np.std(self.scores_128[-1000:])) if self.scores_128 else 0

        # Use step count in filename for 2M vs 14M differentiation
        if self.config.total_steps >= 1_000_000:
            steps_str = f"{self.config.total_steps // 1_000_000}M"
        else:
            steps_str = f"{self.config.total_steps // 1_000}K"
        history_path = data_dir / f"ppo_direct_coevolution_{steps_str}_history.json"

        with open(history_path, 'w') as f:
            json.dump({
                'total_steps': self.total_steps,
                'total_episodes': len(self.round_winners),
                'history': self.history,
                'final_stats': {
                    'win_rate_256': win_rate_256,
                    'win_rate_128': win_rate_128,
                    'draw_rate': draw_rate,
                    'avg_score_256': avg_score_256,
                    'avg_score_128': avg_score_128,
                    'std_score_256': std_score_256,
                    'std_score_128': std_score_128
                },
                'config': {
                    'total_steps': self.config.total_steps,
                    'target_food': self.config.target_food,
                    'num_envs': self.config.num_envs,
                    'grid_size': self.config.grid_size,
                    'actor_lr': self.config.actor_lr,
                    'critic_lr': self.config.critic_lr,
                    'rollout_steps': self.config.rollout_steps,
                    'batch_size': self.config.batch_size,
                    'epochs_per_rollout': self.config.epochs_per_rollout,
                    'gamma': self.config.gamma,
                    'gae_lambda': self.config.gae_lambda,
                    'clip_epsilon': self.config.clip_epsilon,
                    'entropy_coef': self.config.entropy_coef
                }
            }, f, indent=2)

        print(f"Saved training history: {history_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='PPO Direct Co-evolution Training (No Curriculum)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--total-steps', type=int, default=14_000_000,
                        help='Total training steps')
    parser.add_argument('--num-envs', type=int, default=128,
                        help='Number of parallel environments')
    parser.add_argument('--target-food', type=int, default=10,
                        help='Target food for winning')
    parser.add_argument('--save-dir', type=str, default='results/weights/ppo_direct_coevolution',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=67,
                        help='Random seed')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                        default='auto', help='Device to use (auto detects GPU if available)')

    args = parser.parse_args()

    config = DirectCoEvolutionConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        target_food=args.target_food,
        save_dir=args.save_dir,
        seed=args.seed,
        device=args.device
    )

    trainer = DirectCoEvolutionTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
