"""
DQN Training for Two-Snake Competitive Environment (MLP-based)

Direct co-evolution training (NO curriculum):
- Both agents (256x256 and 128x128) learn simultaneously from start
- Uses vectorized two-snake environment (128 parallel games)
- 33-dimensional feature vector state representation
- Comparable to train_ppo_two_snake_mlp.py for fair comparison

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
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from core.networks import DQN_MLP
from core.utils import ReplayBuffer, EpsilonScheduler, set_seed, get_device


@dataclass
class DQNConfig:
    """Configuration for DQN two-snake training"""
    # Environment
    num_envs: int = 128
    grid_size: int = 20
    target_food: int = 10
    max_steps: int = 1000

    # Network sizes
    big_hidden_dims: Tuple[int, int] = (256, 256)
    small_hidden_dims: Tuple[int, int] = (128, 128)

    # DQN hyperparameters
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 50000  # Steps to decay epsilon
    target_update_freq: int = 1000
    train_steps_ratio: float = 0.125  # Train every 8th step

    # Training
    total_steps: int = 1000000  # Sized for ~10 min (~30 min total with PPO)
    log_interval: int = 1000
    save_interval: int = 10000

    # Other
    seed: int = 67
    save_dir: str = 'results/weights/dqn_two_snake_mlp'
    max_time: Optional[int] = None


class TwoSnakeDQN:
    """DQN agent for competitive two-snake training"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple,
        output_dim: int,
        learning_rate: float,
        gamma: float,
        buffer_size: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: int,
        device: torch.device,
        name: str = "Agent"
    ):
        self.device = device
        self.name = name
        self.gamma = gamma

        # Networks
        self.policy_net = DQN_MLP(input_dim, output_dim, hidden_dims).to(device)
        self.target_net = DQN_MLP(input_dim, output_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # FP16 Mixed Precision
        self.scaler = GradScaler()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=67)

        # Epsilon scheduler
        self.epsilon_scheduler = EpsilonScheduler(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            decay_type='linear'
        )

        # Stats
        self.total_steps = 0
        self.training_steps = 0

    def select_action(self, states: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        """Select actions using epsilon-greedy or fully greedy"""
        with torch.no_grad():
            q_values = self.policy_net(states)
            greedy_actions = q_values.argmax(dim=1)

        if greedy:
            return greedy_actions

        # Epsilon-greedy
        epsilon = self.epsilon_scheduler.get_epsilon()
        num_envs = states.shape[0]
        random_mask = torch.rand(num_envs, device=self.device) < epsilon
        random_actions = torch.randint(0, 3, (num_envs,), device=self.device)
        actions = torch.where(random_mask, random_actions, greedy_actions)

        return actions

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """Perform one training step"""
        if not self.replay_buffer.is_ready(batch_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # FP16 Mixed Precision
        with autocast():
            q_values = self.policy_net(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(dim=1)[0]
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

            loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.training_steps += 1
        return loss.item()

    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'training_steps': self.training_steps,
            'epsilon': self.epsilon_scheduler.get_epsilon()
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.training_steps = checkpoint.get('training_steps', 0)


class TwoSnakeDQNTrainer:
    """Direct co-evolution trainer for two-snake DQN (no curriculum)"""

    def __init__(self, config: DQNConfig):
        set_seed(config.seed)
        self.config = config
        self.device = get_device()

        # Save directory
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
        self.agent1 = TwoSnakeDQN(
            input_dim=33,
            hidden_dims=config.big_hidden_dims,
            output_dim=3,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            buffer_size=config.buffer_size,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            device=self.device,
            name="Big-256x256"
        )

        self.agent2 = TwoSnakeDQN(
            input_dim=33,
            hidden_dims=config.small_hidden_dims,
            output_dim=3,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            buffer_size=config.buffer_size,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            device=self.device,
            name="Small-128x128"
        )

        # Metrics tracking
        self.total_steps = 0
        self.total_rounds = 0
        self.round_winners = []
        self.scores1 = []
        self.scores2 = []
        self.losses1 = []
        self.losses2 = []
        self.win_rate_history = []

    def calculate_win_rate(self, window: int = 100) -> float:
        """Calculate agent1 win rate over last N rounds"""
        if len(self.round_winners) < window:
            window = len(self.round_winners)
        if window == 0:
            return 0.0

        recent = self.round_winners[-window:]
        return sum(1 for w in recent if w == 1) / window

    def train(self):
        """Run direct co-evolution training"""
        print("\n" + "=" * 70, flush=True)
        print("DQN TWO-SNAKE DIRECT CO-EVOLUTION", flush=True)
        print("=" * 70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num environments: {self.config.num_envs}", flush=True)
        print(f"Grid size: {self.config.grid_size}x{self.config.grid_size}", flush=True)
        print(f"Total steps: {self.config.total_steps:,}", flush=True)
        print(f"Agent1 (Big): {self.config.big_hidden_dims}", flush=True)
        print(f"Agent2 (Small): {self.config.small_hidden_dims}", flush=True)
        print("=" * 70 + "\n", flush=True)

        start_time = time.time()

        # Reset environment
        obs1, obs2 = self.env.reset()

        # Training loop
        while self.total_steps < self.config.total_steps:
            # Check max time
            if self.config.max_time and (time.time() - start_time) >= self.config.max_time:
                print(f"\n[TIMEOUT] Max time {self.config.max_time}s reached.", flush=True)
                break

            # Get actions from both agents
            actions1 = self.agent1.select_action(obs1)
            actions2 = self.agent2.select_action(obs2)

            # Environment step
            next_obs1, next_obs2, r1, r2, dones, info = self.env.step(actions1, actions2)

            # Store transitions for both agents
            states1_np = obs1.cpu().numpy()
            actions1_np = actions1.cpu().numpy()
            rewards1_np = r1.cpu().numpy()
            next_states1_np = next_obs1.cpu().numpy()
            dones_np = dones.cpu().numpy()

            states2_np = obs2.cpu().numpy()
            actions2_np = actions2.cpu().numpy()
            rewards2_np = r2.cpu().numpy()
            next_states2_np = next_obs2.cpu().numpy()

            for i in range(self.config.num_envs):
                self.agent1.replay_buffer.push(
                    states1_np[i], actions1_np[i], rewards1_np[i],
                    next_states1_np[i], dones_np[i]
                )
                self.agent2.replay_buffer.push(
                    states2_np[i], actions2_np[i], rewards2_np[i],
                    next_states2_np[i], dones_np[i]
                )

            # Training step (controlled frequency)
            should_train = (self.total_steps % max(1, int(1 / self.config.train_steps_ratio)) == 0)
            if should_train:
                loss1 = self.agent1.train_step(self.config.batch_size)
                loss2 = self.agent2.train_step(self.config.batch_size)
                if loss1 is not None:
                    self.losses1.append(loss1)
                if loss2 is not None:
                    self.losses2.append(loss2)

            # Update target networks
            if self.total_steps % self.config.target_update_freq == 0:
                self.agent1.update_target_network()
                self.agent2.update_target_network()

            # Update epsilon
            self.agent1.epsilon_scheduler.step()
            self.agent2.epsilon_scheduler.step()

            # Track completed rounds
            if dones.any():
                num_done = len(info['done_envs'])
                for i in range(num_done):
                    winner = info['winners'][i]
                    self.round_winners.append(int(winner))
                    self.total_rounds += 1
                    self.scores1.append(info['food_counts1'][i])
                    self.scores2.append(info['food_counts2'][i])

            # Update state
            obs1 = next_obs1
            obs2 = next_obs2
            self.total_steps += self.config.num_envs
            self.agent1.total_steps += self.config.num_envs
            self.agent2.total_steps += self.config.num_envs

            # Logging
            if self.total_steps % self.config.log_interval == 0:
                win_rate = self.calculate_win_rate(100)
                self.win_rate_history.append({
                    'step': self.total_steps,
                    'win_rate': win_rate
                })

                avg_loss1 = np.mean(self.losses1[-100:]) if self.losses1 else 0
                avg_score1 = np.mean(self.scores1[-100:]) if self.scores1 else 0
                avg_score2 = np.mean(self.scores2[-100:]) if self.scores2 else 0
                epsilon = self.agent1.epsilon_scheduler.get_epsilon()

                print(f"[Step {self.total_steps:>6}] "
                      f"Win Rate: {win_rate:.2%} | "
                      f"Scores: {avg_score1:.1f} vs {avg_score2:.1f} | "
                      f"Loss: {avg_loss1:.4f} | "
                      f"Epsilon: {epsilon:.3f}", flush=True)

            # Periodic save
            if self.total_steps % self.config.save_interval == 0:
                self.save_checkpoint(f"step{self.total_steps}")

        total_time = time.time() - start_time

        # Final save
        self.save_checkpoint("final")

        # Summary
        print("\n" + "=" * 70, flush=True)
        print("TRAINING COMPLETE!", flush=True)
        print("=" * 70, flush=True)
        print(f"Total time: {total_time / 60:.1f} minutes", flush=True)
        print(f"Total steps: {self.total_steps:,}", flush=True)
        print(f"Total rounds: {self.total_rounds:,}", flush=True)
        print(f"Final win rate (Agent1): {self.calculate_win_rate():.2%}", flush=True)
        print(f"Saved to: {self.save_dir}", flush=True)
        print("=" * 70 + "\n", flush=True)

        return self.get_results()

    def save_checkpoint(self, suffix: str):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save agent1
        agent1_path = self.save_dir / f"big_256x256_{suffix}_{timestamp}.pt"
        self.agent1.save(str(agent1_path))

        # Save agent2
        agent2_path = self.save_dir / f"small_128x128_{suffix}_{timestamp}.pt"
        self.agent2.save(str(agent2_path))

        print(f"Saved checkpoint: {suffix}", flush=True)

    def get_results(self) -> Dict:
        """Get training results for plotting/reporting"""
        return {
            'algorithm': 'DQN',
            'curriculum': False,
            'total_steps': self.total_steps,
            'total_rounds': self.total_rounds,
            'final_win_rate': self.calculate_win_rate(),
            'win_rate_history': self.win_rate_history,
            'avg_score1': float(np.mean(self.scores1[-100:])) if self.scores1 else 0,
            'avg_score2': float(np.mean(self.scores2[-100:])) if self.scores2 else 0,
            'scores1': self.scores1,
            'scores2': self.scores2,
        }


def main():
    parser = argparse.ArgumentParser(
        description='DQN Two-Snake Direct Co-evolution Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num-envs', type=int, default=128,
                        help='Number of parallel environments')
    parser.add_argument('--total-steps', type=int, default=52500,
                        help='Total training steps')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str,
                        default='results/weights/dqn_two_snake_mlp',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=67,
                        help='Random seed')
    parser.add_argument('--max-time', type=int, default=None,
                        help='Maximum training time in seconds')

    args = parser.parse_args()

    config = DQNConfig(
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        seed=args.seed,
        max_time=args.max_time
    )

    trainer = TwoSnakeDQNTrainer(config)
    results = trainer.train()

    # Save results
    results_path = Path(config.save_dir) / 'results.json'
    with open(results_path, 'w') as f:
        # Convert win_rate_history to serializable format
        results_serializable = {k: v for k, v in results.items() if k != 'win_rate_history'}
        results_serializable['win_rate_history'] = results['win_rate_history']
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to: {results_path}", flush=True)


if __name__ == '__main__':
    main()
