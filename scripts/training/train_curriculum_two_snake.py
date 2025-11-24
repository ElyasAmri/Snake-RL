"""
Curriculum Training for Two-Snake Competitive Environment

5-stage curriculum:
1. Stage 0: vs StaticAgent (learn basic movement)
2. Stage 1: vs RandomAgent (handle unpredictability)
3. Stage 2: vs GreedyFoodAgent (compete for food)
4. Stage 3: vs Frozen small network (compete against policy)
5. Stage 4: Co-evolution (both networks learning)

Trains two networks:
- Agent1 (Big): 256x256 network
- Agent2 (Small): 128x128 network
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # For FP16 mixed precision training
import numpy as np
import time
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from datetime import datetime

from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from core.networks import DQN_MLP
from core.utils import ReplayBuffer, EpsilonScheduler, set_seed, get_device
from scripts.baselines.scripted_opponents import get_scripted_agent


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage"""
    stage_id: int
    name: str
    opponent_type: str  # 'static', 'random', 'greedy', 'defensive', 'frozen', 'learning'
    min_steps: int
    win_rate_threshold: Optional[float]  # None for final stage
    description: str
    target_food: int = 10  # Food needed to win (progressive difficulty)
    agent2_trains: bool = False  # Whether agent2 trains in this stage


class TwoSnakeDQN:
    """Simplified DQN agent for competitive training"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple,
        output_dim: int,
        learning_rate: float,
        gamma: float,
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

        # FP16 Mixed Precision: GradScaler for automatic loss scaling
        self.scaler = GradScaler()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=50000, seed=42)

        # Epsilon scheduler
        # FIXED: Extended decay from 5000 to 52500 to cover full curriculum
        # This ensures exploration throughout all stages, not just Stage 0
        self.epsilon_scheduler = EpsilonScheduler(
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=52500,  # Match total curriculum steps (5K+7.5K+10K+10K+20K)
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

        # FP16 Mixed Precision: Wrap forward pass in autocast
        with autocast():
            # Compute Q values
            q_values = self.policy_net(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(dim=1)[0]
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

            # Compute loss
            loss = nn.functional.mse_loss(q_values, target_q_values)

        # FP16 Mixed Precision: Scale loss and backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.training_steps += 1
        return loss.item()

    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_epsilon(self, epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: int = 10000):
        """Reset epsilon scheduler for new curriculum stage"""
        self.epsilon_scheduler = EpsilonScheduler(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            decay_type='linear'
        )
        print(f"  >> {self.name}: Reset epsilon ({epsilon_start:.2f} -> {epsilon_end:.2f} over {epsilon_decay} steps)", flush=True)

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


class CurriculumTrainer:
    """Orchestrates curriculum training for competitive two-snake RL"""

    def __init__(
        self,
        num_envs: int = 128,
        grid_size: int = 20,  # FIXED: Changed from 10 to 20 to match PPO success
        target_food: int = 10,
        max_steps: int = 1000,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        train_steps_ratio: float = 0.125,  # OPTIMIZED: Reduced from 0.25 (train every 8th step instead of 4th)
        log_interval: int = 100,
        save_dir: str = 'results/weights/competitive',
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        """Initialize curriculum trainer"""
        set_seed(seed)
        self.device = device if device else get_device()

        # Config
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.target_food = target_food
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps_ratio = train_steps_ratio
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Environment
        self.env = VectorizedTwoSnakeEnv(
            num_envs=num_envs,
            grid_size=grid_size,
            max_steps=max_steps,
            target_food=target_food,
            device=self.device
        )

        # Agents
        self.agent1 = TwoSnakeDQN(
            input_dim=35,
            hidden_dims=(256, 256),
            output_dim=3,
            learning_rate=learning_rate,
            gamma=gamma,
            device=self.device,
            name="Big-256x256"
        )

        self.agent2 = TwoSnakeDQN(
            input_dim=35,
            hidden_dims=(128, 128),
            output_dim=3,
            learning_rate=learning_rate,
            gamma=gamma,
            device=self.device,
            name="Small-128x128"
        )

        # Scripted opponents
        self.scripted_agents = {}
        for agent_type in ['static', 'random', 'greedy', 'defensive']:
            try:
                self.scripted_agents[agent_type] = get_scripted_agent(
                    agent_type, device=self.device
                )
            except Exception as e:
                print(f"Warning: Could not load {agent_type} agent: {e}", flush=True)

        # Curriculum stages (OPTIMIZED: Reduced from 800K to 52.5K total steps based on empirical testing)
        self.stages = [
            CurriculumStage(
                stage_id=0,
                name="Stage0_Static",
                opponent_type="static",
                min_steps=5000,  # OPTIMIZED: Reduced 75% from 20K (originally 50K)
                win_rate_threshold=0.70,
                description="Learn basic movement vs static opponent",
                target_food=10,  # Easy: static opponent doesn't move
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=1,
                name="Stage1_Random",
                opponent_type="random",
                min_steps=7500,  # OPTIMIZED: Reduced 62.5% from 20K (originally 50K)
                win_rate_threshold=0.60,
                description="Handle unpredictable opponent",
                target_food=10,  # Medium: random opponent is inefficient
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=2,
                name="Stage2_Greedy",
                opponent_type="greedy",
                min_steps=10000,  # OPTIMIZED: Reduced 67% from 30K (originally 100K)
                win_rate_threshold=0.55,
                description="Compete for food against optimal food-seeker",
                target_food=4,  # FIXED: Reduced from 10 (greedy opponent is very effective)
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=3,
                name="Stage3_Frozen",
                opponent_type="frozen",
                min_steps=10000,  # OPTIMIZED: Reduced 67% from 30K (originally 100K)
                win_rate_threshold=0.50,
                description="Face frozen small network",
                target_food=6,  # FIXED: Progressive difficulty (harder than greedy)
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=4,
                name="Stage4_CoEvolution",
                opponent_type="learning",
                min_steps=20000,  # OPTIMIZED: Reduced 87% from 150K (originally 500K)
                win_rate_threshold=None,
                description="Both networks learning simultaneously",
                target_food=8,  # FIXED: Progressive difficulty (both learning together)
                agent2_trains=True
            )
        ]

        # Metrics
        self.current_stage_idx = 0
        self.stage_steps = 0
        self.total_steps = 0
        self.total_rounds = 0
        self.round_winners = []  # Track winners for win rate calculation
        self.stage_metrics = {
            'losses1': [],
            'losses2': [],
            'scores1': [],
            'scores2': [],
            'episode_lengths': []
        }

    def get_actions(self, obs1: torch.Tensor, obs2: torch.Tensor, stage: CurriculumStage) -> tuple:
        """Get actions for both agents based on current stage"""
        # Agent1 always uses its policy
        actions1 = self.agent1.select_action(obs1)

        # Agent2 action depends on stage
        if stage.opponent_type == 'learning':
            # Both learning
            actions2 = self.agent2.select_action(obs2)
        elif stage.opponent_type == 'frozen':
            # Agent2 uses greedy policy (no exploration)
            actions2 = self.agent2.select_action(obs2, greedy=True)
        elif stage.opponent_type in self.scripted_agents:
            # Use scripted opponent
            actions2 = self.scripted_agents[stage.opponent_type].select_action(self.env)
        else:
            # Fallback: random
            actions2 = torch.randint(0, 3, (self.num_envs,), device=self.device)

        return actions1, actions2

    def calculate_win_rate(self, window: int = 100) -> float:
        """Calculate agent1 win rate over last N rounds"""
        if len(self.round_winners) < window:
            window = len(self.round_winners)

        if window == 0:
            return 0.0

        recent_winners = self.round_winners[-window:]
        snake1_wins = sum(1 for w in recent_winners if w == 1)
        return snake1_wins / window

    def should_advance_stage(self, stage: CurriculumStage) -> bool:
        """Check if should advance to next stage"""
        # Check minimum steps
        if self.stage_steps < stage.min_steps:
            return False

        # Check win rate threshold (if specified)
        if stage.win_rate_threshold is None:
            return False  # Final stage, run until manually stopped

        win_rate = self.calculate_win_rate(window=100)
        return win_rate >= stage.win_rate_threshold

    def train_stage(self, stage: CurriculumStage):
        """Train one curriculum stage"""
        print("\n" + "="*70, flush=True)
        print(f"STAGE {stage.stage_id}: {stage.name}", flush=True)
        print("="*70, flush=True)
        print(f"Opponent: {stage.opponent_type}", flush=True)
        print(f"Min steps: {stage.min_steps}", flush=True)
        print(f"Win rate threshold: {stage.win_rate_threshold}", flush=True)
        print(f"Target food: {stage.target_food}", flush=True)
        print(f"Description: {stage.description}", flush=True)
        print("="*70 + "\n", flush=True)

        # Update environment's target_food for this stage
        self.env.target_food = stage.target_food

        # FIXED: Reset epsilon for each stage to enable exploration
        # Agent1 always trains, so always reset its epsilon
        self.agent1.reset_epsilon(
            epsilon_start=0.8,  # Start with strong exploration
            epsilon_end=0.01,
            epsilon_decay=stage.min_steps
        )

        # Agent2 only resets epsilon if it's training in this stage
        if stage.agent2_trains:
            self.agent2.reset_epsilon(
                epsilon_start=0.8,
                epsilon_end=0.01,
                epsilon_decay=stage.min_steps
            )

        # Reset stage metrics
        self.stage_steps = 0
        self.stage_metrics = {
            'losses1': [],
            'losses2': [],
            'scores1': [],
            'scores2': [],
            'episode_lengths': []
        }

        # Reset environment
        obs1, obs2 = self.env.reset()

        # PHASE 1 PROFILING: Initialize timing stats
        timing_stats = {
            'get_actions': [],
            'env_step': [],
            'store_transitions': [],
            'training': [],
            'other': []
        }

        # Main training loop
        while not self.should_advance_stage(stage):
            # Get actions
            t0 = time.perf_counter()
            actions1, actions2 = self.get_actions(obs1, obs2, stage)
            timing_stats['get_actions'].append(time.perf_counter() - t0)

            # Environment step
            t0 = time.perf_counter()
            next_obs1, next_obs2, r1, r2, dones, info = self.env.step(actions1, actions2)
            timing_stats['env_step'].append(time.perf_counter() - t0)

            # Store transitions for agent1 (always training)
            t0 = time.perf_counter()
            states1_np = obs1.cpu().numpy()
            actions1_np = actions1.cpu().numpy()
            rewards1_np = r1.cpu().numpy()
            next_states1_np = next_obs1.cpu().numpy()
            dones_np = dones.cpu().numpy()

            for i in range(self.num_envs):
                self.agent1.replay_buffer.push(
                    states1_np[i],
                    actions1_np[i],
                    rewards1_np[i],
                    next_states1_np[i],
                    dones_np[i]
                )

            # Store transitions for agent2 (if training in this stage)
            if stage.agent2_trains:
                states2_np = obs2.cpu().numpy()
                actions2_np = actions2.cpu().numpy()
                rewards2_np = r2.cpu().numpy()
                next_states2_np = next_obs2.cpu().numpy()

                for i in range(self.num_envs):
                    self.agent2.replay_buffer.push(
                        states2_np[i],
                        actions2_np[i],
                        rewards2_np[i],
                        next_states2_np[i],
                        dones_np[i]
                    )
            timing_stats['store_transitions'].append(time.perf_counter() - t0)

            # Control training frequency
            should_train = (self.total_steps % max(1, int(1 / self.train_steps_ratio)) == 0)

            # Train agent1 (only when should_train is True)
            t0 = time.perf_counter()
            if should_train:
                loss1 = self.agent1.train_step(self.batch_size)
                if loss1 is not None:
                    self.stage_metrics['losses1'].append(loss1)

            # Train agent2 (if applicable and should_train is True)
            if stage.agent2_trains and should_train:
                loss2 = self.agent2.train_step(self.batch_size)
                if loss2 is not None:
                    self.stage_metrics['losses2'].append(loss2)
            timing_stats['training'].append(time.perf_counter() - t0)

            # Update target networks
            if self.total_steps % self.target_update_freq == 0:
                self.agent1.update_target_network()
                if stage.agent2_trains:
                    self.agent2.update_target_network()

            # Update epsilon schedulers
            self.agent1.epsilon_scheduler.step()
            if stage.agent2_trains:
                self.agent2.epsilon_scheduler.step()

            # Track completed rounds
            if dones.any():
                # Use info dict to get winners (saved before auto-reset)
                num_done = len(info['done_envs'])
                for i in range(num_done):
                    env_idx = info['done_envs'][i]
                    winner = info['winners'][i]
                    self.round_winners.append(int(winner))
                    self.total_rounds += 1

                    # Track metrics
                    self.stage_metrics['scores1'].append(info['food_counts1'][i])
                    self.stage_metrics['scores2'].append(info['food_counts2'][i])

            # Update state
            obs1 = next_obs1
            obs2 = next_obs2

            # Increment counters
            self.stage_steps += 1
            self.total_steps += 1
            self.agent1.total_steps += 1
            if stage.agent2_trains:
                self.agent2.total_steps += 1

            # Logging
            if self.total_steps % self.log_interval == 0:
                # PHASE 1 PROFILING: Print timing stats
                print("\n" + "="*70, flush=True)
                print(f"TIMING ANALYSIS (Step {self.total_steps})", flush=True)
                print("="*70, flush=True)
                for key, times in timing_stats.items():
                    if times:
                        # Use last log_interval samples
                        recent = times[-self.log_interval:]
                        avg_ms = np.mean(recent) * 1000
                        total_ms = np.sum(recent) * 1000
                        print(f"{key:20s}: {avg_ms:8.2f} ms avg | {total_ms:10.1f} ms total", flush=True)
                print("="*70 + "\n", flush=True)

                self.log_progress(stage)

        # Save checkpoint at end of stage
        self.save_checkpoint(stage.name)
        print(f"\n[OK] Stage {stage.stage_id} complete!", flush=True)
        print(f"  Final win rate: {self.calculate_win_rate():.2%}", flush=True)
        print(f"  Total steps in stage: {self.stage_steps}", flush=True)

    def log_progress(self, stage: CurriculumStage):
        """Log training progress"""
        win_rate = self.calculate_win_rate(window=100)

        # Calculate average metrics
        avg_loss1 = np.mean(self.stage_metrics['losses1'][-100:]) if self.stage_metrics['losses1'] else 0
        avg_score1 = np.mean(self.stage_metrics['scores1'][-100:]) if self.stage_metrics['scores1'] else 0
        avg_score2 = np.mean(self.stage_metrics['scores2'][-100:]) if self.stage_metrics['scores2'] else 0

        epsilon1 = self.agent1.epsilon_scheduler.get_epsilon()

        print(f"[Step {self.total_steps:>6}] "
              f"Stage: {stage.stage_id} | "
              f"Win Rate: {win_rate:.2%} | "
              f"Loss: {avg_loss1:.4f} | "
              f"Scores: {avg_score1:.1f} vs {avg_score2:.1f} | "
              f"Epsilon: {epsilon1:.3f}", flush=True)

    def save_checkpoint(self, stage_name: str):
        """Save checkpoint after each stage"""
        checkpoint_dir = self.save_dir / stage_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save agent1 (big network)
        agent1_path = checkpoint_dir / f"big_256x256_{timestamp}.pt"
        self.agent1.save(str(agent1_path))
        print(f"Saved agent1: {agent1_path}", flush=True)

        # Save agent2 (small network)
        agent2_path = checkpoint_dir / f"small_128x128_{timestamp}.pt"
        self.agent2.save(str(agent2_path))
        print(f"Saved agent2: {agent2_path}", flush=True)

        # Save stage metrics
        metrics_path = checkpoint_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'stage': asdict(self.stages[self.current_stage_idx]),
                'total_steps': self.total_steps,
                'stage_steps': self.stage_steps,
                'total_rounds': self.total_rounds,
                'final_win_rate': self.calculate_win_rate(),
                'avg_loss1': float(np.mean(self.stage_metrics['losses1'])) if self.stage_metrics['losses1'] else 0,
                'avg_score1': float(np.mean(self.stage_metrics['scores1'])) if self.stage_metrics['scores1'] else 0,
                'avg_score2': float(np.mean(self.stage_metrics['scores2'])) if self.stage_metrics['scores2'] else 0
            }, f, indent=2)
        print(f"Saved metrics: {metrics_path}", flush=True)

    def train(self):
        """Run full curriculum training"""
        print("\n" + "="*70, flush=True)
        print("TWO-SNAKE COMPETITIVE CURRICULUM TRAINING", flush=True)
        print("="*70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num environments: {self.num_envs}", flush=True)
        print(f"Grid size: {self.grid_size}", flush=True)
        print(f"Target food: {self.target_food}", flush=True)
        print(f"Agent1 (Big): 256x256 network", flush=True)
        print(f"Agent2 (Small): 128x128 network", flush=True)
        print(f"Total curriculum stages: {len(self.stages)}", flush=True)
        print("="*70, flush=True)

        start_time = time.time()

        # Train each stage
        for stage_idx, stage in enumerate(self.stages):
            self.current_stage_idx = stage_idx
            self.train_stage(stage)

            # For final stage, train for min_steps then stop
            if stage.win_rate_threshold is None:
                break

        total_time = time.time() - start_time

        # Final summary
        print("\n" + "="*70, flush=True)
        print("TRAINING COMPLETE!", flush=True)
        print("="*70, flush=True)
        print(f"Total time: {total_time/60:.1f} minutes", flush=True)
        print(f"Total steps: {self.total_steps}", flush=True)
        print(f"Total rounds: {self.total_rounds}", flush=True)
        print(f"Final win rate: {self.calculate_win_rate():.2%}", flush=True)
        print(f"Models saved to: {self.save_dir}", flush=True)
        print("="*70 + "\n", flush=True)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Two-Snake Competitive Curriculum Training')
    parser.add_argument('--num-envs', type=int, default=128, help='Number of parallel environments')
    parser.add_argument('--grid-size', type=int, default=20, help='Grid size (FIXED: changed from 10 to 20)')
    parser.add_argument('--target-food', type=int, default=10, help='Food needed to win')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--train-steps-ratio', type=float, default=0.25, help='Training steps per collected transition (0.03125=fast, 0.25=balanced, 0.5=quality)')
    parser.add_argument('--save-dir', type=str, default='results/weights/competitive', help='Save directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')

    args = parser.parse_args()

    # Create trainer
    trainer = CurriculumTrainer(
        num_envs=args.num_envs,
        grid_size=args.grid_size,
        target_food=args.target_food,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_steps_ratio=args.train_steps_ratio,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        seed=args.seed
    )

    # Run training
    trainer.train()


if __name__ == '__main__':
    main()
