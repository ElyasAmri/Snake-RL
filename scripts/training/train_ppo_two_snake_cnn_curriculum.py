"""
PPO Curriculum Training for Two-Snake Competitive Environment (CNN-based)

5-stage curriculum with CNN networks using grid representation.

Curriculum stages:
1. Stage 0: vs StaticAgent (learn basic movement, 20K steps, 70% win threshold)
2. Stage 1: vs RandomAgent (handle unpredictability, 20K steps, 60% win threshold)
3. Stage 2: vs GreedyFoodAgent (compete for food, 30K steps, 55% win threshold)
4. Stage 3: vs Frozen small network (30K steps, 50% win threshold)
5. Stage 4: Co-evolution (both learning, 150K steps, no threshold)

Uses 5-channel grids: self_head, self_body, opp_head, opp_body, food.
Expected to be 5-10x faster than MLP-based curriculum.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import argparse
from typing import Tuple
from dataclasses import dataclass
from datetime import datetime

from train_ppo_two_snake_mlp_curriculum import CurriculumStage, CurriculumPPOTrainer as BaseCurriculumTrainer
from train_ppo_two_snake_cnn import CNNAgent
from train_ppo_two_snake_mlp import PPOConfig, PPOBuffer
from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from core.state_representations_competitive import CompetitiveGridEncoder
from core.utils import set_seed, get_device
from scripts.baselines.scripted_opponents import get_scripted_agent


class CNNCurriculumTrainer(BaseCurriculumTrainer):
    """
    Extends curriculum trainer to use CNN networks with grid encoding.

    Combines curriculum learning strategy with CNN-based state representation.
    """

    def __init__(self, config: PPOConfig):
        # Initialize without calling parent __init__
        self.config = config
        set_seed(config.seed)
        self.device = get_device()

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

        # Scripted opponents
        self.scripted_agents = {}
        for agent_type in ['static', 'random', 'greedy']:
            try:
                self.scripted_agents[agent_type] = get_scripted_agent(
                    agent_type, device=self.device
                )
            except Exception as e:
                print(f"Warning: Could not load {agent_type} agent: {e}", flush=True)

        # Curriculum stages with progressive difficulty
        self.stages = [
            CurriculumStage(
                stage_id=0,
                name="Stage0_Static",
                opponent_type="static",
                target_food=10,  # Easy: static opponent doesn't move
                min_steps=5000,  # OPTIMIZED: Reduced 75% (was 20K)
                win_rate_threshold=0.70,
                description="Learn basic movement vs static opponent (target: 10 food)",
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=1,
                name="Stage1_Random",
                opponent_type="random",
                target_food=10,  # Medium: random opponent is inefficient
                min_steps=7500,  # OPTIMIZED: Reduced 62.5% (was 20K)
                win_rate_threshold=0.60,
                description="Handle unpredictability vs random opponent (target: 10 food)",
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=2,
                name="Stage2_Greedy",
                opponent_type="greedy",
                target_food=4,  # FIXED: Reduced from 10 (greedy opponent is very effective)
                min_steps=10000,  # OPTIMIZED: Reduced 67% (was 30K)
                win_rate_threshold=0.55,
                description="Compete for food vs greedy opponent (target: 4 food)",
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=3,
                name="Stage3_Frozen",
                opponent_type="frozen",
                target_food=6,  # FIXED: Progressive difficulty (harder than greedy)
                min_steps=10000,  # OPTIMIZED: Reduced 67% (was 30K)
                win_rate_threshold=0.50,
                description="Compete against frozen policy (target: 6 food)",
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=4,
                name="Stage4_CoEvolution",
                opponent_type="learning",
                target_food=8,  # FIXED: Progressive difficulty (both learning together)
                min_steps=20000,  # OPTIMIZED: Reduced 87% (was 150K)
                win_rate_threshold=None,
                description="Full co-evolution training (target: 8 food)",
                agent2_trains=True
            )
        ]

        # Buffers
        self.buffer1 = PPOBuffer(config.rollout_steps, self.device)
        self.buffer2 = PPOBuffer(config.rollout_steps, self.device)

        # Metrics
        self.total_steps = 0
        self.stage_steps = 0
        self.current_stage_idx = 0
        self.total_rounds = 0
        self.round_winners = []
        self.scores1 = []
        self.scores2 = []
        self.losses1 = []
        self.losses2 = []

    def get_observations(self):
        """Get grid-based observations for both snakes"""
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

    def get_stage_actions(self, obs1, obs2, stage):
        """Get actions based on curriculum stage (overrides parent)"""
        # Agent1 always learns
        actions1, log_probs1, values1 = self.agent1.select_actions(obs1)

        # Agent2 behavior depends on stage
        if stage.opponent_type == 'learning':
            # Co-evolution: both learn
            actions2, log_probs2, values2 = self.agent2.select_actions(obs2)

        elif stage.opponent_type == 'frozen':
            # Frozen policy: greedy actions, no training
            actions2 = self.agent2.select_greedy_actions(obs2)
            log_probs2 = torch.zeros(self.config.num_envs, device=self.device)
            values2 = torch.zeros(self.config.num_envs, device=self.device)

        elif stage.opponent_type in self.scripted_agents:
            # Scripted agent
            actions2 = self.scripted_agents[stage.opponent_type].select_action(self.env)
            log_probs2 = torch.zeros(self.config.num_envs, device=self.device)
            values2 = torch.zeros(self.config.num_envs, device=self.device)

        else:
            # Fallback: random
            actions2 = torch.randint(0, 3, (self.config.num_envs,), device=self.device)
            log_probs2 = torch.zeros(self.config.num_envs, device=self.device)
            values2 = torch.zeros(self.config.num_envs, device=self.device)

        return actions1, log_probs1, values1, actions2, log_probs2, values2

    def train_curriculum_stage(self, stage):
        """Train a single curriculum stage with CNN (overrides parent)"""
        import time
        import numpy as np

        print("\n" + "="*70, flush=True)
        print(f"STAGE {stage.stage_id}: {stage.name}", flush=True)
        print("="*70, flush=True)
        print(f"Opponent: {stage.opponent_type}", flush=True)
        print(f"Target food: {stage.target_food}", flush=True)
        print(f"Min steps: {stage.min_steps:,}", flush=True)
        print(f"Win rate threshold: {stage.win_rate_threshold}", flush=True)
        print(f"Description: {stage.description}", flush=True)
        print(f"Agent2 trains: {stage.agent2_trains}", flush=True)
        print("="*70 + "\n", flush=True)

        # Reset stage tracking
        self.stage_steps = 0
        stage_start_time = time.time()

        # Set target food for this stage
        self.env.set_target_food(stage.target_food)

        # Reset environment
        self.env.reset()

        # Stage training loop
        while not self.should_advance_stage(stage):
            # Clear buffers
            self.buffer1.clear()
            if stage.agent2_trains:
                self.buffer2.clear()

            # Collect rollout
            for _ in range(max(1, self.config.rollout_steps // self.config.num_envs)):
                # Get grid observations
                obs1, obs2 = self.get_observations()

                # Get actions (stage-dependent)
                actions1, log_probs1, values1, actions2, log_probs2, values2 = \
                    self.get_stage_actions(obs1, obs2, stage)

                # Environment step
                _, _, r1, r2, dones, info = self.env.step(actions1, actions2)

                # Store transitions
                self.buffer1.add(obs1, actions1, r1, dones, log_probs1, values1)
                if stage.agent2_trains:
                    self.buffer2.add(obs2, actions2, r2, dones, log_probs2, values2)

                # Track metrics
                if dones.any():
                    # Use info dict to get winners (saved before auto-reset)
                    num_done = len(info['done_envs'])
                    for i in range(num_done):
                        self.round_winners.append(int(info['winners'][i]))
                        self.total_rounds += 1
                        self.scores1.append(info['food_counts1'][i])
                        self.scores2.append(info['food_counts2'][i])

                # FIX: Count all parallel environment steps, not just iterations
                self.stage_steps += self.config.num_envs
                self.total_steps += self.config.num_envs

                if self.should_advance_stage(stage):
                    break

            if self.should_advance_stage(stage):
                break

            # Get next values
            obs1, obs2 = self.get_observations()

            # Update agent1 (always trains)
            next_value1 = self.agent1.critic(obs1).squeeze().detach()
            states1, actions1, rewards1, dones1, log_probs1, values1 = self.buffer1.get()

            steps_per_env = len(self.buffer1.states)
            rewards1 = rewards1.view(steps_per_env, self.config.num_envs)
            values1 = values1.view(steps_per_env, self.config.num_envs)
            dones1 = dones1.view(steps_per_env, self.config.num_envs)

            # Compute advantages and returns (VECTORIZED - all envs at once)
            advantages1, returns1 = self.compute_gae(
                rewards1, values1, dones1, next_value1
            )
            advantages1 = advantages1.view(-1)
            returns1 = returns1.view(-1)

            actor_loss1, critic_loss1, _ = self.update_agent(
                self.agent1, states1, actions1, log_probs1, advantages1, returns1
            )
            self.losses1.append(actor_loss1 + critic_loss1)

            # Update agent2 if training
            if stage.agent2_trains:
                next_value2 = self.agent2.critic(obs2).squeeze().detach()
                states2, actions2, rewards2, dones2, log_probs2, values2 = self.buffer2.get()

                rewards2 = rewards2.view(steps_per_env, self.config.num_envs)
                values2 = values2.view(steps_per_env, self.config.num_envs)
                dones2 = dones2.view(steps_per_env, self.config.num_envs)

                # Compute advantages and returns (VECTORIZED - all envs at once)
                advantages2, returns2 = self.compute_gae(
                    rewards2, values2, dones2, next_value2
                )
                advantages2 = advantages2.view(-1)
                returns2 = returns2.view(-1)

                actor_loss2, critic_loss2, _ = self.update_agent(
                    self.agent2, states2, actions2, log_probs2, advantages2, returns2
                )
                self.losses2.append(actor_loss2 + critic_loss2)

            # Logging
            if self.total_steps % self.config.log_interval == 0:
                win_rate = self.calculate_win_rate(window=100)
                avg_score1 = np.mean(self.scores1[-100:]) if self.scores1 else 0
                avg_score2 = np.mean(self.scores2[-100:]) if self.scores2 else 0
                avg_loss1 = np.mean(self.losses1[-10:]) if self.losses1 else 0

                print(f"[Step {self.total_steps:>6}] "
                      f"Stage {stage.stage_id} | "
                      f"Win Rate: {win_rate:.2%} | "
                      f"Scores: {avg_score1:.1f} vs {avg_score2:.1f} | "
                      f"Loss: {avg_loss1:.4f}", flush=True)

        # Stage complete
        stage_time = time.time() - stage_start_time
        self.save_checkpoint(stage.name)

        print(f"\n{'='*70}", flush=True)
        print(f"STAGE {stage.stage_id} COMPLETE", flush=True)
        print(f"  Time: {stage_time/60:.1f} minutes", flush=True)
        print(f"  Steps: {self.stage_steps:,}", flush=True)
        print(f"  Win rate: {self.calculate_win_rate():.2%}", flush=True)
        print(f"{'='*70}\n", flush=True)

    def save_checkpoint(self, stage_name: str = None):
        """Override to accept stage name for curriculum checkpoints"""
        import json
        from datetime import datetime
        import numpy as np

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if stage_name is not None:
            # Curriculum checkpoint (per stage)
            suffix = stage_name
        else:
            # Regular checkpoint
            suffix = f"step{self.total_steps}"

        # Save agent1
        agent1_path = self.save_dir / f"big_cnn_{suffix}_{timestamp}.pt"
        torch.save({
            'actor': self.agent1.actor.state_dict(),
            'critic': self.agent1.critic.state_dict(),
            'actor_optimizer': self.agent1.actor_optimizer.state_dict(),
            'critic_optimizer': self.agent1.critic_optimizer.state_dict(),
            'total_steps': self.total_steps
        }, agent1_path)

        # Save agent2
        agent2_path = self.save_dir / f"small_cnn_{suffix}_{timestamp}.pt"
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
                'win_rate': self.calculate_win_rate(),
                'avg_score1': float(np.mean(self.scores1[-100:])) if self.scores1 else 0,
                'avg_score2': float(np.mean(self.scores2[-100:])) if self.scores2 else 0,
            }, f, indent=2)

        print(f"Saved checkpoint: {suffix}", flush=True)

    def train(self):
        """Run full curriculum training with CNN"""
        import time

        print("\n" + "="*70, flush=True)
        print("PPO TWO-SNAKE CURRICULUM TRAINING (CNN)", flush=True)
        print("="*70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num environments: {self.config.num_envs}", flush=True)
        print(f"Grid: {self.config.grid_size}x{self.config.grid_size} (5 channels)", flush=True)
        print(f"Total stages: {len(self.stages)}", flush=True)
        print("="*70, flush=True)

        total_start = time.time()

        # Train each curriculum stage
        for stage_idx, stage in enumerate(self.stages):
            self.current_stage_idx = stage_idx
            self.train_curriculum_stage(stage)

            # Final stage has no threshold - runs for min_steps
            if stage.win_rate_threshold is None:
                break

        total_time = time.time() - total_start

        # Final summary
        print("\n" + "="*70, flush=True)
        print("CURRICULUM TRAINING COMPLETE!", flush=True)
        print("="*70, flush=True)
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)", flush=True)
        print(f"Total steps: {self.total_steps:,}", flush=True)
        print(f"Total rounds: {self.total_rounds:,}", flush=True)
        print(f"Final win rate: {self.calculate_win_rate():.2%}", flush=True)

        # Print profiling stats
        stats = self.grid_encoder.get_profiling_stats()
        if stats:
            print(f"\nGrid Encoding Performance:", flush=True)
            print(f"  Total encodings: {stats['total_encodings']}", flush=True)
            print(f"  Avg encoding time: {stats['avg_encoding_ms']:.2f} ms", flush=True)
            print(f"  Total encoding time: {stats['total_time_s']:.1f} s ({stats['total_time_s']/total_time*100:.1f}% of total)", flush=True)

        print(f"Saved to: {self.save_dir}", flush=True)
        print("="*70 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='PPO Two-Snake Curriculum Training (CNN)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num-envs', type=int, default=128,
                        help='Number of parallel environments')
    parser.add_argument('--rollout-steps', type=int, default=2048,
                        help='Rollout steps before PPO update')
    parser.add_argument('--actor-lr', type=float, default=0.0003,
                        help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=0.0003,
                        help='Critic learning rate')
    parser.add_argument('--save-dir', type=str,
                        default='results/weights/ppo_two_snake_cnn_curriculum',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create config
    config = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        save_dir=args.save_dir,
        seed=args.seed
    )

    # Create and run trainer
    trainer = CNNCurriculumTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
