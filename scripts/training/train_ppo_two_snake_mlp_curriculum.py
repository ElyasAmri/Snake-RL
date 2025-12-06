"""
PPO Curriculum Training for Two-Snake Competitive Environment (MLP-based)

5-stage curriculum implementation extending the base PPO two-snake trainer.

Curriculum stages:
1. Stage 0: vs StaticAgent (learn basic movement, 20K steps, 70% win threshold)
2. Stage 1: vs RandomAgent (handle unpredictability, 20K steps, 60% win threshold)
3. Stage 2: vs GreedyFoodAgent (compete for food, 30K steps, 55% win threshold)
4. Stage 3: vs Frozen small network (30K steps, 50% win threshold)
5. Stage 4: Co-evolution (both learning, 150K steps, no threshold)

Total: 250K steps (optimized from original 800K)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Optional

# Import base trainer from non-curriculum version
from scripts.training.train_ppo_two_snake_mlp import (
    PPOConfig, TwoSnakePPOTrainer, TwoSnakePPOAgent, PPOBuffer
)
from core.utils import set_seed, get_device
from scripts.baselines.scripted_opponents import get_scripted_agent


@dataclass
class CurriculumStage:
    """Curriculum stage configuration"""
    stage_id: int
    name: str
    opponent_type: str  # 'static', 'random', 'greedy', 'frozen', 'learning'
    target_food: int  # Food count required to win
    min_steps: int
    max_steps: int  # Maximum steps before forced advancement
    win_rate_threshold: Optional[float]
    description: str
    agent2_trains: bool = False


class CurriculumPPOTrainer(TwoSnakePPOTrainer):
    """
    Extends base PPO trainer with curriculum learning.

    Key differences from base trainer:
    - Multi-stage training progression
    - Scripted opponents for early stages
    - Win rate thresholds for stage advancement
    - Frozen policy stage
    - Final co-evolution stage
    """

    def __init__(self, config: PPOConfig):
        super().__init__(config)

        # Curriculum-specific setup
        self.setup_curriculum()

    def setup_curriculum(self):
        """Initialize curriculum stages and scripted agents"""

        # Load scripted opponents
        self.scripted_agents = {}
        for agent_type in ['static', 'random', 'greedy']:
            try:
                self.scripted_agents[agent_type] = get_scripted_agent(
                    agent_type, device=self.device
                )
            except Exception as e:
                print(f"Warning: Could not load {agent_type} agent: {e}", flush=True)

        # Define curriculum stages with progressive difficulty
        # max_steps prevents infinite loops if threshold is too hard to reach
        # Stages 0-1: Short (trivial opponents), Stages 2-4: Longer (harder opponents)
        self.stages = [
            CurriculumStage(
                stage_id=0,
                name="Stage0_Static",
                opponent_type="static",
                target_food=10,  # Easy: static opponent doesn't move
                min_steps=200000,  # Short - trivial opponent
                max_steps=500000,  # Safety limit
                win_rate_threshold=0.70,
                description="Learn basic movement vs static opponent (target: 10 food)",
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=1,
                name="Stage1_Random",
                opponent_type="random",
                target_food=10,  # Medium: random opponent is inefficient
                min_steps=200000,  # Short - trivial opponent
                max_steps=500000,  # Safety limit
                win_rate_threshold=0.60,
                description="Handle unpredictability vs random opponent (target: 10 food)",
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=2,
                name="Stage2_Greedy",
                opponent_type="greedy",
                target_food=4,  # Reduced: greedy opponent is very effective at food
                min_steps=2000000,  # ~40 min - hard BFS opponent
                max_steps=5000000,  # Safety limit
                win_rate_threshold=0.35,  # Achievable based on previous training
                description="Compete for food vs greedy opponent (target: 4 food)",
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=3,
                name="Stage3_Frozen",
                opponent_type="frozen",
                target_food=6,  # Progressive difficulty
                min_steps=1000000,  # ~20 min
                max_steps=2000000,  # Safety limit
                win_rate_threshold=0.30,  # Achievable with enough training
                description="Compete against frozen policy (target: 6 food)",
                agent2_trains=False
            ),
            CurriculumStage(
                stage_id=4,
                name="Stage4_CoEvolution",
                opponent_type="learning",
                target_food=8,  # Both agents learning together
                min_steps=6000000,  # ~120 min
                max_steps=8000000,  # Safety limit
                win_rate_threshold=None,  # No threshold - trains for full duration
                description="Full co-evolution training (target: 8 food)",
                agent2_trains=True
            )
        ]

        # Curriculum state
        self.current_stage_idx = 0
        self.stage_steps = 0

        # Win rate history for plotting
        self.win_rate_history = []

    def get_stage_actions(self, obs1, obs2, stage):
        """
        Get actions based on curriculum stage.

        This is the key method that differs from base trainer - handles
        different opponent types per stage.
        """
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

    def should_advance_stage(self, stage):
        """Check if current stage is complete"""
        # Must meet minimum steps
        if self.stage_steps < stage.min_steps:
            return False

        # Check max_steps limit (prevents infinite loops)
        if self.stage_steps >= stage.max_steps:
            print(f"[MAX STEPS] Stage {stage.stage_id} reached max_steps={stage.max_steps}, advancing...", flush=True)
            return True

        # Final stage - no threshold, just meet min_steps
        if stage.win_rate_threshold is None:
            return True  # FIX: Stop after min_steps when no threshold

        # Check win rate threshold
        win_rate = self.calculate_win_rate(window=100)
        return win_rate >= stage.win_rate_threshold

    def train_curriculum_stage(self, stage):
        """Train a single curriculum stage"""
        import time
        import numpy as np

        print("\n" + "="*70, flush=True)
        print(f"STAGE {stage.stage_id}: {stage.name}", flush=True)
        print("="*70, flush=True)
        print(f"Opponent: {stage.opponent_type}", flush=True)
        print(f"Target food: {stage.target_food}", flush=True)
        print(f"Min steps: {stage.min_steps:,}", flush=True)
        print(f"Max steps: {stage.max_steps:,}", flush=True)
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
        obs1, obs2 = self.env.reset()

        # Stage training loop
        print(f"Starting training loop (total_steps: {self.total_steps}, stage_steps: {self.stage_steps})...", flush=True)
        while not self.should_advance_stage(stage):
            # Clear buffers
            self.buffer1.clear()
            if stage.agent2_trains:
                self.buffer2.clear()

            # Collect rollout
            for _ in range(max(1, self.config.rollout_steps // self.config.num_envs)):
                # Get actions (stage-dependent)
                actions1, log_probs1, values1, actions2, log_probs2, values2 = \
                    self.get_stage_actions(obs1, obs2, stage)

                # Environment step
                next_obs1, next_obs2, r1, r2, dones, info = self.env.step(actions1, actions2)

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

                obs1 = next_obs1
                obs2 = next_obs2
                # FIX: Count all parallel environment steps, not just iterations
                self.stage_steps += self.config.num_envs
                self.total_steps += self.config.num_envs

                if self.should_advance_stage(stage):
                    break

                # Check max_time within stage
                if self.config.max_time and hasattr(self, 'training_start_time') and (time.time() - self.training_start_time) >= self.config.max_time:
                    print(f"\n[TIMEOUT] Max time {self.config.max_time}s reached. Stopping training...", flush=True)
                    break

            if self.should_advance_stage(stage):
                break
            if self.config.max_time and hasattr(self, 'training_start_time') and (time.time() - self.training_start_time) >= self.config.max_time:
                break

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

            # Logging (more frequent for co-evolution stage)
            log_freq = 50 if stage.agent2_trains else self.config.log_interval
            if self.total_steps % log_freq == 0:
                import numpy as np
                win_rate = self.calculate_win_rate(window=100)
                avg_score1 = np.mean(self.scores1[-100:]) if self.scores1 else 0
                avg_score2 = np.mean(self.scores2[-100:]) if self.scores2 else 0
                avg_loss1 = np.mean(self.losses1[-10:]) if self.losses1 else 0

                # Track win rate history for plotting
                self.win_rate_history.append({
                    'step': self.total_steps,
                    'stage_step': self.stage_steps,
                    'stage': stage.stage_id,
                    'stage_name': stage.name,
                    'win_rate': win_rate,
                    'avg_score1': float(avg_score1),
                    'avg_score2': float(avg_score2)
                })

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

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if stage_name is not None:
            # Curriculum checkpoint (per stage)
            suffix = stage_name
        else:
            # Regular checkpoint
            suffix = f"step{self.total_steps}"

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
                'win_rate': self.calculate_win_rate(),
                'avg_score1': float(np.mean(self.scores1[-100:])) if self.scores1 else 0,
                'avg_score2': float(np.mean(self.scores2[-100:])) if self.scores2 else 0,
            }, f, indent=2)

        print(f"Saved checkpoint: {suffix}", flush=True)

    def train(self):
        """Run full curriculum training"""
        import time

        print("\n" + "="*70, flush=True)
        print("PPO TWO-SNAKE CURRICULUM TRAINING (MLP)", flush=True)
        print("="*70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num environments: {self.config.num_envs}", flush=True)
        print(f"Grid size: {self.config.grid_size}x{self.config.grid_size}", flush=True)
        print(f"Agent1 (Big): {self.config.big_hidden_dims}", flush=True)
        print(f"Agent2 (Small): {self.config.small_hidden_dims}", flush=True)
        print(f"Total stages: {len(self.stages)}", flush=True)
        print("="*70, flush=True)

        total_start = time.time()
        self.training_start_time = total_start  # Store for use in train_curriculum_stage

        # Train each curriculum stage
        for stage_idx, stage in enumerate(self.stages):
            self.current_stage_idx = stage_idx
            self.train_curriculum_stage(stage)

            # Final stage has no threshold - runs for min_steps
            if stage.win_rate_threshold is None:
                break

            # Check max_time between stages
            if self.config.max_time and (time.time() - total_start) >= self.config.max_time:
                print(f"\n[TIMEOUT] Max time {self.config.max_time}s reached. Stopping training...", flush=True)
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
        stats = self.env.feature_encoder.get_profiling_stats()
        if stats:
            print(f"\nEncoding Performance:", flush=True)
            print(f"  Total encodings: {stats['total_encodings']}", flush=True)
            print(f"  Avg danger_self: {stats['avg_danger_self_ms']:.2f} ms", flush=True)
            print(f"  Avg danger_opp: {stats['avg_danger_opp_ms']:.2f} ms", flush=True)
            print(f"  Avg food_dir: {stats['avg_food_dir_ms']:.2f} ms", flush=True)
            print(f"  Total encoding time: {stats['total_time_s']:.1f} s ({stats['total_time_s']/total_time*100:.1f}% of total)", flush=True)

        print(f"Saved to: {self.save_dir}", flush=True)
        print("="*70 + "\n", flush=True)

        # Return results dict
        return {
            'total_steps': self.total_steps,
            'total_rounds': self.total_rounds,
            'final_win_rate': self.calculate_win_rate(),
            'avg_score1': float(np.mean(self.scores1[-100:])) if self.scores1 else 0,
            'avg_score2': float(np.mean(self.scores2[-100:])) if self.scores2 else 0,
            'win_rate_history': self.win_rate_history,
            'total_time': total_time,
            'stages_completed': self.current_stage_idx + 1
        }


def main():
    parser = argparse.ArgumentParser(
        description='PPO Two-Snake Curriculum Training (MLP)',
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
                        default='results/weights/ppo_two_snake_mlp_curriculum',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=67,
                        help='Random seed')
    parser.add_argument('--max-time', type=int, default=None,
                        help='Maximum training time in seconds')

    args = parser.parse_args()

    # Create config
    config = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        save_dir=args.save_dir,
        seed=args.seed,
        max_time=args.max_time
    )

    # Create and run trainer
    trainer = CurriculumPPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
