"""
PPO Independent Curriculum Training for Two-Snake Competitive Environment

Trains a SINGLE network through curriculum stages 0-3 independently.
Both 128x128 and 256x256 networks can train in parallel using this script.

Curriculum stages (fixed step counts):
- Stage 0: vs StaticAgent (0.5M steps, 95% threshold)
- Stage 1: vs RandomAgent (0.5M steps, 95% threshold)
- Stage 2: vs GreedyFoodAgent (3M steps, 35% threshold)
- Stage 3: vs Frozen self-policy (2M steps, 90% threshold)

After Stage 3, use train_ppo_coevolution_cross.py for Stage 4 (co-evolution, 8M steps).

Usage:
    # Train 128x128 network
    ./venv/Scripts/python.exe scripts/training/train_ppo_curriculum_independent.py \
        --hidden-dims 128 128 --save-dir results/weights/ppo_curriculum_128x128

    # Train 256x256 network (can run in parallel)
    ./venv/Scripts/python.exe scripts/training/train_ppo_curriculum_independent.py \
        --hidden-dims 256 256 --save-dir results/weights/ppo_curriculum_256x256
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
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List
from datetime import datetime

from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from core.networks import PPO_Actor_MLP, PPO_Critic_MLP
from core.utils import set_seed, get_device
from scripts.baselines.scripted_opponents import get_scripted_agent


@dataclass
class StageConfig:
    """Configuration for a curriculum stage"""
    stage_id: int
    name: str
    opponent_type: str  # 'static', 'random', 'greedy', 'frozen'
    target_food: int
    min_steps: int
    max_steps: int
    win_rate_threshold: float
    description: str = ""


@dataclass
class IndependentCurriculumConfig:
    """Configuration for independent curriculum training"""
    # Network architecture
    hidden_dims: Tuple[int, int] = (128, 128)

    # Environment
    num_envs: int = 128
    grid_size: int = 20
    max_steps: int = 1000

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

    # Logging
    log_interval: int = 10000

    # Curriculum stages (fixed step counts - training runs for exactly min_steps)
    stages: List[StageConfig] = field(default_factory=lambda: [
        StageConfig(0, "Static", "static", 10, 500_000, 500_000, 0.95,
                   "Learn basic movement vs static opponent"),
        StageConfig(1, "Random", "random", 10, 500_000, 500_000, 0.95,
                   "Handle unpredictability vs random opponent"),
        StageConfig(2, "Greedy", "greedy", 4, 3_000_000, 3_000_000, 0.35,
                   "Compete for food vs greedy BFS opponent"),
        StageConfig(3, "Frozen", "frozen", 6, 2_000_000, 2_000_000, 0.90,
                   "Compete against frozen self-policy from Stage 2"),
    ])

    # Output
    save_dir: str = "results/weights/ppo_curriculum_independent"
    seed: int = 67
    resume: Optional[str] = None  # Path to checkpoint to resume from


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


class IndependentCurriculumTrainer:
    """Trains a single network through curriculum stages 0-3"""

    def __init__(self, config: IndependentCurriculumConfig):
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
            target_food=10,  # Will be updated per stage
            device=self.device
        )

        # Agent (trainable network)
        self.actor = PPO_Actor_MLP(33, 3, config.hidden_dims).to(self.device)
        self.critic = PPO_Critic_MLP(33, config.hidden_dims).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.actor_scaler = GradScaler('cuda')
        self.critic_scaler = GradScaler('cuda')

        # Frozen self-policy (created after Stage 2)
        self.frozen_actor = None

        # Buffer
        self.buffer = PPOBuffer(config.rollout_steps, self.device)

        # Scripted opponents
        self.scripted_agents = {}
        for agent_type in ['static', 'random', 'greedy']:
            try:
                self.scripted_agents[agent_type] = get_scripted_agent(agent_type, device=self.device)
            except Exception as e:
                print(f"Warning: Could not load {agent_type} agent: {e}")

        # Metrics
        self.total_steps = 0
        self.stage_steps = 0
        self.round_winners = []
        self.scores_agent = []
        self.scores_opponent = []
        self.losses = []

        # Per-stage history for plotting
        self.stage_histories = {}

        # Current stage
        self.current_stage_idx = 0

        # Resume from checkpoint if specified
        if config.resume:
            self.load_checkpoint(config.resume)

    def select_actions(self, states: torch.Tensor):
        """Select actions using current policy"""
        with torch.no_grad():
            logits = self.actor(states)
            values = self.critic(states).squeeze()

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions, log_probs, values

    def select_greedy_actions(self, states: torch.Tensor):
        """Select greedy actions (for frozen policy)"""
        with torch.no_grad():
            logits = self.frozen_actor(states)
            probs = F.softmax(logits, dim=-1)
            actions = probs.argmax(dim=-1)
        return actions

    def get_opponent_actions(self, obs: torch.Tensor, stage: StageConfig):
        """Get opponent actions based on stage type"""
        if stage.opponent_type == 'frozen':
            if self.frozen_actor is None:
                raise RuntimeError("Frozen actor not initialized. Stage 2 must complete first.")
            return self.select_greedy_actions(obs)
        elif stage.opponent_type in self.scripted_agents:
            return self.scripted_agents[stage.opponent_type].select_action(self.env)
        else:
            # Fallback to random
            return torch.randint(0, 3, (self.config.num_envs,), device=self.device)

    def create_frozen_opponent(self):
        """Create frozen copy of current policy after Stage 2"""
        print("Creating frozen self-policy opponent...", flush=True)
        self.frozen_actor = copy.deepcopy(self.actor)
        self.frozen_actor.eval()
        for param in self.frozen_actor.parameters():
            param.requires_grad = False
        print("Frozen opponent created successfully.", flush=True)

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

    def update_agent(self, states, actions, old_log_probs, advantages, returns):
        """Update agent networks"""
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

                with autocast('cuda'):
                    # Evaluate actions
                    logits = self.actor(batch_states)
                    values = self.critic(batch_states).squeeze()

                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()

                    # Actor loss
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                       1 + self.config.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic loss
                    critic_loss = F.mse_loss(values, batch_returns)

                # Update actor
                self.actor_optimizer.zero_grad()
                self.actor_scaler.scale(actor_loss).backward()
                self.actor_scaler.unscale_(self.actor_optimizer)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_scaler.step(self.actor_optimizer)
                self.actor_scaler.update()

                # Update critic
                self.critic_optimizer.zero_grad()
                self.critic_scaler.scale(critic_loss).backward()
                self.critic_scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_scaler.step(self.critic_optimizer)
                self.critic_scaler.update()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_updates += 1

        return total_actor_loss / n_updates, total_critic_loss / n_updates

    def calculate_win_rate(self, window: int = 100) -> float:
        """Calculate agent win rate over last N rounds"""
        if len(self.round_winners) < window:
            window = len(self.round_winners)
        if window == 0:
            return 0.0

        recent_winners = self.round_winners[-window:]
        agent_wins = sum(1 for w in recent_winners if w == 1)  # Agent is snake 1
        return agent_wins / window

    def calculate_rates(self, window: int = 100) -> Tuple[float, float, float]:
        """Calculate win rate, opponent win rate, and draw rate over last N rounds"""
        if len(self.round_winners) < window:
            window = len(self.round_winners)
        if window == 0:
            return 0.0, 0.0, 0.0

        recent_winners = self.round_winners[-window:]
        agent_wins = sum(1 for w in recent_winners if w == 1)  # Agent is snake 1
        opponent_wins = sum(1 for w in recent_winners if w == 2)  # Opponent is snake 2
        draws = sum(1 for w in recent_winners if w == 3)  # Draw
        return agent_wins / window, opponent_wins / window, draws / window

    def should_advance_stage(self, stage: StageConfig) -> bool:
        """Check if current stage is complete"""
        if self.stage_steps < stage.min_steps:
            return False

        if self.stage_steps >= stage.max_steps:
            print(f"[MAX STEPS] Stage {stage.stage_id} reached max_steps={stage.max_steps:,}, advancing...", flush=True)
            return True

        win_rate = self.calculate_win_rate(window=100)
        if win_rate >= stage.win_rate_threshold:
            print(f"[THRESHOLD] Stage {stage.stage_id} reached {win_rate:.2%} >= {stage.win_rate_threshold:.0%}, advancing...", flush=True)
            return True

        return False

    def train_stage(self, stage: StageConfig):
        """Train a single curriculum stage"""
        print("\n" + "=" * 70, flush=True)
        print(f"STAGE {stage.stage_id}: {stage.name}", flush=True)
        print("=" * 70, flush=True)
        print(f"Network: {self.config.hidden_dims}", flush=True)
        print(f"Opponent: {stage.opponent_type}", flush=True)
        print(f"Target food: {stage.target_food}", flush=True)
        print(f"Min steps: {stage.min_steps:,}", flush=True)
        print(f"Max steps: {stage.max_steps:,}", flush=True)
        print(f"Win rate threshold: {stage.win_rate_threshold:.0%}", flush=True)
        print(f"Description: {stage.description}", flush=True)
        print("=" * 70 + "\n", flush=True)

        # Reset stage tracking
        self.stage_steps = 0
        self.round_winners = []
        self.scores_agent = []
        self.scores_opponent = []
        stage_start_time = time.time()

        # Track stage starting step for cumulative tracking
        stage_start_total_steps = self.total_steps

        # Initialize stage history
        self.stage_histories[stage.stage_id] = {
            'name': stage.name,
            'opponent_type': stage.opponent_type,
            'target_food': stage.target_food,
            'threshold': stage.win_rate_threshold,
            'history': []
        }

        # Set target food for this stage
        self.env.set_target_food(stage.target_food)

        # Reset environment
        obs_agent, obs_opponent = self.env.reset()

        # Training loop
        while not self.should_advance_stage(stage):
            self.buffer.clear()

            # Collect rollout
            for _ in range(max(1, self.config.rollout_steps // self.config.num_envs)):
                # Agent selects actions
                actions_agent, log_probs, values = self.select_actions(obs_agent)

                # Opponent selects actions
                actions_opponent = self.get_opponent_actions(obs_opponent, stage)

                # Environment step
                next_obs_agent, next_obs_opponent, r_agent, r_opponent, dones, info = \
                    self.env.step(actions_agent, actions_opponent)

                # Store transitions
                self.buffer.add(obs_agent, actions_agent, r_agent, dones, log_probs, values)

                # Track completed rounds
                if dones.any():
                    num_done = len(info['done_envs'])
                    for i in range(num_done):
                        winner = info['winners'][i]
                        self.round_winners.append(int(winner))
                        self.scores_agent.append(info['food_counts1'][i])
                        self.scores_opponent.append(info['food_counts2'][i])

                obs_agent = next_obs_agent
                obs_opponent = next_obs_opponent
                self.stage_steps += self.config.num_envs
                self.total_steps += self.config.num_envs

                if self.should_advance_stage(stage):
                    break

            if self.should_advance_stage(stage):
                break

            # Update agent
            with torch.no_grad():
                next_value = self.critic(obs_agent).squeeze()

            states, actions, rewards, dones_buf, log_probs, values = self.buffer.get()

            steps_per_env = len(self.buffer.states)
            rewards = rewards.view(steps_per_env, self.config.num_envs)
            values = values.view(steps_per_env, self.config.num_envs)
            dones_buf = dones_buf.view(steps_per_env, self.config.num_envs)

            advantages, returns = self.compute_gae(rewards, values, dones_buf, next_value)
            advantages = advantages.view(-1)
            returns = returns.view(-1)

            actor_loss, critic_loss = self.update_agent(states, actions, log_probs, advantages, returns)
            self.losses.append(actor_loss + critic_loss)

            # Logging - check if we crossed the log interval boundary
            if (self.stage_steps // self.config.log_interval) > ((self.stage_steps - steps_per_env * self.config.num_envs) // self.config.log_interval):
                win_rate, opp_win_rate, draw_rate = self.calculate_rates(window=100)
                avg_score_agent = np.mean(self.scores_agent[-100:]) if self.scores_agent else 0
                avg_score_opp = np.mean(self.scores_opponent[-100:]) if self.scores_opponent else 0
                avg_loss = np.mean(self.losses[-10:]) if self.losses else 0

                # Record history for plotting (with cumulative total_step)
                self.stage_histories[stage.stage_id]['history'].append({
                    'stage_step': self.stage_steps,
                    'total_step': self.total_steps,
                    'win_rate': win_rate,
                    'opponent_win_rate': opp_win_rate,
                    'draw_rate': draw_rate,
                    'avg_score_agent': float(avg_score_agent),
                    'avg_score_opponent': float(avg_score_opp),
                    'loss': float(avg_loss),
                    'episodes_completed': len(self.round_winners)
                })

                print(f"[Stage {stage.stage_id} | Step {self.stage_steps:>8,} / {stage.max_steps:,}] "
                      f"Win: {win_rate:.2%} Draw: {draw_rate:.1%} | "
                      f"Score: {avg_score_agent:.1f} vs {avg_score_opp:.1f} | "
                      f"Loss: {avg_loss:.4f}", flush=True)

        # Stage complete
        stage_time = time.time() - stage_start_time
        final_win_rate, final_opp_rate, final_draw_rate = self.calculate_rates()
        final_avg_score_agent = np.mean(self.scores_agent[-100:]) if self.scores_agent else 0
        final_avg_score_opp = np.mean(self.scores_opponent[-100:]) if self.scores_opponent else 0

        # Update stage history with final stats
        self.stage_histories[stage.stage_id]['final_win_rate'] = final_win_rate
        self.stage_histories[stage.stage_id]['final_opponent_win_rate'] = final_opp_rate
        self.stage_histories[stage.stage_id]['final_draw_rate'] = final_draw_rate
        self.stage_histories[stage.stage_id]['final_avg_score_agent'] = float(final_avg_score_agent)
        self.stage_histories[stage.stage_id]['final_avg_score_opponent'] = float(final_avg_score_opp)
        self.stage_histories[stage.stage_id]['total_steps'] = self.stage_steps
        self.stage_histories[stage.stage_id]['time_seconds'] = stage_time
        self.stage_histories[stage.stage_id]['total_episodes'] = len(self.round_winners)

        # Save stage checkpoint
        self.save_stage_checkpoint(stage.stage_id)

        # Create frozen opponent after Stage 2
        if stage.stage_id == 2:
            self.create_frozen_opponent()

        print(f"\n{'=' * 70}", flush=True)
        print(f"STAGE {stage.stage_id} COMPLETE", flush=True)
        print(f"  Time: {stage_time/60:.1f} minutes", flush=True)
        print(f"  Steps: {self.stage_steps:,}", flush=True)
        print(f"  Final win rate: {final_win_rate:.2%}", flush=True)
        print(f"{'=' * 70}\n", flush=True)

    def save_stage_checkpoint(self, stage_id: int):
        """Save checkpoint after completing a stage"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hidden_str = f"{self.config.hidden_dims[0]}x{self.config.hidden_dims[1]}"

        checkpoint = {
            'stage_id': stage_id,
            'hidden_dims': self.config.hidden_dims,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'stage_histories': self.stage_histories,
        }

        # Include frozen actor if it exists
        if self.frozen_actor is not None:
            checkpoint['frozen_actor'] = self.frozen_actor.state_dict()

        # Save checkpoint
        if stage_id == 3:
            filename = f"stage3_final_{hidden_str}_{timestamp}.pt"
        else:
            filename = f"stage{stage_id}_checkpoint_{hidden_str}_{timestamp}.pt"

        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)

        print(f"Saved checkpoint: {filename}", flush=True)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        print(f"Loading checkpoint: {checkpoint_path}", flush=True)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.current_stage_idx = checkpoint['stage_id'] + 1  # Resume from next stage

        if 'stage_histories' in checkpoint:
            self.stage_histories = checkpoint['stage_histories']

        if 'frozen_actor' in checkpoint:
            self.frozen_actor = PPO_Actor_MLP(33, 3, self.config.hidden_dims).to(self.device)
            self.frozen_actor.load_state_dict(checkpoint['frozen_actor'])
            self.frozen_actor.eval()
            for param in self.frozen_actor.parameters():
                param.requires_grad = False

        print(f"Resumed from stage {checkpoint['stage_id']}, will start stage {self.current_stage_idx}", flush=True)

    def save_training_history(self):
        """Save complete training history to JSON with comprehensive metrics"""
        hidden_str = f"{self.config.hidden_dims[0]}x{self.config.hidden_dims[1]}"

        # Build stage boundaries from history
        stage_boundaries = []
        cumulative_step = 0
        for stage_id in sorted(self.stage_histories.keys()):
            stage_data = self.stage_histories[stage_id]
            stage_steps = stage_data.get('total_steps', 0)
            stage_boundaries.append({
                'stage_id': int(stage_id),
                'name': stage_data.get('name', ''),
                'start_step': cumulative_step,
                'end_step': cumulative_step + stage_steps,
                'threshold': self.config.stages[int(stage_id)].win_rate_threshold if int(stage_id) < len(self.config.stages) else None
            })
            cumulative_step += stage_steps

        history = {
            'network_size': list(self.config.hidden_dims),
            'total_steps': self.total_steps,
            'total_time_seconds': sum(s.get('time_seconds', 0) for s in self.stage_histories.values()),
            'stages': self.stage_histories,
            'stage_boundaries': stage_boundaries,
            'config': {
                'hidden_dims': list(self.config.hidden_dims),
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
                'entropy_coef': self.config.entropy_coef,
            }
        }

        # Save to data directory
        data_dir = Path('results/data')
        data_dir.mkdir(parents=True, exist_ok=True)

        history_path = data_dir / f"curriculum_{hidden_str}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"Saved training history: {history_path}", flush=True)

    def train(self):
        """Run full curriculum training (stages 0-3)"""
        hidden_str = f"{self.config.hidden_dims[0]}x{self.config.hidden_dims[1]}"

        print("\n" + "=" * 70, flush=True)
        print("PPO INDEPENDENT CURRICULUM TRAINING", flush=True)
        print("=" * 70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Network size: {hidden_str}", flush=True)
        print(f"Num environments: {self.config.num_envs}", flush=True)
        print(f"Grid size: {self.config.grid_size}", flush=True)
        print(f"Total stages: {len(self.config.stages)}", flush=True)
        print(f"Save directory: {self.save_dir}", flush=True)
        print("=" * 70, flush=True)

        total_start = time.time()

        # Train each stage
        for stage_idx in range(self.current_stage_idx, len(self.config.stages)):
            stage = self.config.stages[stage_idx]
            self.train_stage(stage)

        total_time = time.time() - total_start

        # Save training history
        self.save_training_history()

        # Final summary
        print("\n" + "=" * 70, flush=True)
        print("CURRICULUM TRAINING COMPLETE!", flush=True)
        print("=" * 70, flush=True)
        print(f"Network: {hidden_str}", flush=True)
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)", flush=True)
        print(f"Total steps: {self.total_steps:,}", flush=True)
        print(f"Saved to: {self.save_dir}", flush=True)
        print("=" * 70 + "\n", flush=True)

        return {
            'total_steps': self.total_steps,
            'total_time': total_time,
            'stage_histories': self.stage_histories
        }


def main():
    parser = argparse.ArgumentParser(
        description='PPO Independent Curriculum Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--hidden-dims', type=int, nargs=2, default=[128, 128],
                        help='Hidden layer dimensions (e.g., 128 128 or 256 256)')
    parser.add_argument('--num-envs', type=int, default=128,
                        help='Number of parallel environments')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save checkpoints (default: results/weights/ppo_curriculum_NxN)')
    parser.add_argument('--seed', type=int, default=67,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    hidden_dims = tuple(args.hidden_dims)
    hidden_str = f"{hidden_dims[0]}x{hidden_dims[1]}"

    if args.save_dir is None:
        save_dir = f"results/weights/ppo_curriculum_{hidden_str}"
    else:
        save_dir = args.save_dir

    config = IndependentCurriculumConfig(
        hidden_dims=hidden_dims,
        num_envs=args.num_envs,
        save_dir=save_dir,
        seed=args.seed,
        resume=args.resume
    )

    trainer = IndependentCurriculumTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
