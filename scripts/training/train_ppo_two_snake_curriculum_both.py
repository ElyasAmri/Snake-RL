"""
PPO Curriculum Training for Both Model Sizes (Big 256x256 and Small 128x128)

Training Flow:
1. Phase 1 (Parallel): Train Big and Small through Stages 0-3 independently
2. Phase 2 (Co-evolution): Stage 4 with both trained models competing
3. Phase 3 (Evaluation): Head-to-head competition stats

Stage Configuration:
- Stage 0: vs Static (200K-500K steps, 70% threshold)
- Stage 1: vs Random (200K-500K steps, 60% threshold)
- Stage 2: vs Greedy (2M-5M steps, 35% threshold)
- Stage 3: vs Frozen Self (1M-2M steps, 30% threshold)
- Stage 4: Co-evolution (6M-8M steps, no threshold)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
import copy
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from core.utils import set_seed, get_device
from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from scripts.baselines.scripted_opponents import get_scripted_agent


@dataclass
class CurriculumConfig:
    """Configuration for curriculum training"""
    num_envs: int = 64  # Reduced for parallel training
    grid_size: int = 20
    rollout_steps: int = 2048
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    actor_lr: float = 0.0003
    critic_lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    log_interval: int = 50
    save_dir: str = "results/weights/ppo_curriculum_both"


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage"""
    stage_id: int
    name: str
    opponent_type: str  # 'static', 'random', 'greedy', 'frozen'
    target_food: int
    min_steps: int
    max_steps: int
    win_rate_threshold: Optional[float]


# Define stages 0-3 (Stage 4 is co-evolution, handled separately)
STAGES_0_TO_3 = [
    StageConfig(0, "Stage0_Static", "static", 10, 200_000, 500_000, 0.70),
    StageConfig(1, "Stage1_Random", "random", 10, 200_000, 500_000, 0.60),
    StageConfig(2, "Stage2_Greedy", "greedy", 4, 2_000_000, 5_000_000, 0.35),
    StageConfig(3, "Stage3_Frozen", "frozen", 6, 1_000_000, 2_000_000, 0.30),
]

STAGE_4_COEVOLUTION = StageConfig(4, "Stage4_CoEvolution", "learning", 8, 6_000_000, 8_000_000, None)


class PPOActor(nn.Module):
    """Policy network for PPO"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )

    def forward(self, x):
        return self.network(x)


class PPOCritic(nn.Module):
    """Value network for PPO"""
    def __init__(self, obs_dim: int, hidden_dims: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x):
        return self.network(x)


class PPOAgent:
    """PPO Agent with actor-critic architecture"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, int],
                 actor_lr: float, critic_lr: float, device: torch.device):
        self.device = device
        self.hidden_dims = hidden_dims

        self.actor = PPOActor(obs_dim, action_dim, hidden_dims).to(device)
        self.critic = PPOCritic(obs_dim, hidden_dims).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_actions(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select actions with exploration"""
        with torch.no_grad():
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            values = self.critic(obs).squeeze(-1)
        return actions, log_probs, values

    def select_greedy_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Select greedy actions (no exploration)"""
        with torch.no_grad():
            logits = self.actor(obs)
            return logits.argmax(dim=-1)

    def get_state_dict(self) -> Dict:
        """Get state dict for saving"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])


class PPOBuffer:
    """Rollout buffer for PPO"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.rewards),
            torch.stack(self.dones),
            torch.stack(self.log_probs),
            torch.stack(self.values)
        )


class SingleModelCurriculumTrainer:
    """
    Train a single model (big or small) through stages 0-3.
    Each model trains independently against scripted opponents.
    """

    def __init__(self, config: CurriculumConfig, model_size: str, device: torch.device):
        """
        Args:
            config: Training configuration
            model_size: 'big' (256x256) or 'small' (128x128)
            device: CUDA device to use
        """
        self.config = config
        self.model_size = model_size
        self.device = device
        self.hidden_dims = (256, 256) if model_size == 'big' else (128, 128)

        # Create save directory
        self.save_dir = Path(config.save_dir) / model_size
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Environment
        self.env = VectorizedTwoSnakeEnv(
            num_envs=config.num_envs,
            grid_size=config.grid_size,
            device=device,
        )
        # Get obs_dim from actual observation shape
        obs1, obs2 = self.env.reset()
        self.obs_dim = obs1.shape[1]  # [num_envs, obs_dim]
        self.action_dim = 3  # LEFT, STRAIGHT, RIGHT

        # Agent
        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            device=device
        )

        # Buffer
        self.buffer = PPOBuffer()

        # Scripted opponents
        self.scripted_agents = {}
        for agent_type in ['static', 'random', 'greedy']:
            try:
                self.scripted_agents[agent_type] = get_scripted_agent(agent_type, device=device)
            except Exception as e:
                print(f"[{model_size}] Warning: Could not load {agent_type} agent: {e}")

        # Frozen copy of self (for Stage 3)
        self.frozen_agent = None

        # Metrics
        self.total_steps = 0
        self.win_rate_history = []
        self.round_winners = []
        self.scores = []
        self.losses = []

    def calculate_win_rate(self, window: int = 100) -> float:
        """Calculate win rate over last N rounds"""
        if not self.round_winners:
            return 0.0
        recent = self.round_winners[-window:]
        wins = sum(1 for w in recent if w == 1)  # Agent1 wins
        return wins / len(recent)

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t].float()) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t].float()) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update_agent(self, states, actions, old_log_probs, advantages, returns):
        """PPO update step"""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(self.config.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Actor loss
                logits = self.agent.actor(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy

                # Critic loss
                values = self.agent.critic(batch_states).squeeze(-1)
                critic_loss = self.config.value_coef * nn.functional.mse_loss(values, batch_returns)

                # Update
                self.agent.actor_optimizer.zero_grad()
                self.agent.critic_optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.config.max_grad_norm)
                self.agent.actor_optimizer.step()
                self.agent.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        return total_actor_loss, total_critic_loss

    def get_opponent_actions(self, obs2: torch.Tensor, stage: StageConfig) -> torch.Tensor:
        """Get opponent actions based on stage type"""
        if stage.opponent_type == 'frozen':
            # Use frozen copy of self
            if self.frozen_agent is None:
                # First time in frozen stage - create frozen copy
                self.frozen_agent = PPOAgent(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    hidden_dims=self.hidden_dims,
                    actor_lr=self.config.actor_lr,
                    critic_lr=self.config.critic_lr,
                    device=self.device
                )
                self.frozen_agent.load_state_dict(self.agent.get_state_dict())
            return self.frozen_agent.select_greedy_actions(obs2)

        elif stage.opponent_type in self.scripted_agents:
            return self.scripted_agents[stage.opponent_type].select_action(self.env)

        else:
            # Fallback: random
            return torch.randint(0, 3, (self.config.num_envs,), device=self.device)

    def should_advance_stage(self, stage: StageConfig, stage_steps: int) -> bool:
        """Check if stage is complete"""
        if stage_steps < stage.min_steps:
            return False
        if stage_steps >= stage.max_steps:
            return True
        if stage.win_rate_threshold is None:
            return True
        return self.calculate_win_rate(window=100) >= stage.win_rate_threshold

    def train_stage(self, stage: StageConfig) -> Dict:
        """Train a single curriculum stage"""
        print(f"\n[{self.model_size.upper()}] {'='*60}")
        print(f"[{self.model_size.upper()}] STAGE {stage.stage_id}: {stage.name}")
        print(f"[{self.model_size.upper()}] Opponent: {stage.opponent_type}")
        print(f"[{self.model_size.upper()}] Target food: {stage.target_food}")
        print(f"[{self.model_size.upper()}] Steps: {stage.min_steps:,} - {stage.max_steps:,}")
        print(f"[{self.model_size.upper()}] Threshold: {stage.win_rate_threshold}")
        print(f"[{self.model_size.upper()}] {'='*60}\n")

        stage_steps = 0
        stage_start = time.time()

        # Set target food
        self.env.set_target_food(stage.target_food)

        # Reset environment
        obs1, obs2 = self.env.reset()

        while not self.should_advance_stage(stage, stage_steps):
            self.buffer.clear()

            # Collect rollout
            for _ in range(max(1, self.config.rollout_steps // self.config.num_envs)):
                # Agent actions
                actions1, log_probs1, values1 = self.agent.select_actions(obs1)

                # Opponent actions
                actions2 = self.get_opponent_actions(obs2, stage)

                # Environment step
                next_obs1, next_obs2, r1, r2, dones, info = self.env.step(actions1, actions2)

                # Store transitions
                self.buffer.add(obs1, actions1, r1, dones, log_probs1, values1)

                # Track metrics
                if dones.any():
                    for i in range(len(info['done_envs'])):
                        self.round_winners.append(int(info['winners'][i]))
                        self.scores.append(info['food_counts1'][i])

                obs1 = next_obs1
                obs2 = next_obs2
                stage_steps += self.config.num_envs
                self.total_steps += self.config.num_envs

                if self.should_advance_stage(stage, stage_steps):
                    break

            if self.should_advance_stage(stage, stage_steps):
                break

            # PPO update
            next_value = self.agent.critic(obs1).squeeze(-1).detach()
            states, actions, rewards, dones_buf, log_probs, values = self.buffer.get()

            steps_per_env = len(self.buffer.states)
            rewards = rewards.view(steps_per_env, self.config.num_envs)
            values = values.view(steps_per_env, self.config.num_envs)
            dones_buf = dones_buf.view(steps_per_env, self.config.num_envs)

            advantages, returns = self.compute_gae(rewards, values, dones_buf, next_value)
            advantages = advantages.view(-1)
            returns = returns.view(-1)

            actor_loss, critic_loss = self.update_agent(
                states.view(-1, self.obs_dim),
                actions.view(-1),
                log_probs.view(-1),
                advantages,
                returns
            )
            self.losses.append(actor_loss + critic_loss)

            # Logging
            if self.total_steps % (self.config.log_interval * self.config.num_envs) < self.config.num_envs:
                win_rate = self.calculate_win_rate()
                avg_score = np.mean(self.scores[-100:]) if self.scores else 0

                self.win_rate_history.append({
                    'step': self.total_steps,
                    'stage': stage.stage_id,
                    'stage_name': stage.name,
                    'win_rate': win_rate,
                    'avg_score': float(avg_score)
                })

                print(f"[{self.model_size.upper()}] Step {self.total_steps:>7,} | "
                      f"Stage {stage.stage_id} | "
                      f"Win: {win_rate:.1%} | "
                      f"Score: {avg_score:.1f}")

        # Save checkpoint
        stage_time = time.time() - stage_start
        self.save_checkpoint(stage.name)

        print(f"\n[{self.model_size.upper()}] Stage {stage.stage_id} Complete!")
        print(f"[{self.model_size.upper()}] Time: {stage_time/60:.1f} min")
        print(f"[{self.model_size.upper()}] Steps: {stage_steps:,}")
        print(f"[{self.model_size.upper()}] Win Rate: {self.calculate_win_rate():.1%}")

        return {
            'stage': stage.stage_id,
            'steps': stage_steps,
            'time': stage_time,
            'win_rate': self.calculate_win_rate()
        }

    def save_checkpoint(self, suffix: str):
        """Save agent checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.save_dir / f"{self.model_size}_{suffix}_{timestamp}.pt"
        torch.save({
            **self.agent.get_state_dict(),
            'total_steps': self.total_steps,
            'hidden_dims': self.hidden_dims,
        }, path)
        print(f"[{self.model_size.upper()}] Saved: {path.name}")

    def train_stages_0_to_3(self) -> PPOAgent:
        """Train through all stages 0-3 and return the trained agent"""
        print(f"\n[{self.model_size.upper()}] Starting curriculum training")
        print(f"[{self.model_size.upper()}] Hidden dims: {self.hidden_dims}")
        print(f"[{self.model_size.upper()}] Device: {self.device}")

        for stage in STAGES_0_TO_3:
            self.train_stage(stage)

            # Create frozen copy after Stage 2 for Stage 3
            if stage.stage_id == 2:
                self.frozen_agent = PPOAgent(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    hidden_dims=self.hidden_dims,
                    actor_lr=self.config.actor_lr,
                    critic_lr=self.config.critic_lr,
                    device=self.device
                )
                self.frozen_agent.load_state_dict(self.agent.get_state_dict())

        # Save final checkpoint and history
        self.save_checkpoint("final_stage3")

        history_path = Path(self.config.save_dir) / f"curriculum_{self.model_size}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.win_rate_history, f, indent=2)

        return self.agent


class CoEvolutionTrainer:
    """
    Stage 4: Co-evolution training where both Big and Small models learn simultaneously.
    """

    def __init__(self, big_agent: PPOAgent, small_agent: PPOAgent, config: CurriculumConfig, device: torch.device):
        self.config = config
        self.device = device

        self.agent1 = big_agent  # 256x256
        self.agent2 = small_agent  # 128x128

        # Environment
        self.env = VectorizedTwoSnakeEnv(
            num_envs=config.num_envs,
            grid_size=config.grid_size,
            device=device,
        )
        # Get obs_dim from actual observation shape
        obs1, obs2 = self.env.reset()
        self.obs_dim = obs1.shape[1]  # [num_envs, obs_dim]

        # Buffers
        self.buffer1 = PPOBuffer()
        self.buffer2 = PPOBuffer()

        # Save directory
        self.save_dir = Path(config.save_dir) / "coevolution"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.total_steps = 0
        self.win_rate_history = []
        self.round_winners = []
        self.scores1 = []
        self.scores2 = []
        self.losses1 = []
        self.losses2 = []

    def calculate_win_rate(self, window: int = 100) -> float:
        """Win rate for agent1 (big)"""
        if not self.round_winners:
            return 0.0
        recent = self.round_winners[-window:]
        wins = sum(1 for w in recent if w == 1)
        return wins / len(recent)

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE"""
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t].float()) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t].float()) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update_agent(self, agent: PPOAgent, states, actions, old_log_probs, advantages, returns):
        """PPO update for one agent"""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0

        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits = agent.actor(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy

                values = agent.critic(batch_states).squeeze(-1)
                critic_loss = self.config.value_coef * nn.functional.mse_loss(values, batch_returns)

                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(agent.critic.parameters(), self.config.max_grad_norm)
                agent.actor_optimizer.step()
                agent.critic_optimizer.step()

                total_loss += (actor_loss + critic_loss).item()

        return total_loss

    def train(self) -> Tuple[PPOAgent, PPOAgent]:
        """Run co-evolution training (Stage 4)"""
        stage = STAGE_4_COEVOLUTION

        print("\n" + "="*70)
        print("STAGE 4: CO-EVOLUTION (Big vs Small)")
        print("="*70)
        print(f"Target food: {stage.target_food}")
        print(f"Steps: {stage.min_steps:,} - {stage.max_steps:,}")
        print("="*70 + "\n")

        stage_start = time.time()
        self.env.set_target_food(stage.target_food)

        obs1, obs2 = self.env.reset()

        while self.total_steps < stage.min_steps:
            self.buffer1.clear()
            self.buffer2.clear()

            # Collect rollout
            for _ in range(max(1, self.config.rollout_steps // self.config.num_envs)):
                # Both agents select actions
                actions1, log_probs1, values1 = self.agent1.select_actions(obs1)
                actions2, log_probs2, values2 = self.agent2.select_actions(obs2)

                # Environment step
                next_obs1, next_obs2, r1, r2, dones, info = self.env.step(actions1, actions2)

                # Store transitions
                self.buffer1.add(obs1, actions1, r1, dones, log_probs1, values1)
                self.buffer2.add(obs2, actions2, r2, dones, log_probs2, values2)

                # Track metrics
                if dones.any():
                    for i in range(len(info['done_envs'])):
                        self.round_winners.append(int(info['winners'][i]))
                        self.scores1.append(info['food_counts1'][i])
                        self.scores2.append(info['food_counts2'][i])

                obs1 = next_obs1
                obs2 = next_obs2
                self.total_steps += self.config.num_envs

            # Update both agents
            next_value1 = self.agent1.critic(obs1).squeeze(-1).detach()
            next_value2 = self.agent2.critic(obs2).squeeze(-1).detach()

            states1, actions1, rewards1, dones1, log_probs1, values1 = self.buffer1.get()
            states2, actions2, rewards2, dones2, log_probs2, values2 = self.buffer2.get()

            steps_per_env = len(self.buffer1.states)

            rewards1 = rewards1.view(steps_per_env, self.config.num_envs)
            values1 = values1.view(steps_per_env, self.config.num_envs)
            dones1 = dones1.view(steps_per_env, self.config.num_envs)

            rewards2 = rewards2.view(steps_per_env, self.config.num_envs)
            values2 = values2.view(steps_per_env, self.config.num_envs)
            dones2 = dones2.view(steps_per_env, self.config.num_envs)

            advantages1, returns1 = self.compute_gae(rewards1, values1, dones1, next_value1)
            advantages2, returns2 = self.compute_gae(rewards2, values2, dones2, next_value2)

            loss1 = self.update_agent(
                self.agent1,
                states1.view(-1, self.obs_dim),
                actions1.view(-1),
                log_probs1.view(-1),
                advantages1.view(-1),
                returns1.view(-1)
            )
            loss2 = self.update_agent(
                self.agent2,
                states2.view(-1, self.obs_dim),
                actions2.view(-1),
                log_probs2.view(-1),
                advantages2.view(-1),
                returns2.view(-1)
            )

            self.losses1.append(loss1)
            self.losses2.append(loss2)

            # Logging
            if self.total_steps % (self.config.log_interval * self.config.num_envs) < self.config.num_envs:
                win_rate = self.calculate_win_rate()
                avg_score1 = np.mean(self.scores1[-100:]) if self.scores1 else 0
                avg_score2 = np.mean(self.scores2[-100:]) if self.scores2 else 0

                self.win_rate_history.append({
                    'step': self.total_steps,
                    'stage': 4,
                    'stage_name': 'CoEvolution',
                    'win_rate': win_rate,
                    'avg_score_big': float(avg_score1),
                    'avg_score_small': float(avg_score2)
                })

                print(f"[COEVO] Step {self.total_steps:>7,} | "
                      f"Big Win: {win_rate:.1%} | "
                      f"Scores: {avg_score1:.1f} vs {avg_score2:.1f}")

        # Save checkpoints
        stage_time = time.time() - stage_start
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        torch.save({
            **self.agent1.get_state_dict(),
            'total_steps': self.total_steps,
            'hidden_dims': self.agent1.hidden_dims,
        }, self.save_dir / f"big_final_{timestamp}.pt")

        torch.save({
            **self.agent2.get_state_dict(),
            'total_steps': self.total_steps,
            'hidden_dims': self.agent2.hidden_dims,
        }, self.save_dir / f"small_final_{timestamp}.pt")

        # Save history
        history_path = Path(self.config.save_dir) / "coevolution_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.win_rate_history, f, indent=2)

        print(f"\nCo-evolution Complete!")
        print(f"Time: {stage_time/60:.1f} min")
        print(f"Steps: {self.total_steps:,}")
        print(f"Final Big Win Rate: {self.calculate_win_rate():.1%}")

        return self.agent1, self.agent2


def evaluate_head_to_head(big_agent: PPOAgent, small_agent: PPOAgent,
                          config: CurriculumConfig, device: torch.device,
                          num_games: int = 1000) -> Dict:
    """
    Evaluate trained models head-to-head (no learning, greedy actions).
    """
    print("\n" + "="*70)
    print("HEAD-TO-HEAD EVALUATION")
    print("="*70)

    env = VectorizedTwoSnakeEnv(
        num_envs=min(num_games, 256),
        grid_size=config.grid_size,
        device=device,
    )
    env.set_target_food(8)

    big_wins = 0
    small_wins = 0
    draws = 0
    big_scores = []
    small_scores = []

    games_played = 0
    obs1, obs2 = env.reset()

    while games_played < num_games:
        # Greedy actions (no exploration)
        actions1 = big_agent.select_greedy_actions(obs1)
        actions2 = small_agent.select_greedy_actions(obs2)

        obs1, obs2, r1, r2, dones, info = env.step(actions1, actions2)

        if dones.any():
            for i in range(len(info['done_envs'])):
                winner = info['winners'][i]
                if winner == 1:
                    big_wins += 1
                elif winner == 2:
                    small_wins += 1
                else:
                    draws += 1

                big_scores.append(info['food_counts1'][i])
                small_scores.append(info['food_counts2'][i])
                games_played += 1

                if games_played >= num_games:
                    break

        if games_played % 100 == 0:
            print(f"Games: {games_played}/{num_games} | "
                  f"Big: {big_wins} | Small: {small_wins} | Draws: {draws}")

    results = {
        'num_games': num_games,
        'big_wins': big_wins,
        'small_wins': small_wins,
        'draws': draws,
        'big_win_rate': big_wins / num_games,
        'small_win_rate': small_wins / num_games,
        'draw_rate': draws / num_games,
        'avg_big_score': float(np.mean(big_scores)),
        'avg_small_score': float(np.mean(small_scores)),
    }

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Games: {num_games}")
    print(f"Big (256x256) Wins: {big_wins} ({results['big_win_rate']:.1%})")
    print(f"Small (128x128) Wins: {small_wins} ({results['small_win_rate']:.1%})")
    print(f"Draws: {draws} ({results['draw_rate']:.1%})")
    print(f"Avg Scores: Big {results['avg_big_score']:.1f} vs Small {results['avg_small_score']:.1f}")
    print("="*70)

    return results


def train_model_wrapper(args):
    """Wrapper for parallel training"""
    config, model_size, device = args
    set_seed(config.seed + (0 if model_size == 'big' else 1))

    trainer = SingleModelCurriculumTrainer(config, model_size, device)
    agent = trainer.train_stages_0_to_3()

    return model_size, agent, trainer.win_rate_history


def main():
    parser = argparse.ArgumentParser(description='PPO Curriculum Training for Both Model Sizes')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='results/weights/ppo_curriculum_both')
    parser.add_argument('--sequential', action='store_true', help='Run sequentially instead of parallel')
    args = parser.parse_args()

    config = CurriculumConfig(
        num_envs=args.num_envs,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    device = get_device()
    set_seed(config.seed)

    print("="*70)
    print("PPO CURRICULUM TRAINING - BOTH MODEL SIZES")
    print("="*70)
    print(f"Device: {device}")
    print(f"Num envs: {config.num_envs}")
    print(f"Save dir: {config.save_dir}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print("="*70)

    total_start = time.time()

    # Phase 1: Train both models through Stages 0-3
    print("\n" + "="*70)
    print("PHASE 1: TRAINING STAGES 0-3 (PARALLEL)" if not args.sequential else "PHASE 1: TRAINING STAGES 0-3 (SEQUENTIAL)")
    print("="*70)

    if args.sequential:
        # Sequential training
        trainer_big = SingleModelCurriculumTrainer(config, 'big', device)
        big_agent = trainer_big.train_stages_0_to_3()
        big_history = trainer_big.win_rate_history

        trainer_small = SingleModelCurriculumTrainer(config, 'small', device)
        small_agent = trainer_small.train_stages_0_to_3()
        small_history = trainer_small.win_rate_history
    else:
        # Parallel training using threads (GPU operations are async)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(train_model_wrapper, (config, 'big', device)),
                executor.submit(train_model_wrapper, (config, 'small', device)),
            ]

            results = [f.result() for f in futures]

            for model_size, agent, history in results:
                if model_size == 'big':
                    big_agent = agent
                    big_history = history
                else:
                    small_agent = agent
                    small_history = history

    phase1_time = time.time() - total_start
    print(f"\nPhase 1 Complete! Time: {phase1_time/60:.1f} min")

    # Phase 2: Co-evolution (Stage 4)
    print("\n" + "="*70)
    print("PHASE 2: CO-EVOLUTION (STAGE 4)")
    print("="*70)

    coevo_trainer = CoEvolutionTrainer(big_agent, small_agent, config, device)
    big_final, small_final = coevo_trainer.train()
    coevo_history = coevo_trainer.win_rate_history

    phase2_time = time.time() - total_start - phase1_time
    print(f"\nPhase 2 Complete! Time: {phase2_time/60:.1f} min")

    # Phase 3: Head-to-head evaluation
    print("\n" + "="*70)
    print("PHASE 3: HEAD-TO-HEAD EVALUATION")
    print("="*70)

    results = evaluate_head_to_head(big_final, small_final, config, device, num_games=1000)

    # Save results
    results_path = Path(config.save_dir) / "head_to_head_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - total_start

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {config.save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
