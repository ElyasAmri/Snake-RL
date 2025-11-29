"""
A2C (Advantage Actor-Critic) Training Script

Synchronous advantage actor-critic algorithm
Simpler than PPO - no clipping, just advantage estimation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal
import time

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import PPO_Actor_MLP, PPO_Actor_CNN, PPO_Critic_MLP, PPO_Critic_CNN
from core.utils import MetricsTracker, set_seed, get_device


class A2CTrainer:
    """A2C Training Manager"""

    def __init__(
        self,
        # Environment config
        num_envs: int = 256,
        grid_size: int = 10,
        action_space_type: Literal['absolute', 'relative'] = 'relative',
        state_representation: Literal['feature', 'grid'] = 'feature',
        use_flood_fill: bool = False,

        # Network config
        hidden_dims: tuple = (128, 128),

        # A2C config
        actor_lr: float = 0.0003,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,

        # Training config
        num_episodes: int = 10000,
        max_steps: int = 1000,
        rollout_steps: int = 5,  # A2C uses short rollouts (5-20 steps)

        # Other
        seed: int = 67,
        device: Optional[torch.device] = None,
        save_dir: str = 'results/weights',
        max_time: Optional[int] = None  # Maximum training time in seconds
    ):
        """Initialize A2C trainer"""
        self.max_time = max_time

        # Set seed
        set_seed(seed)

        # Device
        self.device = device if device else get_device()

        # Store config
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.action_space_type = action_space_type
        self.state_representation = state_representation
        self.use_flood_fill = use_flood_fill
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.rollout_steps = rollout_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create environment
        self.env = VectorizedSnakeEnv(
            num_envs=num_envs,
            grid_size=grid_size,
            action_space_type=action_space_type,
            state_representation=state_representation,
            max_steps=max_steps,
            use_flood_fill=use_flood_fill,
            device=self.device
        )

        # Create networks
        if state_representation == 'feature':
            input_dim = 14 if use_flood_fill else 11

            self.actor = PPO_Actor_MLP(
                input_dim=input_dim,
                output_dim=self.env.action_space.n,
                hidden_dims=hidden_dims
            ).to(self.device)

            self.critic = PPO_Critic_MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
        else:  # grid
            self.actor = PPO_Actor_CNN(
                grid_size=grid_size,
                input_channels=3,
                output_dim=self.env.action_space.n
            ).to(self.device)

            self.critic = PPO_Critic_CNN(
                grid_size=grid_size,
                input_channels=3
            ).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Metrics
        self.metrics = MetricsTracker(window_size=100)

        # Training state
        self.total_steps = 0
        self.episode = 0

    def select_actions(self, states: torch.Tensor):
        """Select actions from policy"""
        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def compute_returns(self, rewards, dones, next_value):
        """Compute returns using n-step bootstrapping"""
        returns = []
        R = next_value

        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)

        return torch.stack(returns)

    def update(self, states, actions, returns, log_probs_old):
        """Update actor and critic networks"""
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        returns = returns.detach()
        log_probs_old = torch.stack(log_probs_old).detach()

        # Critic update
        values = self.critic(states).squeeze()
        advantages = returns - values.detach()

        critic_loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update
        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def train(self, verbose: bool = True, log_interval: int = 100):
        """Main training loop"""

        print("Starting A2C Training...", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num envs: {self.num_envs}", flush=True)
        print(f"Episodes: {self.num_episodes}", flush=True)
        print(f"State representation: {self.state_representation}", flush=True)
        print(f"Action space: {self.action_space_type}", flush=True)
        print(flush=True)

        # Initialize
        states = self.env.reset()
        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, device=self.device)
        completed_episodes = 0
        start_time = time.time()

        while completed_episodes < self.num_episodes:
            # Collect rollout
            rollout_states = []
            rollout_actions = []
            rollout_rewards = []
            rollout_dones = []
            rollout_log_probs = []

            for _ in range(self.rollout_steps):
                # Select actions
                actions, log_probs = self.select_actions(states)

                # Environment step
                next_states, rewards, dones, info = self.env.step(actions)

                # Store transition
                rollout_states.append(states)
                rollout_actions.append(actions)
                rollout_rewards.append(rewards)
                rollout_dones.append(dones.float())
                rollout_log_probs.append(log_probs)

                # Update episode stats
                episode_rewards += rewards
                episode_lengths += 1

                # Handle episode completion
                for i in range(self.num_envs):
                    if dones[i]:
                        # Determine death cause
                        if info['wall_deaths'][i].item():
                            death_cause = 'wall'
                        elif info['self_deaths'][i].item():
                            death_cause = 'self'
                        else:
                            death_cause = 'timeout'

                        self.metrics.add_episode(
                            reward=episode_rewards[i].item(),
                            length=int(episode_lengths[i].item()),
                            score=int(info['scores'][i].item()),
                            death_cause=death_cause
                        )
                        episode_rewards[i] = 0
                        episode_lengths[i] = 0
                        completed_episodes += 1

                        if completed_episodes >= self.num_episodes:
                            break

                        # Early exit if max_time reached
                        if self.max_time and (time.time() - start_time) >= self.max_time:
                            print(f"\n[TIMEOUT] Max time {self.max_time}s reached. Stopping training...", flush=True)
                            break

                states = next_states

                if completed_episodes >= self.num_episodes:
                    break
                if self.max_time and (time.time() - start_time) >= self.max_time:
                    break

            if completed_episodes >= self.num_episodes:
                break
            if self.max_time and (time.time() - start_time) >= self.max_time:
                break

            # Compute returns
            with torch.no_grad():
                next_value = self.critic(states).squeeze()
            returns = self.compute_returns(rollout_rewards, rollout_dones, next_value)

            # Update networks
            actor_loss, critic_loss = self.update(
                rollout_states,
                rollout_actions,
                returns,
                rollout_log_probs
            )

            self.metrics.add_loss(actor_loss + critic_loss)

            # Logging
            if verbose and completed_episodes % log_interval == 0 and completed_episodes > 0:
                stats = self.metrics.get_recent_stats()
                print(
                    f"Episode {completed_episodes}/{self.num_episodes} | "
                    f"Avg Reward: {stats['avg_reward']:.2f} | "
                    f"Avg Score: {stats['avg_score']:.2f} | "
                    f"Avg Length: {stats['avg_length']:.2f} | "
                    f"Loss: {stats.get('avg_loss', 0):.4f}",
                    flush=True
                )

        print("\nTraining complete!", flush=True)
        print(f"Training finished after {completed_episodes} episodes")

    def save(self, filename: str):
        """Save model weights"""
        save_path = self.save_dir / filename
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, save_path)
        return save_path


if __name__ == '__main__':
    # Test training - 500 episodes
    trainer = A2CTrainer(
        num_envs=256,
        num_episodes=500,
        state_representation='feature',
        action_space_type='relative',
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        rollout_steps=5
    )
    trainer.train(verbose=True, log_interval=50)
