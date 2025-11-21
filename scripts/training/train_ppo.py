"""
PPO Training Script

Trains PPO (Proximal Policy Optimization) agent with:
- Actor-Critic architecture
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Multiple epochs per batch
- GPU acceleration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal, List, Tuple
import time

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import PPO_Actor_MLP, PPO_Critic_MLP, PPO_Actor_CNN, PPO_Critic_CNN
from core.utils import MetricsTracker, set_seed, get_device


class PPOBuffer:
    """
    Rollout buffer for PPO training

    Stores trajectories for on-policy learning
    """

    def __init__(self, capacity: int, device: torch.device):
        """Initialize buffer"""
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
        """Add experience to buffer"""
        self.states.append(state.cpu())
        self.actions.append(action.cpu())
        self.rewards.append(reward.cpu())
        self.dones.append(done.cpu())
        self.log_probs.append(log_prob.cpu())
        self.values.append(value.cpu())
        self.size += state.shape[0]  # Number of parallel environments

    def get(self) -> Tuple:
        """Get all experiences and compute advantages"""
        states = torch.cat(self.states, dim=0).to(self.device)
        actions = torch.cat(self.actions, dim=0).to(self.device)
        rewards = torch.cat(self.rewards, dim=0).to(self.device)
        dones = torch.cat(self.dones, dim=0).to(self.device)
        log_probs = torch.cat(self.log_probs, dim=0).to(self.device)
        values = torch.cat(self.values, dim=0).to(self.device)

        return states, actions, rewards, dones, log_probs, values

    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.size >= self.capacity


class PPOTrainer:
    """PPO Training Manager"""

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

        # Training config
        actor_lr: float = 0.0003,
        critic_lr: float = 0.001,
        rollout_steps: int = 2048,
        batch_size: int = 64,
        epochs_per_rollout: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,

        # Training config
        num_episodes: int = 10000,
        max_steps: int = 1000,

        # Other
        seed: int = 42,
        device: Optional[torch.device] = None,
        save_dir: str = 'results/weights'
    ):
        """Initialize PPO trainer"""

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
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.epochs_per_rollout = epochs_per_rollout
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_episodes = num_episodes
        self.max_steps = max_steps
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

        # Determine input/output dimensions
        if state_representation == 'feature':
            input_dim = 14 if use_flood_fill else 11
        else:
            input_dim = grid_size

        if action_space_type == 'relative':
            output_dim = 3
        else:
            output_dim = 4

        # Create networks
        if state_representation == 'feature':
            self.actor = PPO_Actor_MLP(input_dim, output_dim, hidden_dims).to(self.device)
            self.critic = PPO_Critic_MLP(input_dim, hidden_dims).to(self.device)
        else:
            self.actor = PPO_Actor_CNN(grid_size, 3, output_dim).to(self.device)
            self.critic = PPO_Critic_CNN(grid_size, 3).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Buffer
        self.buffer = PPOBuffer(rollout_steps, self.device)

        # Metrics
        self.metrics = MetricsTracker()

        # Episode counter
        self.episode = 0

    def select_actions(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select actions using current policy

        Returns:
            actions: Selected actions
            log_probs: Log probabilities of actions
            values: State values from critic
        """
        with torch.no_grad():
            # Get action logits and values
            logits = self.actor(states)
            values = self.critic(states)

            # Sample actions from categorical distribution
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions, log_probs, values

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        # Convert dones to float for arithmetic operations
        dones = dones.float()

        advantages = []
        gae = 0

        # Append next value for bootstrapping
        values_with_next = torch.cat([values, next_value.unsqueeze(0)])

        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + values

        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        """Update actor and critic networks"""

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.epochs_per_rollout):
            # Generate random indices for mini-batches
            indices = torch.randperm(states.size(0))

            for start in range(0, states.size(0), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                logits = self.actor(batch_states)
                values = self.critic(batch_states)

                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Actor loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return (
            total_actor_loss / n_updates,
            total_critic_loss / n_updates,
            total_entropy / n_updates
        )

    def train(self, verbose: bool = True, log_interval: int = 100):
        """Main training loop"""

        print(f"Starting PPO Training...", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num envs: {self.num_envs}", flush=True)
        print(f"Episodes: {self.num_episodes}", flush=True)
        print(f"State representation: {self.state_representation}", flush=True)
        print(f"Action space: {self.action_space_type}", flush=True)
        print(flush=True)

        # Reset environment
        states = self.env.reset()

        # Episode tracking
        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, device=self.device)

        start_time = time.time()
        total_steps = 0

        while self.episode < self.num_episodes:
            # Collect rollout
            self.buffer.clear()

            # Ensure at least 1 iteration (rollout_steps should be >= num_envs)
            for _ in range(max(1, self.rollout_steps // self.num_envs)):
                # Select actions
                actions, log_probs, values = self.select_actions(states)

                # Step environment
                next_states, rewards, dones, info = self.env.step(actions)

                # Store transition
                self.buffer.add(states, actions, rewards, dones, log_probs, values.squeeze())

                # Update episode stats
                episode_rewards += rewards
                episode_lengths += 1
                total_steps += self.num_envs

                # Check for done episodes
                if dones.any():
                    done_indices = torch.where(dones)[0]
                    for idx in done_indices:
                        # Record episode
                        self.metrics.add_episode(
                            episode_rewards[idx].item(),
                            episode_lengths[idx].item(),
                            info['scores'][idx].item()
                        )

                        self.episode += 1

                        # Reset episode stats
                        episode_rewards[idx] = 0
                        episode_lengths[idx] = 0

                        # Logging
                        if verbose and self.episode % log_interval == 0:
                            stats = self.metrics.get_recent_stats()
                            elapsed = time.time() - start_time
                            fps = total_steps / elapsed if elapsed > 0 else 0

                            print(f"\nEpisode {self.episode}/{self.num_episodes}")
                            print(f"  Avg Score: {stats['avg_score']:.2f}")
                            print(f"  Avg Reward: {stats['avg_reward']:.2f}")
                            print(f"  Avg Length: {stats['avg_length']:.2f}")
                            print(f"  Max Score: {stats['max_score']}")
                            print(f"  FPS: {fps:.0f}")

                        if self.episode >= self.num_episodes:
                            break

                states = next_states

                if self.episode >= self.num_episodes:
                    break

            if self.episode >= self.num_episodes:
                break

            # Compute final value for bootstrapping
            with torch.no_grad():
                next_value = self.critic(states).squeeze()

            # Get buffer data
            buffer_states, buffer_actions, buffer_rewards, buffer_dones, buffer_log_probs, buffer_values = self.buffer.get()

            # Reshape for GAE computation (steps, envs)
            steps_per_env = len(self.buffer.states)
            buffer_rewards = buffer_rewards.view(steps_per_env, self.num_envs)
            buffer_values = buffer_values.view(steps_per_env, self.num_envs)
            buffer_dones = buffer_dones.view(steps_per_env, self.num_envs)

            # Compute advantages and returns for each environment
            all_advantages = []
            all_returns = []

            for env_idx in range(self.num_envs):
                advantages, returns = self.compute_gae(
                    buffer_rewards[:, env_idx],
                    buffer_values[:, env_idx],
                    buffer_dones[:, env_idx],
                    next_value[env_idx]
                )
                all_advantages.append(advantages)
                all_returns.append(returns)

            advantages = torch.stack(all_advantages, dim=1).view(-1)
            returns = torch.stack(all_returns, dim=1).view(-1)

            # Update networks
            actor_loss, critic_loss, entropy = self.update(
                buffer_states,
                buffer_actions,
                buffer_log_probs,
                advantages,
                returns
            )

        print("\nTraining complete!", flush=True)
        print(f"Training finished after {self.episode} episodes")

    def save(self, filename: str):
        """Save model weights"""
        filepath = self.save_dir / filename
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': self.episode,
            'metrics': self.metrics.get_recent_stats()
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.episode = checkpoint['episode']


if __name__ == '__main__':
    # Test training - 500 episodes
    trainer = PPOTrainer(
        num_envs=256,
        num_episodes=500,
        state_representation='feature',
        action_space_type='relative',
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        rollout_steps=2048,  # Should be >= num_envs for proper rollout collection
        epochs_per_rollout=4
    )
    trainer.train(verbose=True, log_interval=50)
