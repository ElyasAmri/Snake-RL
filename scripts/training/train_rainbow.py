"""
Rainbow DQN Training Script

Trains Rainbow DQN agent combining all DQN improvements:
- Double DQN
- Prioritized Experience Replay
- Dueling Architecture
- Noisy Networks
- N-step Returns
- Distributional RL (Categorical/C51)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Literal
import time

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import RainbowDQN_MLP
from core.utils import (
    PrioritizedNStepReplayBuffer,
    MetricsTracker,
    set_seed,
    get_device,
    project_distribution,
    categorical_dqn_loss
)


class RainbowTrainer:
    """Rainbow DQN Training Manager"""

    def __init__(
        self,
        # Environment config
        num_envs: int = 256,
        grid_size: int = 10,
        action_space_type: Literal['absolute', 'relative'] = 'relative',
        use_flood_fill: bool = False,
        use_enhanced_features: bool = False,
        use_selective_features: bool = False,

        # Network config
        hidden_dims: tuple = (128, 128),
        n_atoms: int = 51,
        v_min: float = -20.0,  # Must cover min possible return (death penalty)
        v_max: float = 500.0,  # Must cover max possible return (high scores)
        noisy_sigma: float = 0.5,

        # N-step config
        n_step: int = 3,

        # PER config
        per_alpha: float = 0.5,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,

        # Training config
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        buffer_size: int = 100000,
        gamma: float = 0.99,
        target_update_freq: int = 500,

        # Training config
        num_episodes: int = 3000,
        max_steps: int = 1000,
        min_buffer_size: int = 1000,
        train_steps_ratio: float = 0.03125,

        # Reward config
        reward_death: float = -10.0,

        # Other
        seed: int = 67,
        device: Optional[torch.device] = None,
        save_dir: str = 'results/weights',
        max_time: Optional[int] = None
    ):
        """Initialize Rainbow DQN trainer"""
        self.max_time = max_time

        # Set seed
        set_seed(seed)

        # Device
        self.device = device if device else get_device()

        # Store config
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.action_space_type = action_space_type
        self.use_flood_fill = use_flood_fill
        self.use_enhanced_features = use_enhanced_features
        self.use_selective_features = use_selective_features
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_step = n_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.min_buffer_size = min_buffer_size
        self.train_steps_ratio = train_steps_ratio
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create environment
        self.env = VectorizedSnakeEnv(
            num_envs=num_envs,
            grid_size=grid_size,
            action_space_type=action_space_type,
            state_representation='feature',
            max_steps=max_steps,
            reward_death=reward_death,
            use_flood_fill=use_flood_fill,
            use_enhanced_features=use_enhanced_features,
            use_selective_features=use_selective_features,
            device=self.device
        )

        # Determine input dimension
        input_dim = 10  # Base features
        if use_flood_fill:
            input_dim = 13
        if use_selective_features:
            input_dim = 18
        if use_enhanced_features:
            input_dim = 23

        # Create Rainbow networks
        self.policy_net = RainbowDQN_MLP(
            input_dim=input_dim,
            output_dim=self.env.action_space.n,
            hidden_dims=hidden_dims,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            sigma_init=noisy_sigma
        ).to(self.device)

        self.target_net = RainbowDQN_MLP(
            input_dim=input_dim,
            output_dim=self.env.action_space.n,
            hidden_dims=hidden_dims,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            sigma_init=noisy_sigma
        ).to(self.device)

        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer (Adam with lower learning rate for Rainbow)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Prioritized N-step replay buffer (with per-env n-step buffers for vectorized training)
        self.replay_buffer = PrioritizedNStepReplayBuffer(
            capacity=buffer_size,
            n_step=n_step,
            gamma=gamma,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
            num_envs=num_envs,
            seed=seed
        )

        # Metrics
        self.metrics = MetricsTracker(window_size=100)

        # Training state
        self.total_steps = 0
        self.episode = 0

    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Select actions using noisy network (no epsilon needed)"""
        with torch.no_grad():
            # Reset noise for exploration
            self.policy_net.reset_noise()

            # Get Q-values from distributional network
            q_values = self.policy_net.get_q_values(states)
            actions = q_values.argmax(dim=1)

        return actions

    def train_step(self):
        """Perform one training step with distributional loss"""
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Sample batch from prioritized n-step buffer
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Get current distributions for selected actions
        current_dist = self.policy_net(states)  # (batch, n_actions, n_atoms)
        current_dist = current_dist[
            torch.arange(self.batch_size, device=self.device),
            actions
        ]  # (batch, n_atoms)

        # Compute target distribution using Double DQN style
        with torch.no_grad():
            # Use policy net to select best actions (Double DQN)
            next_q_values = self.policy_net.get_q_values(next_states)
            best_actions = next_q_values.argmax(dim=1)

            # Use target net to get distribution for selected actions
            next_dist = self.target_net(next_states)  # (batch, n_actions, n_atoms)
            next_dist = next_dist[
                torch.arange(self.batch_size, device=self.device),
                best_actions
            ]  # (batch, n_atoms)

            # Project target distribution onto support
            target_dist = project_distribution(
                next_dist=next_dist,
                rewards=rewards,
                dones=dones,
                support=self.policy_net.support,
                gamma=self.gamma,
                n_step=self.n_step
            )

        # Compute cross-entropy loss with proper TD errors for PER
        loss, td_errors = categorical_dqn_loss(
            current_dist, target_dist, self.policy_net.support, weights
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Conservative clipping
        self.optimizer.step()

        # Update priorities
        td_errors_cpu = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors_cpu)

        return loss.item()

    def train(self, verbose: bool = True, log_interval: int = 100):
        """Main training loop"""

        print("Starting Rainbow DQN Training...", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num envs: {self.num_envs}", flush=True)
        print(f"Episodes: {self.num_episodes}", flush=True)
        print(f"N-step: {self.n_step}", flush=True)
        print(f"Atoms: {self.n_atoms}, V_min: {self.v_min}, V_max: {self.v_max}", flush=True)
        print(flush=True)

        # Initialize
        states = self.env.reset(seed=67)
        start_time = time.time()

        # Episode tracking
        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        while self.episode < self.num_episodes:
            # Select actions
            actions = self.select_actions(states)

            # Step environment
            next_states, rewards, dones, info = self.env.step(actions)

            # Accumulate episode stats
            episode_rewards += rewards
            episode_lengths += 1

            # Store transitions in n-step buffer (with per-env tracking)
            for i in range(self.num_envs):
                self.replay_buffer.push(
                    states[i].cpu().numpy(),
                    actions[i].item(),
                    rewards[i].item(),
                    next_states[i].cpu().numpy(),
                    dones[i].item(),
                    env_id=i  # Critical: track per-environment for proper n-step returns
                )

            # Training
            if self.replay_buffer.is_ready(self.min_buffer_size):
                num_train_steps = max(1, int(self.num_envs * self.train_steps_ratio))
                for _ in range(num_train_steps):
                    loss = self.train_step()
                    if loss is not None:
                        self.metrics.add_loss(loss)

            # Update target network
            if self.total_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Check for done episodes
            if dones.any():
                done_indices = torch.where(dones)[0]
                for idx in done_indices:
                    # Determine death cause
                    if info['entrapments'][idx].item():
                        death_cause = 'entrapment'
                    elif info['wall_deaths'][idx].item():
                        death_cause = 'wall'
                    elif info['self_deaths'][idx].item():
                        death_cause = 'self'
                    else:
                        death_cause = 'timeout'

                    # Record episode
                    self.metrics.add_episode(
                        episode_rewards[idx].item(),
                        episode_lengths[idx].item(),
                        info['scores'][idx].item(),
                        death_cause
                    )

                    self.episode += 1

                    # Reset episode stats
                    episode_rewards[idx] = 0
                    episode_lengths[idx] = 0

                    # Logging
                    if verbose and self.episode % log_interval == 0:
                        stats = self.metrics.get_recent_stats()
                        elapsed = time.time() - start_time
                        fps = self.total_steps / elapsed if elapsed > 0 else 0

                        print(f"Episode {self.episode}/{self.num_episodes}", flush=True)
                        print(f"  Avg Score: {stats['avg_score']:.2f}", flush=True)
                        print(f"  Avg Reward: {stats['avg_reward']:.2f}", flush=True)
                        print(f"  Avg Length: {stats['avg_length']:.2f}", flush=True)
                        print(f"  Max Score: {stats['max_score']}", flush=True)
                        if 'avg_loss' in stats:
                            print(f"  Avg Loss: {stats['avg_loss']:.4f}", flush=True)
                        print(f"  FPS: {fps:.0f}", flush=True)
                        print(f"  Buffer: {len(self.replay_buffer)}/{self.replay_buffer.capacity}", flush=True)
                        print(flush=True)

                    # Early exit conditions
                    if self.episode >= self.num_episodes:
                        break

                    if self.max_time and (time.time() - start_time) >= self.max_time:
                        print(f"\n[TIMEOUT] Max time {self.max_time}s reached.", flush=True)
                        break

            # Update state
            states = next_states
            self.total_steps += self.num_envs

            # Check max_time
            if self.max_time and (time.time() - start_time) >= self.max_time:
                break

        elapsed = time.time() - start_time
        print("Training complete!")
        print(f"Training finished after {self.episode} episodes in {elapsed:.1f}s")

    def save(self, filename: str):
        """Save model"""
        filepath = self.save_dir / filename
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'total_steps': self.total_steps,
            'n_atoms': self.n_atoms,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'n_step': self.n_step
        }, filepath)
        print(f"Model saved to {filepath}")


if __name__ == '__main__':
    # Test training - 500 episodes
    trainer = RainbowTrainer(
        num_envs=256,
        num_episodes=500,
        grid_size=10,
        action_space_type='relative',
        use_flood_fill=True,
        buffer_size=100000,
        learning_rate=0.001,
        batch_size=64,
        gamma=0.99,
        n_step=1,
        n_atoms=51,
        v_min=-20.0,  # Must cover min possible return
        v_max=500.0,  # Must cover max possible return
        target_update_freq=1000
    )
    trainer.train(verbose=True, log_interval=50)
