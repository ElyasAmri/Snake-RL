"""
DQN Training Script

Trains DQN agent with:
- Experience replay
- Target networks
- Epsilon-greedy exploration
- Curriculum learning (optional)
- GPU acceleration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Literal
import time

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DQN_MLP, DQN_CNN, DuelingDQN_MLP, NoisyDQN_MLP
from core.utils import ReplayBuffer, PrioritizedReplayBuffer, EpsilonScheduler, MetricsTracker, set_seed, get_device


class DQNTrainer:
    """DQN Training Manager"""

    def __init__(
        self,
        # Environment config
        num_envs: int = 256,
        grid_size: int = 10,
        action_space_type: Literal['absolute', 'relative'] = 'relative',
        state_representation: Literal['feature', 'grid'] = 'feature',
        use_flood_fill: bool = False,
        use_enhanced_features: bool = False,
        use_selective_features: bool = False,

        # Network config
        hidden_dims: tuple = (128, 128),
        use_dueling: bool = False,
        use_noisy: bool = False,
        noisy_sigma: float = 0.5,

        # DQN variant config
        use_double_dqn: bool = False,
        use_prioritized_replay: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
        use_multistep: bool = False,
        n_step: int = 3,

        # Training config
        learning_rate: float = 0.001,
        batch_size: int = 64,
        buffer_size: int = 100000,
        gamma: float = 0.99,
        target_update_freq: int = 1000,

        # Exploration config
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,

        # Training config
        num_episodes: int = 10000,
        max_steps: int = 1000,
        min_buffer_size: int = 1000,
        train_steps_ratio: float = 0.03125,  # Train once per ~32 collected transitions (fast default)

        # Curriculum learning
        use_curriculum: bool = False,
        curriculum_stages: list = None,

        # Other
        seed: int = 42,
        device: Optional[torch.device] = None,
        save_dir: str = 'results/weights',
        max_time: Optional[int] = None  # Maximum training time in seconds
    ):
        """Initialize DQN trainer"""
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
        self.use_enhanced_features = use_enhanced_features
        self.use_selective_features = use_selective_features
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay
        self.use_multistep = use_multistep
        self.n_step = n_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.min_buffer_size = min_buffer_size
        self.train_steps_ratio = train_steps_ratio
        self.use_curriculum = use_curriculum
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
            use_enhanced_features=use_enhanced_features,
            use_selective_features=use_selective_features,
            device=self.device
        )

        # Create networks
        if state_representation == 'feature':
            # Determine input dimension based on feature configuration
            input_dim = 11
            if use_flood_fill:
                input_dim = 14
            if use_selective_features:
                input_dim = 19
            if use_enhanced_features:
                input_dim = 24

            if use_noisy:
                # Noisy DQN with NoisyLinear layers
                self.policy_net = NoisyDQN_MLP(
                    input_dim=input_dim,
                    output_dim=self.env.action_space.n,
                    hidden_dims=hidden_dims,
                    sigma_init=noisy_sigma
                ).to(self.device)

                self.target_net = NoisyDQN_MLP(
                    input_dim=input_dim,
                    output_dim=self.env.action_space.n,
                    hidden_dims=hidden_dims,
                    sigma_init=noisy_sigma
                ).to(self.device)
            elif use_dueling:
                self.policy_net = DuelingDQN_MLP(
                    input_dim=input_dim,
                    output_dim=self.env.action_space.n,
                    hidden_dims=hidden_dims
                ).to(self.device)

                self.target_net = DuelingDQN_MLP(
                    input_dim=input_dim,
                    output_dim=self.env.action_space.n,
                    hidden_dims=hidden_dims
                ).to(self.device)
            else:
                self.policy_net = DQN_MLP(
                    input_dim=input_dim,
                    output_dim=self.env.action_space.n,
                    hidden_dims=hidden_dims
                ).to(self.device)

                self.target_net = DQN_MLP(
                    input_dim=input_dim,
                    output_dim=self.env.action_space.n,
                    hidden_dims=hidden_dims
                ).to(self.device)
        else:  # grid
            if use_noisy:
                raise ValueError("Noisy DQN currently only supports feature-based state (MLP)")

            self.policy_net = DQN_CNN(
                grid_size=grid_size,
                input_channels=3,
                output_dim=self.env.action_space.n
            ).to(self.device)

            self.target_net = DQN_CNN(
                grid_size=grid_size,
                input_channels=3,
                output_dim=self.env.action_space.n
            ).to(self.device)

        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
                seed=seed
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)

        # Epsilon scheduler
        self.epsilon_scheduler = EpsilonScheduler(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            decay_type='exponential'
        )

        # Metrics
        self.metrics = MetricsTracker(window_size=100)

        # Training state
        self.total_steps = 0
        self.episode = 0

    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Select actions using epsilon-greedy policy (or greedy for noisy networks)"""
        with torch.no_grad():
            # Reset noise for noisy networks
            if self.use_noisy:
                self.policy_net.reset_noise()

            q_values = self.policy_net(states)
            greedy_actions = q_values.argmax(dim=1)

        # Noisy networks don't need epsilon-greedy (noise provides exploration)
        if self.use_noisy:
            return greedy_actions

        # Epsilon-greedy for non-noisy networks
        epsilon = self.epsilon_scheduler.get_epsilon()
        random_mask = torch.rand(self.num_envs, device=self.device) < epsilon

        random_actions = torch.randint(
            0, self.env.action_space.n,
            (self.num_envs,),
            device=self.device
        )

        actions = torch.where(random_mask, random_actions, greedy_actions)
        return actions

    def train_step(self):
        """Perform one training step"""
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Reset noise for noisy networks
        if self.use_noisy:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        # Sample batch
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy net to select actions, target net to evaluate
                best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze()
            else:
                # Vanilla DQN: use target net for both selection and evaluation
                next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute TD errors for PER priority updates
        td_errors = target_q_values - current_q_values.squeeze()

        # Compute loss (use Huber loss for robustness to large rewards)
        if self.use_prioritized_replay:
            # Weighted Huber loss for PER
            elementwise_loss = nn.functional.smooth_l1_loss(
                current_q_values.squeeze(), target_q_values, reduction='none'
            )
            loss = (weights * elementwise_loss).mean()
        else:
            # Huber loss (smooth_l1_loss) is more robust to outliers than MSE
            loss = nn.functional.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in PER buffer
        if self.use_prioritized_replay:
            td_errors_cpu = td_errors.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors_cpu)

        return loss.item()

    def train(self, verbose: bool = True, log_interval: int = 100):
        """Main training loop"""

        print("Starting DQN Training...", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Num envs: {self.num_envs}", flush=True)
        print(f"Episodes: {self.num_episodes}", flush=True)
        print(f"State representation: {self.state_representation}", flush=True)
        print(f"Action space: {self.action_space_type}", flush=True)
        print(flush=True)

        # Initialize
        states = self.env.reset(seed=42)
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

            # Store transitions from ALL environments
            for i in range(self.num_envs):
                self.replay_buffer.push(
                    states[i].cpu().numpy(),
                    actions[i].item(),
                    rewards[i].item(),
                    next_states[i].cpu().numpy(),
                    dones[i].item()
                )

            # Train multiple times per step based on train_steps_ratio
            # With N parallel envs, we collect N transitions per step
            # train_steps_ratio controls how many training steps per collected transition
            if self.replay_buffer.is_ready(self.min_buffer_size):
                # Configurable training frequency (default: ~8 steps with 256 envs)
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

                    # Decay epsilon
                    self.epsilon_scheduler.step()

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
                        print(f"  Epsilon: {self.epsilon_scheduler.get_epsilon():.4f}", flush=True)
                        print(f"  FPS: {fps:.0f}", flush=True)
                        print(f"  Buffer: {len(self.replay_buffer)}/{self.replay_buffer.capacity}", flush=True)
                        print(flush=True)

                    # Early exit if reached episode limit
                    if self.episode >= self.num_episodes:
                        break

                    # Early exit if max_time reached
                    if self.max_time and (time.time() - start_time) >= self.max_time:
                        print(f"\n[TIMEOUT] Max time {self.max_time}s reached. Stopping training...", flush=True)
                        break

            # Update state
            states = next_states
            # FIX: Count all parallel environment steps, not just iterations
            self.total_steps += self.num_envs

            # Check max_time at outer loop level
            if self.max_time and (time.time() - start_time) >= self.max_time:
                break

        print("Training complete!")
        print(f"Training finished after {self.episode} episodes")

    def save(self, filename: str):
        """Save model"""
        filepath = self.save_dir / filename
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon_scheduler.get_epsilon()
        }, filepath)
        print(f"Model saved to {filepath}")


if __name__ == '__main__':
    # Test training - 500 episodes
    trainer = DQNTrainer(
        num_envs=256,
        num_episodes=500,
        state_representation='feature',
        action_space_type='relative',
        buffer_size=100000,
        learning_rate=0.001,
        batch_size=64,
        gamma=0.95,
        epsilon_decay=0.995,
        target_update_freq=1000
    )
    trainer.train(verbose=True, log_interval=50)
