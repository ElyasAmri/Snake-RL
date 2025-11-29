"""
REINFORCE Training Script

Trains REINFORCE agent with:
- Policy gradient (Monte Carlo)
- No critic (unlike Actor-Critic methods)
- Episode-based updates
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
from core.networks import PPO_Actor_MLP, PPO_Actor_CNN
from core.utils import MetricsTracker, set_seed, get_device


class REINFORCEBuffer:
    """
    Episode buffer for REINFORCE training

    Stores complete episodes for Monte Carlo returns
    """

    def __init__(self, device: torch.device):
        """Initialize buffer"""
        self.device = device
        self.clear()

    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """Add experience to buffer"""
        self.states.append(state.cpu())
        self.actions.append(action.cpu())
        self.rewards.append(reward.cpu())
        self.log_probs.append(log_prob.cpu())

    def get(self) -> Tuple:
        """Get all experiences"""
        states = torch.cat(self.states, dim=0).to(self.device)
        actions = torch.cat(self.actions, dim=0).to(self.device)
        rewards = torch.cat(self.rewards, dim=0).to(self.device)
        log_probs = torch.cat(self.log_probs, dim=0).to(self.device)

        return states, actions, rewards, log_probs

    def size(self) -> int:
        """Get buffer size"""
        return sum(s.shape[0] for s in self.states) if self.states else 0


class REINFORCETrainer:
    """REINFORCE Training Manager"""

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
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,

        # Training config
        num_episodes: int = 10000,
        max_steps: int = 1000,

        # Other
        seed: int = 67,
        device: Optional[torch.device] = None,
        save_dir: str = 'results/weights',
        max_time: Optional[int] = None  # Maximum training time in seconds
    ):
        """Initialize REINFORCE trainer"""
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

        # Create policy network (reuse PPO actor architecture)
        if state_representation == 'feature':
            self.policy = PPO_Actor_MLP(input_dim, output_dim, hidden_dims).to(self.device)
        else:
            self.policy = PPO_Actor_CNN(grid_size, 3, output_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Buffer
        self.buffer = REINFORCEBuffer(self.device)

        # Metrics
        self.metrics = MetricsTracker()

        # Episode counter
        self.episode = 0

    def select_actions(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select actions using current policy

        Returns:
            actions: Selected actions
            log_probs: Log probabilities of actions
        """
        # Get action logits
        logits = self.policy(states)

        # Sample actions from categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute discounted returns (Monte Carlo)

        Returns:
            returns: Discounted returns for each timestep
        """
        returns = []
        G = 0

        # Compute returns backwards
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self):
        """Update policy network"""

        if self.buffer.size() == 0:
            return None

        # Get buffer data
        states, actions, rewards, log_probs = self.buffer.get()

        # Compute returns
        returns = self.compute_returns(rewards)

        # Compute policy loss
        policy_loss = -(log_probs * returns).mean()

        # Compute entropy bonus
        logits = self.policy(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()

        # Total loss
        loss = policy_loss - self.entropy_coef * entropy

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def train(self, verbose: bool = True, log_interval: int = 100):
        """Main training loop"""

        print(f"Starting REINFORCE Training...", flush=True)
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
            # Clear buffer for new episodes
            self.buffer.clear()

            # Collect episode
            step = 0
            while step < self.max_steps and self.episode < self.num_episodes:
                # Select actions
                with torch.no_grad():
                    actions, log_probs = self.select_actions(states)

                # Step environment
                next_states, rewards, dones, info = self.env.step(actions)

                # Store transition
                self.buffer.add(states, actions, rewards, log_probs)

                # Update episode stats
                episode_rewards += rewards
                episode_lengths += 1
                total_steps += self.num_envs
                step += 1

                # Check for done episodes
                if dones.any():
                    done_indices = torch.where(dones)[0]
                    for idx in done_indices:
                        # Determine death cause
                        if info['wall_deaths'][idx].item():
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
                            fps = total_steps / elapsed if elapsed > 0 else 0

                            print(f"\nEpisode {self.episode}/{self.num_episodes}")
                            print(f"  Avg Score: {stats['avg_score']:.2f}")
                            print(f"  Avg Reward: {stats['avg_reward']:.2f}")
                            print(f"  Avg Length: {stats['avg_length']:.2f}")
                            print(f"  Max Score: {stats['max_score']}")
                            print(f"  FPS: {fps:.0f}")

                        if self.episode >= self.num_episodes:
                            break

                        # Early exit if max_time reached
                        if self.max_time and (time.time() - start_time) >= self.max_time:
                            print(f"\n[TIMEOUT] Max time {self.max_time}s reached. Stopping training...", flush=True)
                            break

                states = next_states

                if self.episode >= self.num_episodes:
                    break
                if self.max_time and (time.time() - start_time) >= self.max_time:
                    break

            # Update policy after collecting episodes
            if self.buffer.size() > 0:
                loss = self.update()
                if loss is not None:
                    self.metrics.add_loss(loss)

        print("\nTraining complete!", flush=True)
        print(f"Training finished after {self.episode} episodes")

    def save(self, filename: str):
        """Save model weights"""
        filepath = self.save_dir / filename
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'metrics': self.metrics.get_recent_stats()
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode = checkpoint['episode']


if __name__ == '__main__':
    # Test training - 500 episodes
    trainer = REINFORCETrainer(
        num_envs=256,
        num_episodes=500,
        state_representation='feature',
        action_space_type='relative',
        learning_rate=0.001,
        gamma=0.99
    )
    trainer.train(verbose=True, log_interval=50)
