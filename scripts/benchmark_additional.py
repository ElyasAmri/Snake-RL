"""
Additional Performance Benchmark Tests

Tests other potential optimizations:
1. Reduce num_envs (test if per-env loops are bottleneck)
2. Different training frequencies (1, 4, 16, 64 steps)
3. Batch size variations
4. Target update frequency
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import time
from datetime import datetime

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DQN_MLP
from core.utils import ReplayBuffer, MetricsTracker, set_seed, get_device, EpsilonScheduler


class AdditionalBenchmark:
    """Additional optimization tests"""

    def __init__(
        self,
        num_envs: int = 256,
        num_episodes: int = 1000,
        num_train_steps: int = 1,
        batch_size: int = 256,
        device: torch.device = None
    ):
        set_seed(42)
        self.device = device if device else get_device()
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        # Training config
        self.gamma = 0.99
        self.buffer_size = 100000
        self.target_update_freq = 1000
        self.min_buffer_size = 1000

        # Create environment
        self.env = VectorizedSnakeEnv(
            num_envs=num_envs,
            grid_size=10,
            action_space_type='relative',
            state_representation='feature',
            use_flood_fill=False,
            device=self.device
        )

        # Create networks
        self.policy_net = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(self.device)
        self.target_net = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0005)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, seed=42)

        # Epsilon scheduler
        self.epsilon_scheduler = EpsilonScheduler(
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=5000
        )

        # Metrics
        self.metrics = MetricsTracker(window_size=100)

        # Training state
        self.total_steps = 0
        self.episode = 0

    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Select actions using epsilon-greedy policy"""
        with torch.no_grad():
            q_values = self.policy_net(states)
            greedy_actions = q_values.argmax(dim=1)

        epsilon = self.epsilon_scheduler.get_epsilon()
        random_mask = torch.rand(self.num_envs, device=self.device) < epsilon
        random_actions = torch.randint(0, 3, (self.num_envs,), device=self.device)
        actions = torch.where(random_mask, random_actions, greedy_actions)
        return actions

    def train_step(self):
        """Perform one training step"""
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def train(self):
        """Main training loop with timing"""

        start_time = time.time()

        # Initialize
        states = self.env.reset(seed=42)
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

            # Store transitions - OPTIMIZED (batch transfer)
            states_np = states.cpu().numpy()
            actions_np = actions.cpu().numpy()
            rewards_np = rewards.cpu().numpy()
            next_states_np = next_states.cpu().numpy()
            dones_np = dones.cpu().numpy()

            for i in range(self.num_envs):
                self.replay_buffer.push(
                    states_np[i],
                    actions_np[i],
                    rewards_np[i],
                    next_states_np[i],
                    dones_np[i]
                )

            # Train
            if self.replay_buffer.is_ready(self.min_buffer_size):
                for _ in range(self.num_train_steps):
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
                    self.metrics.add_episode(
                        episode_rewards[idx].item(),
                        episode_lengths[idx].item(),
                        info['scores'][idx].item()
                    )
                    self.episode += 1
                    episode_rewards[idx] = 0
                    episode_lengths[idx] = 0
                    self.epsilon_scheduler.step()

                    if self.episode >= self.num_episodes:
                        break

            states = next_states
            self.total_steps += 1

        total_time = time.time() - start_time
        stats = self.metrics.get_recent_stats()

        return {
            'num_envs': self.num_envs,
            'num_train_steps': self.num_train_steps,
            'batch_size': self.batch_size,
            'total_time': total_time,
            'episodes': self.episode,
            'steps': self.total_steps,
            'avg_score': stats['avg_score'],
            'eps_per_sec': self.episode / total_time
        }


def test_num_envs():
    """Test different numbers of parallel environments"""
    print("\n" + "="*80)
    print("TEST 1: Number of Parallel Environments")
    print("="*80)
    print("Testing: 32, 64, 128, 256 parallel environments")
    print()

    env_counts = [32, 64, 128, 256]
    results = []

    for num_envs in env_counts:
        print(f"Testing {num_envs} environments...", flush=True)
        benchmark = AdditionalBenchmark(
            num_envs=num_envs,
            num_episodes=1000,
            num_train_steps=1
        )
        result = benchmark.train()
        results.append(result)
        print(f"  Time: {result['total_time']:.2f}s, Eps/sec: {result['eps_per_sec']:.2f}")

    print("\nResults:")
    print(f"{'Num Envs':<12} {'Time (s)':<12} {'Eps/sec':<12} {'Score':<12}")
    print("-"*50)
    for r in results:
        print(f"{r['num_envs']:<12} {r['total_time']:<12.2f} {r['eps_per_sec']:<12.2f} {r['avg_score']:<12.2f}")

    return results


def test_training_frequency():
    """Test different training step frequencies"""
    print("\n" + "="*80)
    print("TEST 2: Training Step Frequency")
    print("="*80)
    print("Testing: 1, 4, 16, 64 training steps per env step")
    print()

    train_steps = [1, 4, 16, 64]
    results = []

    for num_steps in train_steps:
        print(f"Testing {num_steps} training steps per env step...", flush=True)
        benchmark = AdditionalBenchmark(
            num_envs=256,
            num_episodes=1000,
            num_train_steps=num_steps
        )
        result = benchmark.train()
        results.append(result)
        print(f"  Time: {result['total_time']:.2f}s, Eps/sec: {result['eps_per_sec']:.2f}, Score: {result['avg_score']:.2f}")

    print("\nResults:")
    print(f"{'Train Steps':<15} {'Time (s)':<12} {'Eps/sec':<12} {'Score':<12}")
    print("-"*55)
    for r in results:
        print(f"{r['num_train_steps']:<15} {r['total_time']:<12.2f} {r['eps_per_sec']:<12.2f} {r['avg_score']:<12.2f}")

    return results


def test_batch_size():
    """Test different batch sizes"""
    print("\n" + "="*80)
    print("TEST 3: Batch Size")
    print("="*80)
    print("Testing: 64, 128, 256, 512 batch sizes")
    print()

    batch_sizes = [64, 128, 256, 512]
    results = []

    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...", flush=True)
        benchmark = AdditionalBenchmark(
            num_envs=256,
            num_episodes=1000,
            num_train_steps=1,
            batch_size=batch_size
        )
        result = benchmark.train()
        results.append(result)
        print(f"  Time: {result['total_time']:.2f}s, Eps/sec: {result['eps_per_sec']:.2f}")

    print("\nResults:")
    print(f"{'Batch Size':<15} {'Time (s)':<12} {'Eps/sec':<12} {'Score':<12}")
    print("-"*55)
    for r in results:
        print(f"{r['batch_size']:<15} {r['total_time']:<12.2f} {r['eps_per_sec']:<12.2f} {r['avg_score']:<12.2f}")

    return results


def main():
    print("="*80)
    print("ADDITIONAL PERFORMANCE BENCHMARKS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_results = {}

    # Test 1: Number of environments
    all_results['num_envs'] = test_num_envs()

    # Test 2: Training frequency
    all_results['training_freq'] = test_training_frequency()

    # Test 3: Batch size
    all_results['batch_size'] = test_batch_size()

    # Save results
    import json
    results_dir = Path('results/benchmarks')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'additional_benchmark_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print(f"Results saved to: {results_file}")
    print("="*80)


if __name__ == '__main__':
    main()
