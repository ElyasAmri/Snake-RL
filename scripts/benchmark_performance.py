"""
Performance Benchmark Script

Tests different optimizations to identify bottlenecks in DQN training.
Runs 2000 episodes with different configurations and measures time.
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


class DQNBenchmark:
    """Stripped down DQN trainer for benchmarking"""

    def __init__(
        self,
        num_envs: int = 256,
        num_episodes: int = 2000,
        optimization: str = "baseline",
        device: torch.device = None
    ):
        """Initialize benchmark trainer

        Args:
            num_envs: Number of parallel environments
            num_episodes: Number of episodes to train
            optimization: Which optimization to test
                - "baseline": Current implementation
                - "reduce_training": Train 1 step instead of 64
                - "vectorize_push": Batch GPU->CPU transfers
                - "both": Apply both optimizations
            device: PyTorch device
        """
        set_seed(67)
        self.device = device if device else get_device()
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.optimization = optimization

        # Training config
        self.gamma = 0.99
        self.batch_size = 256
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
        self.replay_buffer = ReplayBuffer(self.buffer_size, seed=67)

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

        print("="*80)
        print(f"PERFORMANCE BENCHMARK - {self.optimization.upper()}")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Num envs: {self.num_envs}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Optimization: {self.optimization}")
        print()

        # Timing
        start_time = time.time()

        # Initialize
        states = self.env.reset(seed=67)
        episode_rewards = torch.zeros(self.num_envs, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Track timings for different operations
        select_time = 0
        step_time = 0
        push_time = 0
        train_time = 0

        while self.episode < self.num_episodes:
            # Select actions
            t0 = time.time()
            actions = self.select_actions(states)
            select_time += time.time() - t0

            # Step environment
            t0 = time.time()
            next_states, rewards, dones, info = self.env.step(actions)
            step_time += time.time() - t0

            # Accumulate episode stats
            episode_rewards += rewards
            episode_lengths += 1

            # Store transitions - OPTIMIZATION POINT 1
            t0 = time.time()
            if self.optimization in ["baseline", "reduce_training"]:
                # BASELINE: Loop through each environment (SLOW)
                for i in range(self.num_envs):
                    self.replay_buffer.push(
                        states[i].cpu().numpy(),
                        actions[i].item(),
                        rewards[i].item(),
                        next_states[i].cpu().numpy(),
                        dones[i].item()
                    )
            else:  # vectorize_push or both
                # OPTIMIZED: Batch GPU->CPU transfer
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
            push_time += time.time() - t0

            # Train - OPTIMIZATION POINT 2
            t0 = time.time()
            if self.replay_buffer.is_ready(self.min_buffer_size):
                if self.optimization in ["baseline", "vectorize_push"]:
                    # BASELINE: 64 training steps per env step
                    num_train_steps = max(1, self.num_envs // 4)
                else:  # reduce_training or both
                    # OPTIMIZED: 1 training step per env step
                    num_train_steps = 1

                for _ in range(num_train_steps):
                    loss = self.train_step()
                    if loss is not None:
                        self.metrics.add_loss(loss)
            train_time += time.time() - t0

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

        # Final timing
        total_time = time.time() - start_time

        # Print results
        stats = self.metrics.get_recent_stats()

        print()
        print("="*80)
        print("RESULTS")
        print("="*80)
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Episodes: {self.episode}")
        print(f"Steps: {self.total_steps}")
        print(f"Episodes/second: {self.episode / total_time:.2f}")
        print()
        print(f"Avg Score: {stats['avg_score']:.2f}")
        print(f"Avg Reward: {stats['avg_reward']:.2f}")
        print(f"Avg Length: {stats['avg_length']:.2f}")
        print()
        print("Time Breakdown:")
        print(f"  Select actions: {select_time:.2f}s ({select_time/total_time*100:.1f}%)")
        print(f"  Env step:       {step_time:.2f}s ({step_time/total_time*100:.1f}%)")
        print(f"  Replay push:    {push_time:.2f}s ({push_time/total_time*100:.1f}%)")
        print(f"  Training:       {train_time:.2f}s ({train_time/total_time*100:.1f}%)")
        print(f"  Other:          {total_time - select_time - step_time - push_time - train_time:.2f}s")
        print("="*80)

        return {
            'optimization': self.optimization,
            'total_time': total_time,
            'episodes': self.episode,
            'steps': self.total_steps,
            'avg_score': stats['avg_score'],
            'select_time': select_time,
            'step_time': step_time,
            'push_time': push_time,
            'train_time': train_time
        }


def run_benchmark_suite():
    """Run all benchmark tests"""

    print("="*80)
    print("PERFORMANCE BENCHMARK SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = []

    # Test 1: Baseline
    print("\n[1/4] Testing BASELINE configuration...")
    benchmark = DQNBenchmark(num_episodes=2000, optimization="baseline")
    result = benchmark.train()
    results.append(result)

    # Test 2: Reduce training frequency
    print("\n[2/4] Testing REDUCED TRAINING configuration...")
    benchmark = DQNBenchmark(num_episodes=2000, optimization="reduce_training")
    result = benchmark.train()
    results.append(result)

    # Test 3: Vectorize push
    print("\n[3/4] Testing VECTORIZED PUSH configuration...")
    benchmark = DQNBenchmark(num_episodes=2000, optimization="vectorize_push")
    result = benchmark.train()
    results.append(result)

    # Test 4: Both optimizations
    print("\n[4/4] Testing BOTH OPTIMIZATIONS configuration...")
    benchmark = DQNBenchmark(num_episodes=2000, optimization="both")
    result = benchmark.train()
    results.append(result)

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Config':<20} {'Time (s)':<12} {'Time (m)':<12} {'Speedup':<10} {'Score':<10}")
    print("-"*80)

    baseline_time = results[0]['total_time']

    for r in results:
        speedup = baseline_time / r['total_time']
        print(f"{r['optimization']:<20} {r['total_time']:<12.2f} {r['total_time']/60:<12.2f} {speedup:<10.2f}x {r['avg_score']:<10.2f}")

    print("="*80)
    print()

    # Time breakdown comparison
    print("TIME BREAKDOWN COMPARISON:")
    print(f"{'Config':<20} {'Select %':<12} {'Step %':<12} {'Push %':<12} {'Train %':<12}")
    print("-"*80)

    for r in results:
        total = r['total_time']
        print(f"{r['optimization']:<20} "
              f"{r['select_time']/total*100:<12.1f} "
              f"{r['step_time']/total*100:<12.1f} "
              f"{r['push_time']/total*100:<12.1f} "
              f"{r['train_time']/total*100:<12.1f}")

    print("="*80)

    # Save results
    from pathlib import Path
    import json

    results_dir = Path('results/benchmarks')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'benchmark_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark DQN performance optimizations')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'baseline', 'reduce_training', 'vectorize_push', 'both'],
                        help='Which test to run')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes to train')

    args = parser.parse_args()

    if args.test == 'all':
        run_benchmark_suite()
    else:
        benchmark = DQNBenchmark(num_episodes=args.episodes, optimization=args.test)
        benchmark.train()
