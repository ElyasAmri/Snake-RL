"""
Classic DQN-based Two-Snake Competitive Training

This is the simple, fast training script from the archive that achieved
excellent results (7.16/10 avg score, 39% win rate) in 4-8 hours of training.

Migrated from: archive/experiments/train_two_snake_dqn.py

Key differences from vectorized version:
- Single environment (not batched)
- Episode-based training (not step-based)
- Simple DQN with 128 hidden neurons
- Target score: 10 food items
- Max steps: 500 per episode
- Proven fast and effective

Expected performance:
- 10,000 episodes: 4-8 hours
- Final avg score: 5-8 food items
- Win rate: ~40-60% (balanced)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import argparse
import time
from datetime import datetime

from core.environment_two_snake_classic import TwoSnakeCompetitiveEnv
from agents.vanilla_dqn import VanillaDQNAgent


class DQNTwoSnakeTrainer:
    """
    DQN-based competitive two-snake trainer.

    Both agents learn independently with their own Q-networks and replay buffers.
    """

    def __init__(
        self,
        grid_size: int = 10,
        target_score: int = 10,
        max_episodes: int = 10000,
        eval_freq: int = 100,
        save_freq: int = 2500,
        output_dir: str = "results/weights/competitive/classic",
        max_time: int = None,  # Maximum training time in seconds
    ):
        """
        Initialize DQN trainer.

        Args:
            grid_size: Grid size
            target_score: Score needed to win
            max_episodes: Maximum training episodes
            eval_freq: Frequency of evaluation logging
            save_freq: Frequency of checkpoint saving
            output_dir: Directory to save checkpoints
            max_time: Maximum training time in seconds
        """
        self.grid_size = grid_size
        self.target_score = target_score
        self.max_episodes = max_episodes
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.max_time = max_time

        # Create directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create environment
        self.env = TwoSnakeCompetitiveEnv(
            grid_size=grid_size,
            target_score=target_score,
            max_steps=500  # Shorter episodes for faster training
        )

        # Create agents
        state_size = self.env.observation_space.shape[0]  # 20 features
        action_size = self.env.action_space.n  # 3 actions

        print(f"State size: {state_size}, Action size: {action_size}", flush=True)

        # Agent 1: Standard DQN (128 hidden neurons)
        self.agent1 = VanillaDQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=128,
            learning_rate=0.0005,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=50000,
            batch_size=64,
            use_target_network=True,
            target_update_freq=1000
        )

        # Agent 2: Standard DQN (128 hidden neurons)
        self.agent2 = VanillaDQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=128,
            learning_rate=0.0005,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=50000,
            batch_size=64,
            use_target_network=True,
            target_update_freq=1000
        )

        # Training statistics
        self.stats = {
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'agent1_total_reward': 0.0,
            'agent2_total_reward': 0.0,
            'agent1_avg_score': 0.0,
            'agent2_avg_score': 0.0,
            'episodes': 0
        }

        # Track training speed
        self.start_time = None
        self.total_steps = 0

    def train(self):
        """Train both agents competitively."""
        print("=" * 70, flush=True)
        print("CLASSIC DQN TWO-SNAKE COMPETITIVE TRAINING", flush=True)
        print("=" * 70, flush=True)
        print(f"Environment: TwoSnakeCompetitiveEnv (classic)", flush=True)
        print(f"Agent1: DQN with 128 hidden neurons", flush=True)
        print(f"Agent2: DQN with 128 hidden neurons", flush=True)
        print(f"Max episodes: {self.max_episodes}", flush=True)
        print(f"Target score: {self.target_score}", flush=True)
        print(f"Max steps per episode: 500", flush=True)
        print(f"State representation: 20 features", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)

        self.start_time = time.time()

        for episode in range(1, self.max_episodes + 1):
            observations, _ = self.env.reset()
            obs1 = observations['agent1']
            obs2 = observations['agent2']

            episode_reward1 = 0.0
            episode_reward2 = 0.0
            done = False
            steps = 0

            while not done:
                steps += 1
                self.total_steps += 1

                # Select actions
                action1 = self.agent1.select_action(obs1, training=True)
                action2 = self.agent2.select_action(obs2, training=True)

                actions = {'agent1': action1, 'agent2': action2}
                next_observations, rewards, terminated, truncated, info = self.env.step(actions)

                next_obs1 = next_observations['agent1']
                next_obs2 = next_observations['agent2']
                reward1 = rewards['agent1']
                reward2 = rewards['agent2']
                done1 = terminated['agent1'] or truncated['agent1']
                done2 = terminated['agent2'] or truncated['agent2']
                done = done1 or done2

                # Train both agents
                self.agent1.train_step(obs1, action1, reward1, next_obs1, done1)
                self.agent2.train_step(obs2, action2, reward2, next_obs2, done2)

                episode_reward1 += reward1
                episode_reward2 += reward2

                obs1 = next_obs1
                obs2 = next_obs2

            # Decay epsilon
            self.agent1.decay_epsilon()
            self.agent2.decay_epsilon()

            # Increment episode count for agents
            self.agent1.episodes += 1
            self.agent2.episodes += 1

            # Update statistics
            self.stats['episodes'] += 1
            self.stats['agent1_total_reward'] += episode_reward1
            self.stats['agent2_total_reward'] += episode_reward2
            self.stats['agent1_avg_score'] += info['score1']
            self.stats['agent2_avg_score'] += info['score2']

            # Determine winner
            winner = info.get('winner', 0)
            if winner == 1:
                self.stats['agent1_wins'] += 1
            elif winner == 2:
                self.stats['agent2_wins'] += 1
            else:
                self.stats['draws'] += 1

            # Evaluation logging
            if episode % self.eval_freq == 0:
                elapsed = time.time() - self.start_time
                episodes_per_hour = (episode / elapsed) * 3600
                avg_reward1 = self.stats['agent1_total_reward'] / self.eval_freq
                avg_reward2 = self.stats['agent2_total_reward'] / self.eval_freq
                avg_score1 = self.stats['agent1_avg_score'] / self.eval_freq
                avg_score2 = self.stats['agent2_avg_score'] / self.eval_freq
                win_rate1 = (self.stats['agent1_wins'] / self.eval_freq) * 100
                win_rate2 = (self.stats['agent2_wins'] / self.eval_freq) * 100
                draw_rate = (self.stats['draws'] / self.eval_freq) * 100

                print(f"Episode {episode}/{self.max_episodes} | Time: {elapsed/60:.1f}min | Speed: {episodes_per_hour:.0f} ep/hr", flush=True)
                print(f"  Agent1: Reward={avg_reward1:.2f}, Score={avg_score1:.2f}, Wins={win_rate1:.1f}%, Epsilon={self.agent1.epsilon:.3f}", flush=True)
                print(f"  Agent2: Reward={avg_reward2:.2f}, Score={avg_score2:.2f}, Wins={win_rate2:.1f}%, Epsilon={self.agent2.epsilon:.3f}", flush=True)
                print(f"  Draws: {draw_rate:.1f}%, Avg Steps: {steps}", flush=True)
                print(flush=True)

                # Reset stats for next window
                self.stats['agent1_total_reward'] = 0.0
                self.stats['agent2_total_reward'] = 0.0
                self.stats['agent1_avg_score'] = 0.0
                self.stats['agent2_avg_score'] = 0.0
                self.stats['agent1_wins'] = 0
                self.stats['agent2_wins'] = 0
                self.stats['draws'] = 0
                self.stats['episodes'] = 0

            # Save checkpoints
            if episode % self.save_freq == 0:
                self.agent1.save(str(self.output_dir / f"agent1_dqn_ep{episode}.pt"))
                self.agent2.save(str(self.output_dir / f"agent2_dqn_ep{episode}.pt"))
                print(f"  >> Saved checkpoints at episode {episode}", flush=True)
                print(flush=True)

            # Early exit if max_time reached
            if self.max_time and (time.time() - self.start_time) >= self.max_time:
                print(f"\n[TIMEOUT] Max time {self.max_time}s reached. Stopping training...", flush=True)
                break

        # Save final models
        self.agent1.save(str(self.output_dir / f"agent1_dqn_final.pt"))
        self.agent2.save(str(self.output_dir / f"agent2_dqn_final.pt"))

        total_time = time.time() - self.start_time

        print("\n" + "=" * 70, flush=True)
        print("TRAINING COMPLETED SUCCESSFULLY!", flush=True)
        print("=" * 70, flush=True)
        print(f"Total time: {total_time/3600:.2f} hours", flush=True)
        print(f"Total episodes: {self.max_episodes}", flush=True)
        print(f"Total steps: {self.total_steps}", flush=True)
        print(f"Average speed: {(self.max_episodes / total_time) * 3600:.0f} episodes/hour", flush=True)
        print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train two-snake competitive with classic DQN')
    parser.add_argument('--grid-size', type=int, default=10, help='Grid size')
    parser.add_argument('--target-score', type=int, default=10, help='Score needed to win')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=2500, help='Checkpoint save frequency')
    parser.add_argument('--max-time', type=int, default=None, help='Maximum training time in seconds')

    args = parser.parse_args()

    trainer = DQNTwoSnakeTrainer(
        grid_size=args.grid_size,
        target_score=args.target_score,
        max_episodes=args.episodes,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        max_time=args.max_time
    )

    trainer.train()


if __name__ == '__main__':
    main()
