"""
Snake Game Visualizer

Visualize Snake RL in 3 modes:
1. Random: Watch random agent play
2. Trained: Watch saved model play
3. Training: Watch agent train in real-time

Usage:
    python scripts/visualizer/visualize_snake.py --mode random --fps 10
    python scripts/visualizer/visualize_snake.py --mode trained --weights results/weights/dueling_dqn_mlp.pt --network dueling
    python scripts/visualizer/visualize_snake.py --mode training --algorithm dueling --episodes 1000 --fps 30 --num-envs 256
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pygame
import torch
import argparse

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DQN_MLP, DuelingDQN_MLP, NoisyDQN_MLP
from scripts.training.train_dqn import DQNTrainer

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)


class SnakeVisualizer:
    """Pygame visualizer for Snake RL"""

    def __init__(self, grid_size=10, cell_size=50, fps=10, device='cuda'):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Pygame
        pygame.init()
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size + 100
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake RL Visualizer')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Environment (single snake for visualization)
        self.env = VectorizedSnakeEnv(
            num_envs=1,
            grid_size=grid_size,
            action_space_type='relative',
            state_representation='feature',
            max_steps=1000,
            device=self.device
        )

        # Stats
        self.episode = 0
        self.total_score = 0
        self.current_score = 0
        self.current_steps = 0

    def draw_grid(self):
        """Draw grid lines"""
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.width), 1)
        for y in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)

    def draw_snake(self, snake, length):
        """Draw snake"""
        for i in range(length):
            x = int(snake[i, 0].item()) * self.cell_size
            y = int(snake[i, 1].item()) * self.cell_size

            if i == 0:
                # Head
                pygame.draw.rect(self.screen, GREEN, (x, y, self.cell_size, self.cell_size))
                # Eyes
                eye_size = self.cell_size // 6
                pygame.draw.circle(self.screen, BLACK,
                                   (x + self.cell_size//3, y + self.cell_size//3), eye_size)
                pygame.draw.circle(self.screen, BLACK,
                                   (x + 2*self.cell_size//3, y + self.cell_size//3), eye_size)
            else:
                # Body
                pygame.draw.rect(self.screen, DARK_GREEN,
                                 (x + 2, y + 2, self.cell_size - 4, self.cell_size - 4))

    def draw_food(self, food):
        """Draw food"""
        x = int(food[0].item()) * self.cell_size
        y = int(food[1].item()) * self.cell_size
        pygame.draw.circle(self.screen, RED,
                           (x + self.cell_size//2, y + self.cell_size//2),
                           self.cell_size//3)

    def draw_stats(self, mode, info_lines):
        """Draw stats panel"""
        stats_y = self.width
        pygame.draw.rect(self.screen, BLACK, (0, stats_y, self.width, 100))

        # Title
        title = self.font_large.render(f"Mode: {mode.upper()}", True, YELLOW)
        self.screen.blit(title, (10, stats_y + 5))

        # Info lines
        for i, line in enumerate(info_lines):
            surface = self.font.render(line, True, WHITE)
            self.screen.blit(surface, (10, stats_y + 40 + i * 20))

    def run_random(self, num_episodes):
        """Mode 1: Random agent"""
        print(f"Random mode: {num_episodes} episodes")

        state = self.env.reset()
        running = True

        while running and self.episode < num_episodes:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # Random action
            action = torch.randint(0, 3, (1,), device=self.device)
            next_state, reward, done, info = self.env.step(action)

            self.current_score = int(info['scores'][0].item())
            self.current_steps += 1

            # Draw
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_snake(self.env.snakes[0], self.env.snake_lengths[0])
            self.draw_food(self.env.foods[0])

            action_names = ['STRAIGHT', 'LEFT', 'RIGHT']
            avg_score = self.total_score / max(1, self.episode)
            self.draw_stats("Random", [
                f"Episode: {self.episode}/{num_episodes}  Score: {self.current_score}  Steps: {self.current_steps}",
                f"Avg Score: {avg_score:.1f}  Action: {action_names[action[0].item()]}",
            ])

            pygame.display.flip()
            self.clock.tick(self.fps)

            if done[0]:
                self.episode += 1
                self.total_score += self.current_score
                print(f"Episode {self.episode}: Score {self.current_score}, Steps {self.current_steps}")
                self.current_steps = 0
                state = self.env.reset()
            else:
                state = next_state

        pygame.quit()
        print(f"\nFinished {self.episode} episodes. Avg Score: {self.total_score / max(1, self.episode):.2f}")

    def run_trained(self, weights_path, network_type, num_episodes):
        """Mode 2: Trained model"""
        print(f"Trained mode: Loading {weights_path}")

        # Load network
        if network_type == 'dueling':
            network = DuelingDQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128))
        elif network_type == 'noisy':
            network = NoisyDQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128))
        else:
            network = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128))

        checkpoint = torch.load(weights_path, map_location=self.device)
        if 'policy_net' in checkpoint:
            network.load_state_dict(checkpoint['policy_net'])
        else:
            network.load_state_dict(checkpoint)

        network.to(self.device)
        network.eval()

        state = self.env.reset()
        running = True

        with torch.no_grad():
            while running and self.episode < num_episodes:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                       (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False

                # Get action
                q_values = network(state)
                action = q_values.argmax(dim=1)
                next_state, reward, done, info = self.env.step(action)

                self.current_score = int(info['scores'][0].item())
                self.current_steps += 1

                # Draw
                self.screen.fill(BLACK)
                self.draw_grid()
                self.draw_snake(self.env.snakes[0], self.env.snake_lengths[0])
                self.draw_food(self.env.foods[0])

                action_names = ['STRAIGHT', 'LEFT', 'RIGHT']
                q_str = f"Q: [{q_values[0,0]:.1f}, {q_values[0,1]:.1f}, {q_values[0,2]:.1f}]"
                avg_score = self.total_score / max(1, self.episode)
                self.draw_stats("Trained", [
                    f"Episode: {self.episode}/{num_episodes}  Score: {self.current_score}  Steps: {self.current_steps}",
                    f"Avg Score: {avg_score:.1f}  Action: {action_names[action[0].item()]}  {q_str}",
                ])

                pygame.display.flip()
                self.clock.tick(self.fps)

                if done[0]:
                    self.episode += 1
                    self.total_score += self.current_score
                    print(f"Episode {self.episode}: Score {self.current_score}, Steps {self.current_steps}")
                    self.current_steps = 0
                    state = self.env.reset()
                else:
                    state = next_state

        pygame.quit()
        print(f"\nFinished {self.episode} episodes. Avg Score: {self.total_score / max(1, self.episode):.2f}")

    def run_training(self, algorithm, num_episodes, num_envs=4):
        """Mode 3: Watch training in real-time"""
        print(f"Training mode: {algorithm.upper()} for {num_episodes} episodes with {num_envs} parallel environments")

        # Create trainer
        use_dueling = algorithm == 'dueling'
        use_noisy = algorithm == 'noisy'

        trainer = DQNTrainer(
            num_envs=num_envs,
            grid_size=self.grid_size,
            action_space_type='relative',
            state_representation='feature',
            use_dueling=use_dueling,
            use_noisy=use_noisy,
            num_episodes=num_episodes,
            max_steps=500,
            batch_size=32,
            buffer_size=5000,
            learning_rate=0.001,
            gamma=0.99,
            target_update_freq=500,
            min_buffer_size=500,
            seed=42,
            device=self.device
        )

        states = trainer.env.reset()
        running = True
        episode_rewards = torch.zeros(trainer.num_envs, device=self.device)
        episode_lengths = torch.zeros(trainer.num_envs, dtype=torch.long, device=self.device)

        while running and self.episode < num_episodes:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # Training step
            actions = trainer.select_actions(states)
            next_states, rewards, dones, info = trainer.env.step(actions)

            episode_rewards += rewards
            episode_lengths += 1

            # Store transitions
            for i in range(min(2, trainer.num_envs)):
                trainer.replay_buffer.push(
                    states[i].cpu().numpy(),
                    actions[i].item(),
                    rewards[i].item(),
                    next_states[i].cpu().numpy(),
                    dones[i].item()
                )

            # Train
            loss = None
            if trainer.replay_buffer.is_ready(trainer.min_buffer_size):
                loss = trainer.train_step()
                if trainer.total_steps % trainer.target_update_freq == 0:
                    trainer.target_net.load_state_dict(trainer.policy_net.state_dict())

            # Handle done
            if dones[0]:
                self.episode += 1
                self.total_score += int(info['scores'][0].item())
                self.current_score = int(info['scores'][0].item())
                self.current_steps = int(episode_lengths[0].item())
                episode_rewards[0] = 0
                episode_lengths[0] = 0
                print(f"Episode {self.episode}: Score {self.current_score}, Steps {self.current_steps}")

            # Draw (show env[0])
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_snake(trainer.env.snakes[0], trainer.env.snake_lengths[0])
            self.draw_food(trainer.env.foods[0])

            epsilon = trainer.epsilon_scheduler.get_epsilon() if not use_noisy else 0.0
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            avg_score = self.total_score / max(1, self.episode)
            self.draw_stats("Training", [
                f"Episode: {self.episode}/{num_episodes}  Score: {self.current_score}  Steps: {self.current_steps}",
                f"Avg Score: {avg_score:.1f}  Loss: {loss_str}  Epsilon: {epsilon:.3f}  Buffer: {len(trainer.replay_buffer)}",
            ])

            pygame.display.flip()
            self.clock.tick(self.fps)

            states = next_states
            trainer.total_steps += 1

        pygame.quit()
        print(f"\nFinished {self.episode} episodes. Avg Score: {self.total_score / max(1, self.episode):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Snake RL Visualizer')
    parser.add_argument('--mode', required=True, choices=['random', 'trained', 'training'])
    parser.add_argument('--weights', type=str, help='Path to weights (trained mode)')
    parser.add_argument('--network', type=str, default='dueling',
                        choices=['dqn', 'dueling', 'noisy'])
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'dueling', 'noisy'],
                        help='Algorithm for training mode')
    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--cell-size', type=int, default=50)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments for training mode')

    args = parser.parse_args()

    if args.mode == 'trained' and not args.weights:
        parser.error("--weights required for trained mode")

    viz = SnakeVisualizer(
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        fps=args.fps
    )

    if args.mode == 'random':
        viz.run_random(args.episodes)
    elif args.mode == 'trained':
        viz.run_trained(args.weights, args.network, args.episodes)
    elif args.mode == 'training':
        viz.run_training(args.algorithm, args.episodes, args.num_envs)


if __name__ == '__main__':
    main()
