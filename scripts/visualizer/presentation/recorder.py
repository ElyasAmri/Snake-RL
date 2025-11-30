"""
Presentation Recorder

Main orchestrator for recording presentation videos.
Coordinates game rendering, feature visualization, and video encoding.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import glob
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import cv2
import torch

from core.environment_vectorized import VectorizedSnakeEnv
from core.environment import SnakeEnv
from scripts.baselines.random_agent import RandomAgent
from scripts.baselines.shortest_path import ShortestPathAgent
from scripts.baselines.hamiltonian import HamiltonianAgent

# Handle both relative and absolute imports
try:
    from .renderers.game_renderer import GameRenderer
    from .renderers.feature_visualizer import FeatureVisualizer
    from .renderers.frame_composer import FrameComposer
    from .agents.trained_agent import TrainedAgent, load_agent_from_path, FEATURE_DIMS
    from .presets import PRESETS, FALLBACK_WEIGHTS, get_preset, list_presets
except ImportError:
    from scripts.visualizer.presentation.renderers.game_renderer import GameRenderer
    from scripts.visualizer.presentation.renderers.feature_visualizer import FeatureVisualizer
    from scripts.visualizer.presentation.renderers.frame_composer import FrameComposer
    from scripts.visualizer.presentation.agents.trained_agent import TrainedAgent, load_agent_from_path, FEATURE_DIMS
    from scripts.visualizer.presentation.presets import PRESETS, FALLBACK_WEIGHTS, get_preset, list_presets


class PresentationRecorder:
    """
    Records high-quality presentation videos.

    Produces 1080p MP4 videos with:
    - Game visualization
    - Feature heatmap overlays
    - Feature value panels
    - Q-value charts
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 10,
        grid_size: int = 10
    ):
        """
        Initialize presentation recorder.

        Args:
            resolution: Output video resolution (width, height)
            fps: Frames per second
            grid_size: Snake game grid size
        """
        self.width, self.height = resolution
        self.fps = fps
        self.grid_size = grid_size
        self.game_area_size = min(self.height, 1080)  # Square game area

        # Initialize renderers
        self.game_renderer = GameRenderer(
            grid_size=grid_size,
            game_area_size=self.game_area_size
        )
        self.feature_visualizer = FeatureVisualizer(
            grid_size=grid_size,
            game_area_size=self.game_area_size
        )
        self.frame_composer = FrameComposer(
            width=self.width,
            height=self.height,
            game_area_size=self.game_area_size
        )

        # Device for trained agents
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def record_video(
        self,
        output_path: str,
        agent_type: str = 'random',
        baseline: str = 'random',
        weights_path: Optional[str] = None,
        network: str = 'dqn',
        feature_mode: str = 'basic',
        duration: int = 30,
        show_features: bool = False,
        show_q_values: bool = False,
        heatmap: Optional[str] = None,
        show_path: bool = False,
        seed: int = 67
    ) -> Dict[str, Any]:
        """
        Record a single video.

        Args:
            output_path: Path to save MP4 file
            agent_type: 'baseline' or 'trained'
            baseline: Baseline agent type (if agent_type='baseline')
            weights_path: Path to model weights (if agent_type='trained')
            network: Network architecture type
            feature_mode: Feature mode for trained agent
            duration: Video duration in seconds
            show_features: Whether to show feature panel
            show_q_values: Whether to show Q-value chart
            heatmap: Heatmap type ('danger', 'flood_fill', 'combined', or None)
            show_path: Whether to show A* path (for greedy baseline)
            seed: Random seed for environment

        Returns:
            Recording stats dict
        """
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Create agent
        if agent_type == 'trained':
            agent = self._create_trained_agent(weights_path, network, feature_mode)
            use_features = True
        else:
            agent = self._create_baseline_agent(baseline)
            use_features = False

        # Create environment
        # Use VectorizedSnakeEnv with num_envs=1 for trained agents (has feature flags)
        # Use regular SnakeEnv for baselines (simpler interface)
        if agent_type == 'trained':
            env = VectorizedSnakeEnv(
                num_envs=1,
                grid_size=self.grid_size,
                action_space_type='relative',
                state_representation='feature',
                max_steps=10000,
                use_flood_fill=(feature_mode in ('flood-fill', 'floodfill', 'selective', 'enhanced')),
                use_selective_features=(feature_mode == 'selective'),
                use_enhanced_features=(feature_mode == 'enhanced'),
                device='cpu'  # Use CPU for single-env visualization
            )
            is_vectorized = True
        else:
            env = SnakeEnv(
                grid_size=self.grid_size,
                action_space_type='absolute',
                state_representation='feature',
                max_steps=10000,
                reward_distance=False
            )
            is_vectorized = False

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Recording loop
        total_frames = duration * self.fps
        frame_count = 0
        episode = 1
        total_score = 0
        steps = 0

        # Reset environment
        if is_vectorized:
            obs = env.reset(seed=seed)
            # Convert tensor to numpy for agent
            obs_np = obs[0].cpu().numpy()
        else:
            obs, info = env.reset(seed=seed)
            obs_np = obs

        print(f"Recording: {output_path}")
        print(f"  Duration: {duration}s, FPS: {self.fps}, Resolution: {self.width}x{self.height}")

        while frame_count < total_frames:
            # Get action
            if agent_type == 'trained':
                action = agent.get_action(obs_np)
                q_values = agent.get_q_values()
            else:
                action = agent.get_action(env)
                q_values = np.zeros(3)

            # Step environment
            if is_vectorized:
                action_tensor = torch.tensor([action], device='cpu')
                obs, reward, done, info = env.step(action_tensor)
                obs_np = obs[0].cpu().numpy()
                done = done[0].item()
                current_score = int(info['scores'][0].item())
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                obs_np = obs
                done = terminated or truncated
                current_score = env.score
            steps += 1

            # Render game - extract snake and food positions
            if is_vectorized:
                snake_length = int(env.snake_lengths[0].item())
                snake_list = [(int(env.snakes[0, i, 0].item()), int(env.snakes[0, i, 1].item()))
                             for i in range(snake_length)]
                food_pos = (int(env.foods[0, 0].item()), int(env.foods[0, 1].item()))
                direction = int(env.directions[0].item())
            else:
                snake_list = [(int(x), int(y)) for x, y in env.snake]
                food_pos = (int(env.food[0]), int(env.food[1])) if env.food is not None else None
                direction = env.direction
                current_score = env.score

            if show_path and hasattr(agent, 'get_path'):
                path = agent.get_path(env)
                game_frame = self.game_renderer.render_with_path(
                    snake_list, food_pos, direction, current_score, episode, path, steps
                )
            else:
                game_frame = self.game_renderer.render(
                    snake_list, food_pos, direction, current_score, episode, steps
                )

            # Create heatmap overlay
            heatmap_overlay = None
            if heatmap and use_features:
                head_pos = snake_list[0]
                if heatmap == 'danger':
                    heatmap_overlay = self.feature_visualizer.create_danger_overlay(
                        obs_np, head_pos, direction
                    )
                elif heatmap == 'flood_fill':
                    heatmap_overlay = self.feature_visualizer.create_flood_fill_overlay(
                        obs_np, head_pos, direction
                    )
                elif heatmap == 'combined':
                    heatmap_overlay = self.feature_visualizer.create_combined_overlay(
                        obs_np, head_pos, direction, feature_mode
                    )

            # Create feature panel
            feature_panel = None
            if show_features and use_features:
                feature_panel = self.feature_visualizer.create_feature_panel(
                    obs_np, feature_mode, width=840, height=720
                )

            # Create Q-value chart (normalized bars with actual values as labels)
            q_chart = None
            if show_q_values and use_features:
                selected_action = agent.last_action if hasattr(agent, 'last_action') else 0
                q_chart = self.feature_visualizer.create_q_value_chart(
                    q_values, selected_action, width=840, height=360
                )

            # Compose final frame
            if show_features or show_q_values:
                frame = self.frame_composer.compose(
                    game_frame, heatmap_overlay, feature_panel, q_chart
                )
            else:
                frame = self.frame_composer.compose_simple(game_frame, heatmap_overlay)

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            frame_count += 1

            # Handle episode end
            if done:
                total_score += current_score
                print(f"  Episode {episode}: Score {current_score}, Steps {steps}")
                episode += 1
                steps = 0
                if is_vectorized:
                    obs = env.reset()
                    obs_np = obs[0].cpu().numpy()
                else:
                    obs, info = env.reset()
                    obs_np = obs

            # Progress update
            if frame_count % (self.fps * 10) == 0:
                print(f"  Progress: {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.0f}%)")

        out.release()

        # Stats
        stats = {
            'output_path': output_path,
            'duration': frame_count / self.fps,
            'frames': frame_count,
            'episodes': episode,
            'avg_score': total_score / max(1, episode - 1)
        }

        print(f"  Saved: {output_path}")
        print(f"  Episodes: {episode-1}, Avg Score: {stats['avg_score']:.1f}")

        return stats

    def _create_baseline_agent(self, baseline: str):
        """Create baseline agent."""
        if baseline == 'random':
            return RandomAgent(action_space_type='absolute')
        elif baseline == 'greedy':
            return ShortestPathAgent(action_space_type='absolute')
        elif baseline == 'hamiltonian':
            return HamiltonianAgent(action_space_type='absolute', grid_size=self.grid_size)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

    def _create_trained_agent(
        self,
        weights_path: Optional[str],
        network: str,
        feature_mode: str
    ) -> TrainedAgent:
        """Create trained agent from weights."""
        if weights_path is None:
            raise ValueError("weights_path required for trained agent")

        # Resolve glob pattern
        if '*' in weights_path:
            matches = glob.glob(weights_path)
            if not matches:
                # Try fallback patterns
                fallbacks = FALLBACK_WEIGHTS.get(feature_mode, [])
                for pattern in fallbacks:
                    matches = glob.glob(pattern)
                    if matches:
                        break

            if not matches:
                raise FileNotFoundError(f"No weights found matching: {weights_path}")

            weights_path = sorted(matches)[-1]
            print(f"  Using weights: {weights_path}")

        return TrainedAgent(
            weights_path=weights_path,
            network_type=network,
            feature_mode=feature_mode,
            device=self.device
        )

    def record_preset(self, preset_name: str, output_dir: str) -> Dict[str, Any]:
        """
        Record video using a preset configuration.

        Args:
            preset_name: Name of preset (e.g., '01_random', 'flood_fill')
            output_dir: Directory to save video

        Returns:
            Recording stats
        """
        preset = get_preset(preset_name)

        output_path = str(Path(output_dir) / f"{preset_name}.mp4")

        return self.record_video(
            output_path=output_path,
            agent_type=preset.get('agent_type', 'baseline'),
            baseline=preset.get('baseline', 'random'),
            weights_path=preset.get('weights_pattern'),
            network=preset.get('network', 'dqn'),
            feature_mode=preset.get('feature_mode', 'basic'),
            duration=preset.get('duration', 30),
            show_features=preset.get('show_features', False),
            show_q_values=preset.get('show_q_values', False),
            heatmap=preset.get('heatmap'),
            show_path=preset.get('show_path', False)
        )

    def record_all_presets(self, output_dir: str) -> List[Dict[str, Any]]:
        """
        Record all preset videos.

        Args:
            output_dir: Directory to save videos

        Returns:
            List of recording stats for each video
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stats = []
        for preset_name in PRESETS:
            try:
                result = self.record_preset(preset_name, output_dir)
                stats.append(result)
            except Exception as e:
                print(f"  Error recording {preset_name}: {e}")
                stats.append({'preset': preset_name, 'error': str(e)})

        return stats

    def cleanup(self):
        """Clean up resources."""
        self.game_renderer.cleanup()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Record Snake RL presentation videos')

    parser.add_argument('--preset', type=str, help='Record specific preset')
    parser.add_argument('--all', action='store_true', help='Record all presets')
    parser.add_argument('--list', action='store_true', help='List available presets')
    parser.add_argument('--output', type=str, default='presentation/videos/',
                       help='Output directory or file path')

    # Custom recording options
    parser.add_argument('--agent', type=str, choices=['baseline', 'trained'],
                       help='Agent type')
    parser.add_argument('--baseline', type=str, choices=['random', 'greedy', 'hamiltonian'],
                       help='Baseline agent type')
    parser.add_argument('--weights', type=str, help='Path to trained weights')
    parser.add_argument('--network', type=str, default='dqn',
                       choices=['dqn', 'double_dqn', 'dueling', 'noisy', 'per', 'ppo'],
                       help='Network type')
    parser.add_argument('--feature-mode', type=str, default='basic',
                       choices=['basic', 'flood-fill', 'selective', 'enhanced'],
                       help='Feature mode')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    parser.add_argument('--show-features', action='store_true', help='Show feature panel')
    parser.add_argument('--show-q-values', action='store_true', help='Show Q-value chart')
    parser.add_argument('--heatmap', type=str, choices=['danger', 'flood_fill', 'combined'],
                       help='Heatmap type')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--resolution', type=str, default='1080p',
                       choices=['720p', '1080p', '4k'], help='Output resolution')

    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_presets()
        return

    # Parse resolution
    resolutions = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '4k': (3840, 2160)
    }
    resolution = resolutions[args.resolution]

    # Create recorder
    recorder = PresentationRecorder(resolution=resolution, fps=args.fps)

    try:
        if args.all:
            # Record all presets
            recorder.record_all_presets(args.output)

        elif args.preset:
            # Record specific preset
            recorder.record_preset(args.preset, args.output)

        elif args.agent:
            # Custom recording
            if args.agent == 'baseline':
                output_path = args.output if args.output.endswith('.mp4') else \
                             str(Path(args.output) / f"{args.baseline}_custom.mp4")
            else:
                output_path = args.output if args.output.endswith('.mp4') else \
                             str(Path(args.output) / f"{args.feature_mode}_custom.mp4")

            recorder.record_video(
                output_path=output_path,
                agent_type=args.agent,
                baseline=args.baseline or 'random',
                weights_path=args.weights,
                network=args.network,
                feature_mode=args.feature_mode,
                duration=args.duration,
                show_features=args.show_features,
                show_q_values=args.show_q_values,
                heatmap=args.heatmap
            )
        else:
            parser.print_help()

    finally:
        recorder.cleanup()

    print("\nDone!")


if __name__ == '__main__':
    main()
