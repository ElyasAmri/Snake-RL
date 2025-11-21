"""
Simple Model Evaluation Script

Evaluates all trained DQN models by loading them with DQNTrainer and running evaluation episodes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DQN_MLP, DuelingDQN_MLP, DQN_CNN


def evaluate_model(model_path: Path, num_episodes: int = 100) -> dict:
    """Evaluate a trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = model_path.stem.lower()

    # Determine model config from filename
    if 'enhanced' in filename:
        input_dim = 24
        use_flood_fill = True
        use_selective = False
        use_enhanced = True
    elif 'selective' in filename:
        input_dim = 19
        use_flood_fill = True
        use_selective = True
        use_enhanced = False
    elif 'floodfill' in filename:
        input_dim = 14
        use_flood_fill = True
        use_selective = False
        use_enhanced = False
    else:
        input_dim = 11
        use_flood_fill = False
        use_selective = False
        use_enhanced = False

    # Hidden dims
    if 'large' in filename or '256x256' in filename:
        hidden_dims = (256, 256)
    else:
        hidden_dims = (128, 128)

    # Model type
    is_dueling = 'dueling' in filename
    is_cnn = 'cnn' in filename

    # Skip non-DQN models
    if not any(x in filename for x in ['dqn', 'double', 'dueling', 'per']):
        return {'status': 'Not a DQN model', 'avg_score': 0}

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Create model
        if is_cnn:
            model = DQN_CNN(grid_size=10, input_channels=3, output_dim=3)
        elif is_dueling:
            model = DuelingDQN_MLP(input_dim=input_dim, output_dim=3, hidden_dims=hidden_dims)
        else:
            model = DQN_MLP(input_dim=input_dim, output_dim=3, hidden_dims=hidden_dims)

        # Load weights
        if 'policy_net' in checkpoint:
            model.load_state_dict(checkpoint['policy_net'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        # Create environment with appropriate state representation
        if is_cnn:
            env = VectorizedSnakeEnv(
                num_envs=num_episodes,
                grid_size=10,
                action_space_type='relative',
                state_representation='grid'
            )
        else:
            env = VectorizedSnakeEnv(
                num_envs=num_episodes,
                grid_size=10,
                action_space_type='relative',
                state_representation='feature',
                use_flood_fill=use_flood_fill,
                use_selective_features=use_selective,
                use_enhanced_features=use_enhanced
            )

        # Run evaluation
        obs = env.reset()
        episode_rewards = np.zeros(num_episodes)
        episode_done = np.zeros(num_episodes, dtype=bool)
        scores = np.zeros(num_episodes)
        lengths = np.zeros(num_episodes)

        max_steps = 1000
        for step in range(max_steps):
            # Use environment's built-in feature extraction
            state = obs.to(device)

            # Get actions (greedy)
            with torch.no_grad():
                q_values = model(state)
                actions = q_values.argmax(dim=1)

            # Step environment
            obs, rewards, dones, info = env.step(actions)

            # Track rewards
            episode_rewards += rewards.cpu().numpy() * ~episode_done

            # Check for done episodes
            done_np = dones.cpu().numpy()
            new_done = done_np & ~episode_done

            if new_done.any():
                done_indices = np.where(new_done)[0]
                for idx in done_indices:
                    scores[idx] = info['scores'][idx].item()
                    lengths[idx] = step + 1
                episode_done |= done_np

            if episode_done.all():
                break

        # Handle remaining episodes (reached max steps)
        remaining = ~episode_done
        if remaining.any():
            for idx in np.where(remaining)[0]:
                scores[idx] = env.scores[idx].item()
                lengths[idx] = max_steps

        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'avg_reward': np.mean(episode_rewards),
            'avg_length': np.mean(lengths),
            'status': 'OK'
        }

    except Exception as e:
        return {
            'avg_score': 0,
            'std_score': 0,
            'max_score': 0,
            'min_score': 0,
            'avg_reward': 0,
            'avg_length': 0,
            'status': f'Error: {str(e)[:60]}'
        }


def main():
    weights_dir = Path('results/weights')

    print("=" * 100)
    print("EVALUATING TRAINED DQN MODELS")
    print("=" * 100)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get DQN model files (exclude PPO, REINFORCE, A2C for now)
    model_files = sorted(weights_dir.glob('*.pt'))
    dqn_models = [f for f in model_files if any(x in f.stem.lower() for x in ['dqn', 'double', 'dueling', 'per'])]

    print(f"Found {len(dqn_models)} DQN models to evaluate")
    print()

    results = []

    for model_path in dqn_models:
        print(f"Evaluating {model_path.name}...", end=" ", flush=True)
        result = evaluate_model(model_path, num_episodes=100)
        result['model'] = model_path.name
        results.append(result)

        if result['status'] == 'OK':
            print(f"Avg: {result['avg_score']:.2f}, Max: {result['max_score']:.0f}")
        else:
            print(f"{result['status']}")

    # Print summary table
    print()
    print("=" * 100)
    print("RESULTS SUMMARY (sorted by average score)")
    print("=" * 100)
    print()
    print(f"{'Model':<50} {'Avg':>8} {'Std':>8} {'Max':>6} {'Min':>6} {'Reward':>10} {'Length':>8}")
    print("-" * 100)

    # Sort by average score
    results_ok = [r for r in results if r['status'] == 'OK']
    results_err = [r for r in results if r['status'] != 'OK']
    results_sorted = sorted(results_ok, key=lambda x: x['avg_score'], reverse=True)

    for r in results_sorted:
        print(f"{r['model']:<50} {r['avg_score']:>8.2f} {r['std_score']:>8.2f} {r['max_score']:>6.0f} {r['min_score']:>6.0f} {r['avg_reward']:>10.2f} {r['avg_length']:>8.1f}")

    if results_err:
        print()
        print("Models with errors:")
        for r in results_err:
            print(f"  {r['model']}: {r['status']}")

    print("-" * 100)

    # Print top 5
    if results_sorted:
        print()
        print("TOP 5 MODELS:")
        for i, r in enumerate(results_sorted[:5], 1):
            print(f"  {i}. {r['model']}")
            print(f"     Avg Score: {r['avg_score']:.2f} +/- {r['std_score']:.2f}")
            print(f"     Max Score: {r['max_score']:.0f}")
            print()


if __name__ == '__main__':
    main()
