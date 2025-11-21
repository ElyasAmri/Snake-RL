"""
Evaluate All Trained Models

Loads and evaluates all trained models from results/weights/
Reports average score, reward, and survival length for each model.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DQN_MLP, DQN_CNN, DuelingDQN_MLP
from core.state_representations import FeatureEncoder


def evaluate_dqn_model(model_path: Path, num_episodes: int = 100) -> dict:
    """Evaluate a DQN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine model type and input size from filename
    filename = model_path.stem.lower()

    # Determine input dimension based on model name
    if 'enhanced' in filename:
        input_dim = 24
        use_flood_fill = True
        use_enhanced = True
    elif 'selective' in filename:
        input_dim = 19
        use_flood_fill = True
        use_enhanced = False
    elif 'floodfill' in filename:
        input_dim = 14
        use_flood_fill = True
        use_enhanced = False
    else:
        input_dim = 11
        use_flood_fill = False
        use_enhanced = False

    # Determine hidden layer size
    if 'large' in filename or '256x256' in filename:
        hidden_dims = (256, 256)
    else:
        hidden_dims = (128, 128)

    # Check for CNN models
    is_cnn = 'cnn' in filename

    # Check for special architectures
    is_dueling = 'dueling' in filename

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Create appropriate model
        if is_cnn:
            model = DQN_CNN(grid_size=10, input_channels=3, output_dim=3)
            model.load_state_dict(checkpoint.get('policy_net', checkpoint.get('model', checkpoint)))
        elif is_dueling:
            model = DuelingDQN_MLP(input_dim=input_dim, output_dim=3, hidden_dims=hidden_dims)
            model.load_state_dict(checkpoint.get('policy_net', checkpoint.get('model', checkpoint)))
        else:
            model = DQN_MLP(input_dim=input_dim, output_dim=3, hidden_dims=hidden_dims)
            model.load_state_dict(checkpoint.get('policy_net', checkpoint.get('model', checkpoint)))

        model = model.to(device)
        model.eval()

        # Create environment
        env = VectorizedSnakeEnv(num_envs=num_episodes, grid_size=10)
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=use_flood_fill, use_enhanced_features=use_enhanced)

        # Run evaluation
        scores = []
        rewards = []
        lengths = []

        obs = env.reset()
        episode_rewards = np.zeros(num_episodes)
        episode_done = np.zeros(num_episodes, dtype=bool)

        max_steps = 1000
        for step in range(max_steps):
            # Encode observations
            if is_cnn:
                # For CNN, use grid representation
                state = torch.tensor(obs, dtype=torch.float32, device=device)
            else:
                # For MLP, encode features
                features = []
                for i in range(num_episodes):
                    if not episode_done[i]:
                        snake = [tuple(env.snakes[i, j].cpu().numpy()) for j in range(env.lengths[i])]
                        food = tuple(env.foods[i].cpu().numpy())
                        direction = env.directions[i].item()
                        feat = encoder.encode(snake, food, direction)
                        features.append(feat)
                    else:
                        features.append(np.zeros(input_dim, dtype=np.float32))
                state = torch.tensor(np.array(features), dtype=torch.float32, device=device)

            # Get actions
            with torch.no_grad():
                q_values = model(state)
                actions = q_values.argmax(dim=1)

            # Step environment
            obs, reward, done, info = env.step(actions)

            # Track rewards for non-done episodes
            episode_rewards += reward.cpu().numpy() * ~episode_done

            # Check for newly done episodes
            new_done = done.cpu().numpy() & ~episode_done
            if new_done.any():
                done_indices = np.where(new_done)[0]
                for idx in done_indices:
                    scores.append(env.foods_eaten[idx].item())
                    rewards.append(episode_rewards[idx])
                    lengths.append(step + 1)
                episode_done |= done.cpu().numpy()

            if episode_done.all():
                break

        # Handle any remaining episodes
        remaining = ~episode_done
        if remaining.any():
            for idx in np.where(remaining)[0]:
                scores.append(env.foods_eaten[idx].item())
                rewards.append(episode_rewards[idx])
                lengths.append(max_steps)

        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'avg_reward': np.mean(rewards),
            'avg_length': np.mean(lengths),
            'status': 'OK'
        }

    except Exception as e:
        return {
            'avg_score': 0,
            'std_score': 0,
            'max_score': 0,
            'avg_reward': 0,
            'avg_length': 0,
            'status': f'Error: {str(e)[:50]}'
        }


def evaluate_ppo_model(model_path: Path, num_episodes: int = 100) -> dict:
    """Evaluate a PPO model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filename = model_path.stem.lower()

    # Determine input dimension
    if 'floodfill' in filename:
        input_dim = 14
        use_flood_fill = True
    else:
        input_dim = 11
        use_flood_fill = False

    is_cnn = 'cnn' in filename

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # PPO uses actor-critic, we only need actor for evaluation
        if is_cnn:
            from core.networks import PPO_CNN_Actor
            actor = PPO_CNN_Actor(grid_size=10, input_channels=3, output_dim=3)
        else:
            from core.networks import PPO_MLP_Actor
            actor = PPO_MLP_Actor(input_dim=input_dim, output_dim=3)

        actor.load_state_dict(checkpoint['actor'])
        actor = actor.to(device)
        actor.eval()

        # Create environment
        env = VectorizedSnakeEnv(num_envs=num_episodes, grid_size=10)
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=use_flood_fill)

        # Run evaluation
        scores = []
        rewards = []
        lengths = []

        obs = env.reset()
        episode_rewards = np.zeros(num_episodes)
        episode_done = np.zeros(num_episodes, dtype=bool)

        max_steps = 1000
        for step in range(max_steps):
            # Encode observations
            if is_cnn:
                state = torch.tensor(obs, dtype=torch.float32, device=device)
            else:
                features = []
                for i in range(num_episodes):
                    if not episode_done[i]:
                        snake = [tuple(env.snakes[i, j].cpu().numpy()) for j in range(env.lengths[i])]
                        food = tuple(env.foods[i].cpu().numpy())
                        direction = env.directions[i].item()
                        feat = encoder.encode(snake, food, direction)
                        features.append(feat)
                    else:
                        features.append(np.zeros(input_dim, dtype=np.float32))
                state = torch.tensor(np.array(features), dtype=torch.float32, device=device)

            # Get actions (sample from policy)
            with torch.no_grad():
                action_probs = actor(state)
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()

            # Step environment
            obs, reward, done, info = env.step(actions)

            episode_rewards += reward.cpu().numpy() * ~episode_done

            new_done = done.cpu().numpy() & ~episode_done
            if new_done.any():
                done_indices = np.where(new_done)[0]
                for idx in done_indices:
                    scores.append(env.foods_eaten[idx].item())
                    rewards.append(episode_rewards[idx])
                    lengths.append(step + 1)
                episode_done |= done.cpu().numpy()

            if episode_done.all():
                break

        remaining = ~episode_done
        if remaining.any():
            for idx in np.where(remaining)[0]:
                scores.append(env.foods_eaten[idx].item())
                rewards.append(episode_rewards[idx])
                lengths.append(max_steps)

        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'avg_reward': np.mean(rewards),
            'avg_length': np.mean(lengths),
            'status': 'OK'
        }

    except Exception as e:
        return {
            'avg_score': 0,
            'std_score': 0,
            'max_score': 0,
            'avg_reward': 0,
            'avg_length': 0,
            'status': f'Error: {str(e)[:50]}'
        }


def evaluate_reinforce_model(model_path: Path, num_episodes: int = 100) -> dict:
    """Evaluate a REINFORCE model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filename = model_path.stem.lower()

    if 'floodfill' in filename:
        input_dim = 14
        use_flood_fill = True
    else:
        input_dim = 11
        use_flood_fill = False

    is_cnn = 'cnn' in filename

    try:
        checkpoint = torch.load(model_path, map_location=device)

        if is_cnn:
            from core.networks import PolicyNetwork_CNN
            policy = PolicyNetwork_CNN(grid_size=10, input_channels=3, output_dim=3)
        else:
            from core.networks import PolicyNetwork_MLP
            policy = PolicyNetwork_MLP(input_dim=input_dim, output_dim=3)

        policy.load_state_dict(checkpoint['policy'])
        policy = policy.to(device)
        policy.eval()

        env = VectorizedSnakeEnv(num_envs=num_episodes, grid_size=10)
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=use_flood_fill)

        scores = []
        rewards = []
        lengths = []

        obs = env.reset()
        episode_rewards = np.zeros(num_episodes)
        episode_done = np.zeros(num_episodes, dtype=bool)

        max_steps = 1000
        for step in range(max_steps):
            if is_cnn:
                state = torch.tensor(obs, dtype=torch.float32, device=device)
            else:
                features = []
                for i in range(num_episodes):
                    if not episode_done[i]:
                        snake = [tuple(env.snakes[i, j].cpu().numpy()) for j in range(env.lengths[i])]
                        food = tuple(env.foods[i].cpu().numpy())
                        direction = env.directions[i].item()
                        feat = encoder.encode(snake, food, direction)
                        features.append(feat)
                    else:
                        features.append(np.zeros(input_dim, dtype=np.float32))
                state = torch.tensor(np.array(features), dtype=torch.float32, device=device)

            with torch.no_grad():
                action_probs = policy(state)
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()

            obs, reward, done, info = env.step(actions)

            episode_rewards += reward.cpu().numpy() * ~episode_done

            new_done = done.cpu().numpy() & ~episode_done
            if new_done.any():
                done_indices = np.where(new_done)[0]
                for idx in done_indices:
                    scores.append(env.foods_eaten[idx].item())
                    rewards.append(episode_rewards[idx])
                    lengths.append(step + 1)
                episode_done |= done.cpu().numpy()

            if episode_done.all():
                break

        remaining = ~episode_done
        if remaining.any():
            for idx in np.where(remaining)[0]:
                scores.append(env.foods_eaten[idx].item())
                rewards.append(episode_rewards[idx])
                lengths.append(max_steps)

        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'avg_reward': np.mean(rewards),
            'avg_length': np.mean(lengths),
            'status': 'OK'
        }

    except Exception as e:
        return {
            'avg_score': 0,
            'std_score': 0,
            'max_score': 0,
            'avg_reward': 0,
            'avg_length': 0,
            'status': f'Error: {str(e)[:50]}'
        }


def evaluate_a2c_model(model_path: Path, num_episodes: int = 100) -> dict:
    """Evaluate an A2C model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filename = model_path.stem.lower()

    if 'floodfill' in filename:
        input_dim = 14
        use_flood_fill = True
    else:
        input_dim = 11
        use_flood_fill = False

    try:
        checkpoint = torch.load(model_path, map_location=device)

        from core.networks import PPO_MLP_Actor
        actor = PPO_MLP_Actor(input_dim=input_dim, output_dim=3)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor = actor.to(device)
        actor.eval()

        env = VectorizedSnakeEnv(num_envs=num_episodes, grid_size=10)
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=use_flood_fill)

        scores = []
        rewards = []
        lengths = []

        obs = env.reset()
        episode_rewards = np.zeros(num_episodes)
        episode_done = np.zeros(num_episodes, dtype=bool)

        max_steps = 1000
        for step in range(max_steps):
            features = []
            for i in range(num_episodes):
                if not episode_done[i]:
                    snake = [tuple(env.snakes[i, j].cpu().numpy()) for j in range(env.lengths[i])]
                    food = tuple(env.foods[i].cpu().numpy())
                    direction = env.directions[i].item()
                    feat = encoder.encode(snake, food, direction)
                    features.append(feat)
                else:
                    features.append(np.zeros(input_dim, dtype=np.float32))
            state = torch.tensor(np.array(features), dtype=torch.float32, device=device)

            with torch.no_grad():
                action_probs = actor(state)
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()

            obs, reward, done, info = env.step(actions)

            episode_rewards += reward.cpu().numpy() * ~episode_done

            new_done = done.cpu().numpy() & ~episode_done
            if new_done.any():
                done_indices = np.where(new_done)[0]
                for idx in done_indices:
                    scores.append(env.foods_eaten[idx].item())
                    rewards.append(episode_rewards[idx])
                    lengths.append(step + 1)
                episode_done |= done.cpu().numpy()

            if episode_done.all():
                break

        remaining = ~episode_done
        if remaining.any():
            for idx in np.where(remaining)[0]:
                scores.append(env.foods_eaten[idx].item())
                rewards.append(episode_rewards[idx])
                lengths.append(max_steps)

        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'avg_reward': np.mean(rewards),
            'avg_length': np.mean(lengths),
            'status': 'OK'
        }

    except Exception as e:
        return {
            'avg_score': 0,
            'std_score': 0,
            'max_score': 0,
            'avg_reward': 0,
            'avg_length': 0,
            'status': f'Error: {str(e)[:50]}'
        }


def main():
    weights_dir = Path('results/weights')

    print("=" * 100)
    print("EVALUATING ALL TRAINED MODELS")
    print("=" * 100)
    print()

    # Get all model files
    model_files = sorted(weights_dir.glob('*.pt'))

    results = []

    for model_path in model_files:
        filename = model_path.stem.lower()
        print(f"Evaluating {model_path.name}...", end=" ", flush=True)

        # Determine model type and evaluate
        if 'ppo' in filename and 'cnn' not in filename and filename not in ['ppo.pt']:
            result = evaluate_ppo_model(model_path, num_episodes=50)
        elif 'reinforce' in filename and 'cnn' not in filename and filename not in ['reinforce.pt']:
            result = evaluate_reinforce_model(model_path, num_episodes=50)
        elif 'a2c' in filename:
            result = evaluate_a2c_model(model_path, num_episodes=50)
        elif any(x in filename for x in ['dqn', 'double', 'dueling', 'per']):
            result = evaluate_dqn_model(model_path, num_episodes=50)
        else:
            result = {'avg_score': 0, 'std_score': 0, 'max_score': 0,
                     'avg_reward': 0, 'avg_length': 0, 'status': 'Unknown type'}

        result['model'] = model_path.name
        results.append(result)

        if result['status'] == 'OK':
            print(f"Score: {result['avg_score']:.2f} +/- {result['std_score']:.2f}")
        else:
            print(f"{result['status']}")

    # Print summary table
    print()
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print()
    print(f"{'Model':<45} {'Avg Score':>10} {'Max Score':>10} {'Avg Reward':>12} {'Avg Length':>12}")
    print("-" * 100)

    # Sort by average score
    results_sorted = sorted(results, key=lambda x: x['avg_score'], reverse=True)

    for r in results_sorted:
        if r['status'] == 'OK':
            print(f"{r['model']:<45} {r['avg_score']:>10.2f} {r['max_score']:>10.0f} {r['avg_reward']:>12.2f} {r['avg_length']:>12.1f}")
        else:
            print(f"{r['model']:<45} {'--':>10} {'--':>10} {'--':>12} {r['status']:>12}")

    print("-" * 100)

    # Print top 5
    print()
    print("TOP 5 MODELS:")
    for i, r in enumerate(results_sorted[:5], 1):
        if r['status'] == 'OK':
            print(f"  {i}. {r['model']}: {r['avg_score']:.2f} avg score, {r['max_score']:.0f} max")


if __name__ == '__main__':
    main()
