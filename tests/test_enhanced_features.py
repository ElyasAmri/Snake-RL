"""Test enhanced features implementation"""
import torch
from core.environment_vectorized import VectorizedSnakeEnv
from core.state_representations import FeatureEncoder

def test_vectorized_env():
    """Test VectorizedSnakeEnv with enhanced features"""
    print('Testing VectorizedSnakeEnv with enhanced features...')

    env = VectorizedSnakeEnv(
        num_envs=8,
        grid_size=10,
        use_flood_fill=True,
        use_enhanced_features=True
    )

    obs = env.reset()
    print(f'Initial observation shape: {obs.shape}')
    assert obs.shape == (8, 24), f"Expected shape (8, 24), got {obs.shape}"

    # Run a few steps
    for i in range(10):
        actions = torch.randint(0, 3, (env.num_envs,), device=env.device)
        obs, rewards, dones, info = env.step(actions)
        print(f'Step {i+1}: obs_shape={obs.shape}, dones={dones.sum().item()}/{env.num_envs}')

    print('\nChecking feature ranges:')
    print(f'  Min value: {obs.min().item():.4f}')
    print(f'  Max value: {obs.max().item():.4f}')
    print(f'  Mean value: {obs.mean().item():.4f}')

    print('VectorizedSnakeEnv test passed!\n')


def test_feature_encoder():
    """Test FeatureEncoder with enhanced features"""
    print('Testing FeatureEncoder with enhanced features...')

    encoder = FeatureEncoder(
        grid_size=10,
        use_flood_fill=True,
        use_enhanced_features=True
    )

    # Test case 1: Basic snake
    snake = [(5, 5), (4, 5), (3, 5)]
    food = (7, 3)
    direction = 1  # RIGHT

    obs = encoder.encode(snake, food, direction)
    print(f'Observation shape: {obs.shape}')
    assert obs.shape == (24,), f"Expected shape (24,), got {obs.shape}"

    print('Feature breakdown:')
    print(f'  Danger detection [0-3]: {obs[0:4]}')
    print(f'  Food direction [4-7]: {obs[4:8]}')
    print(f'  Current direction [8-10]: {obs[8:11]}')
    print(f'  Flood-fill [11-13]: {obs[11:14]}')
    print(f'  Escape routes [14-16]: {obs[14:17]}')
    print(f'  Tail direction [17-20]: {obs[17:21]}')
    print(f'  Tail reachability [21]: {obs[21]}')
    print(f'  Distance to tail [22]: {obs[22]}')
    print(f'  Snake length ratio [23]: {obs[23]}')

    print('FeatureEncoder test passed!\n')


def test_compatibility():
    """Test backward compatibility with flood-fill only and base features"""
    print('Testing backward compatibility...')

    # Test base features (11-dim)
    env_base = VectorizedSnakeEnv(
        num_envs=4,
        grid_size=10,
        use_flood_fill=False,
        use_enhanced_features=False
    )
    obs = env_base.reset()
    assert obs.shape == (4, 11), f"Expected (4, 11), got {obs.shape}"
    print(f'Base features: shape={obs.shape} [OK]')

    # Test flood-fill only (14-dim)
    env_flood = VectorizedSnakeEnv(
        num_envs=4,
        grid_size=10,
        use_flood_fill=True,
        use_enhanced_features=False
    )
    obs = env_flood.reset()
    assert obs.shape == (4, 14), f"Expected (4, 14), got {obs.shape}"
    print(f'Flood-fill features: shape={obs.shape} [OK]')

    # Test all features (24-dim)
    env_all = VectorizedSnakeEnv(
        num_envs=4,
        grid_size=10,
        use_flood_fill=True,
        use_enhanced_features=True
    )
    obs = env_all.reset()
    assert obs.shape == (4, 24), f"Expected (4, 24), got {obs.shape}"
    print(f'All features: shape={obs.shape} [OK]')

    print('Backward compatibility test passed!\n')


if __name__ == '__main__':
    test_feature_encoder()
    test_vectorized_env()
    test_compatibility()
    print('All tests passed successfully!')
