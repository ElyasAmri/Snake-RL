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
    assert obs.shape == (8, 23), f"Expected shape (8, 23), got {obs.shape}"

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
    assert obs.shape == (23,), f"Expected shape (23,), got {obs.shape}"

    print('Feature breakdown:')
    print(f'  Danger detection [0-2]: {obs[0:3]}')
    print(f'  Food direction [3-6]: {obs[3:7]}')
    print(f'  Current direction [7-9]: {obs[7:10]}')
    print(f'  Flood-fill [10-12]: {obs[10:13]}')
    print(f'  Escape routes [13-15]: {obs[13:16]}')
    print(f'  Tail direction [16-19]: {obs[16:20]}')
    print(f'  Tail reachability [20]: {obs[20]}')
    print(f'  Distance to tail [21]: {obs[21]}')
    print(f'  Snake length ratio [22]: {obs[22]}')

    print('FeatureEncoder test passed!\n')


def test_compatibility():
    """Test backward compatibility with flood-fill only and base features"""
    print('Testing backward compatibility...')

    # Test base features (10-dim)
    env_base = VectorizedSnakeEnv(
        num_envs=4,
        grid_size=10,
        use_flood_fill=False,
        use_enhanced_features=False
    )
    obs = env_base.reset()
    assert obs.shape == (4, 10), f"Expected (4, 10), got {obs.shape}"
    print(f'Base features: shape={obs.shape} [OK]')

    # Test flood-fill only (13-dim)
    env_flood = VectorizedSnakeEnv(
        num_envs=4,
        grid_size=10,
        use_flood_fill=True,
        use_enhanced_features=False
    )
    obs = env_flood.reset()
    assert obs.shape == (4, 13), f"Expected (4, 13), got {obs.shape}"
    print(f'Flood-fill features: shape={obs.shape} [OK]')

    # Test all features (23-dim)
    env_all = VectorizedSnakeEnv(
        num_envs=4,
        grid_size=10,
        use_flood_fill=True,
        use_enhanced_features=True
    )
    obs = env_all.reset()
    assert obs.shape == (4, 23), f"Expected (4, 23), got {obs.shape}"
    print(f'All features: shape={obs.shape} [OK]')

    print('Backward compatibility test passed!\n')


if __name__ == '__main__':
    test_feature_encoder()
    test_vectorized_env()
    test_compatibility()
    print('All tests passed successfully!')
