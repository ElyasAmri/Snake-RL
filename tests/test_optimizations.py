"""Quick test of optimizations"""
from core.environment_vectorized import VectorizedSnakeEnv
import time

print('Testing optimized BFS...')

# Test selective features
env_sel = VectorizedSnakeEnv(num_envs=8, use_flood_fill=True, use_selective_features=True)
env_enh = VectorizedSnakeEnv(num_envs=8, use_flood_fill=True, use_enhanced_features=True)

print(f'Selective features shape: {env_sel.reset().shape}')
print(f'Enhanced features shape: {env_enh.reset().shape}')

# Test BFS speed
start = time.time()
for i in range(10):
    obs = env_sel.reset()
end = time.time()
print(f'Selective BFS time (10 resets): {end-start:.3f}s')

start = time.time()
for i in range(10):
    obs = env_enh.reset()
end = time.time()
print(f'Enhanced BFS time (10 resets): {end-start:.3f}s')

print('All optimizations working!')
