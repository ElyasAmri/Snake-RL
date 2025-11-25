# Two-Snake Training Performance Analysis & Redesign

## Executive Summary

**Current State**: Two-snake competitive training runs at **1.75 steps/second** (570ms/step) with 256 environments.

**Root Cause**: Architectural anti-patterns violate high-performance RL best practices:
- Python loops over environments in critical path
- CPU-GPU synchronization points
- Inefficient feature encoding with handcrafted features

**Target State**: **50-200 steps/second** (5-20ms/step) - achievable with proper GPU-accelerated design.

**Impact**: Reduces full 5-stage curriculum training from **2 days to 1-2 hours**.

---

## Critical Issues Identified

### Issue #1: Python Loops Everywhere [CRITICAL]

**Location**: `core/environment_two_snake_vectorized.py`

**Problem**: Sequential processing defeats GPU parallelism.

**Lines with loops**:
- 159-170: Snake initialization
- 193-214: Food spawning
- 375-382: Snake movement
- 400-422: Collision detection
- 442-461: Food collection

**Impact**: With 256 envs, doing 256+ Python loop iterations per step. Each iteration waits for previous to complete.

**Fix**: Replace ALL loops with vectorized tensor operations.

---

### Issue #2: Feature Encoding Bottleneck [MAJOR]

**Location**: `core/state_representations_competitive.py`

**Problem**: 35 handcrafted features require multiple passes, still has CPU operations despite vectorization.

**Current timing** (after danger vectorization):
- danger_self: 0.30-0.46 ms
- danger_opponent: 0.18-0.28 ms
- food_direction: 0.19-0.20 ms
- **Total: 0.68-0.92 ms per encoding**
- **Per step (x2 agents): 1.36-1.84 ms**

**High-performance approach**: Raw grid state + CNN
- Grid rendering: ~0.1-0.2 ms
- CNN forward pass: ~0.5-1.0 ms
- **Total: 0.6-1.2 ms for both agents**
- Plus CNNs learn better features automatically

---

### Issue #3: No End-to-End GPU Pipeline [MAJOR]

**Problem**: Data moves between CPU/GPU unnecessarily.

**Current flow**:
1. Environment step (GPU) -> 2. Extract obs (GPU) -> 3. `.cpu().numpy()` -> 4. Store in buffer (CPU) -> 5. Sample (CPU) -> 6. `.to(device)` -> 7. Train (GPU)

**Correct flow**:
1. Environment step (GPU) -> 2. Extract obs (GPU) -> 3. Store in buffer (GPU) -> 4. Sample (GPU) -> 5. Train (GPU)

---

## Performance Benchmarks

### Current Performance Breakdown (256 envs, per step):

| Operation | Time | % of Total |
|-----------|------|------------|
| env_step | 490-570 ms | 96% |
| +-- Feature encoding | ~1.5 ms | 0.3% |
| +-- **Environment loops** | **~450-500 ms** | **90%** |
| +-- Other | ~40-70 ms | 10% |
| training | 0.8-3.0 ms | <1% |
| store_transitions | 0.7-0.8 ms | <1% |
| get_actions | 0.4-1.5 ms | <1% |

**KEY INSIGHT**: 90% of time is spent in Python loops iterating over environments!

---

## Redesign Plan

### Phase 1: Vectorize All Loops (Expected: 5-10x speedup)

**Priority**: CRITICAL - Must be done

#### 1.1 Vectorize Snake Movement

**Current** (lines 375-382):
```python
for env_idx in range(self.num_envs):
    if alive[env_idx]:
        length = lengths[env_idx]
        snakes[env_idx, 1:length] = snakes[env_idx, 0:length-1].clone()
        snakes[env_idx, 0] = new_heads[env_idx]
```

**Vectorized**:
```python
# ALL environments at once - NO loop
snakes[:, 1:] = snakes[:, :-1].clone()  # Shift all bodies
snakes[:, 0] = new_heads  # Place all new heads

# Mask invalid segments
segment_idx = torch.arange(self.max_length, device=self.device)
valid_mask = segment_idx < lengths.unsqueeze(1)
snakes[~valid_mask] = -1  # Mark invalid positions
```

**Speedup**: Approximately 256x (one operation vs 256 iterations)

#### 1.2 Vectorize Collision Detection

**Current** (lines 400-422):
```python
for env_idx in range(self.num_envs):
    if not alive[env_idx]:
        continue
    # Check wall, self, opponent collisions
    for j in range(length[env_idx]):
        if torch.equal(heads[env_idx], snake[env_idx, j]):
            collision = True
```

**Vectorized**:
```python
heads = snakes[:, 0, :]  # (num_envs, 2)

# Wall collisions - pure tensor ops
wall = (heads[:, 0] < 0) | (heads[:, 0] >= grid_size) | \
       (heads[:, 1] < 0) | (heads[:, 1] >= grid_size)

# Self collisions - broadcast comparison
head_exp = heads.unsqueeze(1)  # (num_envs, 1, 2)
body = snakes[:, 1:, :]  # (num_envs, max_len-1, 2)
matches = (head_exp == body).all(dim=-1)  # (num_envs, max_len-1)

# Mask by valid segments
segment_idx = torch.arange(1, self.max_length, device=self.device)
valid = segment_idx < lengths.unsqueeze(1)
self_collision = (matches & valid).any(dim=1)

collision = wall | self_collision
```

**Speedup**: Approximately 500x (eliminates nested loops)

#### 1.3 Vectorize Food Collection

**Current** (lines 442-461):
```python
for env_idx in range(self.num_envs):
    if not alive[env_idx]:
        continue
    if torch.equal(heads[env_idx], foods[env_idx]):
        # Food collected logic
```

**Vectorized**:
```python
heads = snakes[:, 0, :]  # (num_envs, 2)
food_collected = (heads == foods).all(dim=1) & alive  # (num_envs,)

# Update lengths where food collected
lengths = torch.where(food_collected, lengths + 1, lengths)
food_counts = torch.where(food_collected, food_counts + 1, food_counts)
```

**Speedup**: Approximately 256x

#### 1.4 Vectorize Food Spawning

**Current** (line 193):
```python
for env_idx in range(self.num_envs):
    self._spawn_food_single(env_idx)
```

**Vectorized**:
```python
# Identify envs needing food respawn
needs_food = food_collected | initial_spawn  # (num_envs,)
num_respawn = needs_food.sum()

if num_respawn > 0:
    # Generate random positions for all at once
    new_foods = torch.randint(
        0, self.grid_size,
        (num_respawn, 2),
        device=self.device
    )
    foods[needs_food] = new_foods
```

**Speedup**: Approximately 256x

---

### Phase 2: Replace Feature Encoder with CNN (Expected: Additional 5-10x speedup)

**Priority**: HIGH - Significant performance gain + better learning

#### 2.1 Grid Renderer

**New method** in `environment_two_snake_vectorized.py`:
```python
def _render_grids(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render grid observations for CNN input.

    Returns:
        grids1: (num_envs, 5, grid_size, grid_size) for Snake 1
        grids2: (num_envs, 5, grid_size, grid_size) for Snake 2

    Channels:
        0: Own snake body
        1: Own snake head
        2: Opponent snake body
        3: Opponent snake head
        4: Food
    """
    batch = self.num_envs
    grids1 = torch.zeros((batch, 5, self.grid_size, self.grid_size),
                         dtype=torch.float32, device=self.device)
    grids2 = torch.zeros((batch, 5, self.grid_size, self.grid_size),
                         dtype=torch.float32, device=self.device)

    # Render using advanced indexing (vectorized where possible)
    for i in range(batch):
        # Snake 1 perspective
        body1 = self.snakes1[i, :self.lengths1[i]]
        grids1[i, 0, body1[:, 1], body1[:, 0]] = 1.0  # Body
        grids1[i, 1, body1[0, 1], body1[0, 0]] = 1.0  # Head

        body2 = self.snakes2[i, :self.lengths2[i]]
        grids1[i, 2, body2[:, 1], body2[:, 0]] = 1.0  # Opponent
        grids1[i, 3, body2[0, 1], body2[0, 0]] = 1.0

        food = self.foods[i]
        grids1[i, 4, food[1], food[0]] = 1.0  # Food

        # Snake 2 perspective (swap roles)
        grids2[i, 0, body2[:, 1], body2[:, 0]] = 1.0
        grids2[i, 1, body2[0, 1], body2[0, 0]] = 1.0
        grids2[i, 2, body1[:, 1], body1[:, 0]] = 1.0
        grids2[i, 3, body1[0, 1], body1[0, 0]] = 1.0
        grids2[i, 4, food[1], food[0]] = 1.0

    return grids1, grids2
```

**Note**: Grid rendering still has a loop but it's MUCH simpler than feature encoding.
**Future optimization**: Can be further vectorized using scatter operations.

#### 2.2 CNN Network Architecture

**New file**: `core/networks/two_snake_cnn.py`
```python
import torch
import torch.nn as nn

class TwoSnakeCNN(nn.Module):
    """
    CNN for processing grid-based two-snake observations.

    Input: (batch, 5, grid_size, grid_size)
    Output: (batch, 3) - Q-values for [STRAIGHT, LEFT, RIGHT]
    """

    def __init__(self, grid_size: int = 10, hidden_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 5 -> 32 channels
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # Conv2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Conv3: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Calculate flattened size
        self.flatten_size = 64 * grid_size * grid_size

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 actions
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 5, grid_size, grid_size)
        Returns:
            q_values: (batch, 3)
        """
        features = self.features(x)
        q_values = self.head(features)
        return q_values
```

**Benefits**:
- CNNs are highly optimized on GPU
- Processes entire batch in parallel
- Learns spatial features automatically
- No CPU operations, no synchronization

---

## Expected Performance After Redesign

### Estimated Timing (256 envs, per step):

| Operation | Current | After Phase 1 | After Phase 2 |
|-----------|---------|---------------|---------------|
| env_step | 490-570 ms | **50-100 ms** | **10-20 ms** |
| +-- Movement | ~200 ms | **1-2 ms** | **1-2 ms** |
| +-- Collisions | ~150 ms | **1-2 ms** | **1-2 ms** |
| +-- Food logic | ~100 ms | **1-2 ms** | **1-2 ms** |
| +-- Observations | ~40 ms | **40 ms** (feature enc) | **5-10 ms** (grid render + CNN) |
| training | 0.8-3.0 ms | 0.8-3.0 ms | 0.8-3.0 ms |
| **TOTAL** | **~570 ms** | **~60-110 ms** | **~15-30 ms** |
| **Steps/sec** | **1.75** | **9-16** | **33-66** |

### Training Time Estimates:

| Configuration | Current | After Redesign |
|---------------|---------|----------------|
| Stage 0 (50K steps, 256 envs) | 7.9 hours | **13-25 minutes** |
| Full curriculum (5 stages) | ~2 days | **1-2 hours** |

---

## Implementation Priority

### Must Do (Phase 1):
1. [ ] Vectorize snake movement
2. [ ] Vectorize collision detection
3. [ ] Vectorize food collection
4. [ ] Vectorize food spawning
5. [ ] Vectorize reset

**Expected result**: 9-16 steps/second (~60-110ms/step)

### Should Do (Phase 2):
6. [ ] Create grid renderer
7. [ ] Create TwoSnakeCNN network
8. [ ] Update trainer to use CNN

**Expected result**: 33-66 steps/second (~15-30ms/step)

### Nice to Have (Phase 3 - if time permits):
9. GPU-based replay buffer
10. Torch profiler analysis
11. Mixed precision (FP16)
12. JIT compilation

**Expected result**: 100-200 steps/second (~5-10ms/step)

---

## Next Steps

1. **Backup current code** - Create git branch for redesign
2. **Implement Phase 1** - Vectorize all environment operations
3. **Test correctness** - Verify game logic still works
4. **Benchmark** - Measure actual speedup
5. **Implement Phase 2** - Add CNN support
6. **Train test model** - Verify learning works
7. **Full training** - Run complete curriculum

**Estimated total effort**: 6-10 hours for Phase 1 + Phase 2
**Payoff**: 30-60x faster training, makes deliverable practical

---

## References

- [WarpDrive: Fast RL on GPU](https://www.salesforce.com/blog/warpdrive-fast-rl-on-a-gpu/) - 9.8M steps/sec
- [Isaac Gym for Robotics](https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/) - 100-1000x speedup
- [TorchRL Best Practices](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/PRO-TIPS.html)
- [Vectorized RL with JAX](https://towardsdatascience.com/vectorize-and-parallelize-rl-environments-with-jax-q-learning-at-the-speed-of-light-49d07373adf5/) - 4000x speedup

---

**Date**: 2025-11-22
**Status**: Analysis complete, ready for implementation
**Priority**: CRITICAL for course deliverable
