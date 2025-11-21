# Performance Bottleneck Investigation Plan

## Current Performance
- **1024 episodes**: ~3 minutes (175 seconds) for basic DQN
- **5000 episodes**: ~199 minutes (3.3 hours) for basic DQN
- **Expected**: 175s × (5000/1024) = 854s = 14.2 minutes
- **Actual slowdown**: 14x slower than expected

## Suspected Bottlenecks

### 1. Replay Buffer Push Loop (HIGH PRIORITY)
**Location**: `scripts/training/train_dqn.py:350-357`

**Issue**:
```python
for i in range(self.num_envs):  # 256 iterations
    self.replay_buffer.push(
        states[i].cpu().numpy(),      # GPU → CPU transfer
        actions[i].item(),
        rewards[i].item(),
        next_states[i].cpu().numpy(),  # GPU → CPU transfer
        dones[i].item()
    )
```

**Cost per step**:
- 256 loop iterations
- 512 GPU→CPU tensor transfers (states + next_states)
- 256 × 3 = 768 `.item()` calls

**Test**:
```python
# Option A: Vectorized push (if replay buffer supports it)
self.replay_buffer.push_batch(
    states.cpu().numpy(),
    actions.cpu().numpy(),
    rewards.cpu().numpy(),
    next_states.cpu().numpy(),
    dones.cpu().numpy()
)

# Option B: Keep tensors on GPU longer
states_np = states.cpu().numpy()
next_states_np = next_states.cpu().numpy()
for i in range(self.num_envs):
    self.replay_buffer.push(
        states_np[i], actions[i].item(),
        rewards[i].item(), next_states_np[i],
        dones[i].item()
    )
```

**Expected improvement**: 2-5x speedup

---

### 2. Excessive Training Steps Per Environment Step (HIGH PRIORITY)
**Location**: `scripts/training/train_dqn.py:364-368`

**Issue**:
```python
num_train_steps = max(1, self.num_envs // 4)  # 256 // 4 = 64
for _ in range(num_train_steps):
    loss = self.train_step()  # Network forward/backward pass
```

**Cost per step**:
- 64 network forward passes
- 64 network backward passes
- 64 optimizer steps

**Test configurations**:
```python
# Current: 64 training steps per env step
num_train_steps = max(1, self.num_envs // 4)

# Test 1: Reduce to 1 step per env step
num_train_steps = 1

# Test 2: Train every N steps instead
if self.total_steps % 4 == 0:
    num_train_steps = max(1, self.num_envs // 4)

# Test 3: Use more reasonable ratio
num_train_steps = max(1, self.num_envs // 16)  # 16 instead of 64
```

**Expected improvement**: 2-4x speedup (but may affect learning quality)

---

### 3. Flood-Fill BFS Computation (MEDIUM PRIORITY - only affects flood-fill models)
**Location**: `core/environment_vectorized.py:595-660`

**Issue**:
```python
# Called 3 times per step (straight, right, left)
for env_idx in range(self.num_envs):  # 256 iterations
    # BFS flood-fill on CPU with Python loops
    queue = deque([start_pos])
    while queue:
        current = queue.popleft()
        # ... BFS logic
```

**Cost per step** (for flood-fill models only):
- 3 directions × 256 envs = 768 BFS operations
- Each BFS visits ~50-100 cells on average
- All on CPU with Python loops

**Test**:
```python
# Option A: Cache flood-fill results for similar states
# Option B: Reduce frequency (compute every N steps)
# Option C: Disable flood-fill entirely for speed test

# Quick test: disable flood-fill
use_flood_fill = False
```

**Expected improvement**: 2-3x speedup for flood-fill models only

---

### 4. Environment Step Overhead (LOW PRIORITY)
**Location**: `core/environment_vectorized.py`

**Possible issues**:
- Feature computation every step
- Unnecessary CPU/GPU transfers
- Inefficient collision detection

**Test**: Profile a single `env.step()` call

---

### 5. Logging/Metrics Overhead (LOW PRIORITY)
**Location**: Various print statements and metrics tracking

**Test**: Disable all logging and metrics
```python
verbose = False
# Comment out all print() statements
# Disable metrics.add_episode(), metrics.add_loss()
```

**Expected improvement**: Minimal (<5%)

---

## Test Procedure

### Phase 1: Identify Primary Bottleneck (Quick Tests)

**Test 1A: Minimal Training (Replay Buffer Test)**
```bash
# Modify train_dqn.py to skip training entirely
# Comment out lines 362-368 (training loop)
./venv/Scripts/python.exe scripts/training/train_dqn_mlp.py --episodes 100
```
**If fast**: Replay buffer push is NOT the bottleneck
**If slow**: Replay buffer push IS the bottleneck

---

**Test 1B: Reduce Training Frequency**
```python
# Change line 364 from:
num_train_steps = max(1, self.num_envs // 4)  # 64 steps
# To:
num_train_steps = 1  # 1 step
```
```bash
./venv/Scripts/python.exe scripts/training/train_dqn_mlp.py --episodes 100
```
**Expected**: If this is 64x faster, training frequency is the bottleneck

---

**Test 1C: Vectorize Replay Buffer Push**
```python
# Modify lines 350-357 to do batch CPU transfer:
states_np = states.cpu().numpy()
actions_np = actions.cpu().numpy()
rewards_np = rewards.cpu().numpy()
next_states_np = next_states.cpu().numpy()
dones_np = dones.cpu().numpy()

for i in range(self.num_envs):
    self.replay_buffer.push(
        states_np[i], actions_np[i],
        rewards_np[i], next_states_np[i],
        dones_np[i]
    )
```
```bash
./venv/Scripts/python.exe scripts/training/train_dqn_mlp.py --episodes 100
```
**Expected**: 2-3x speedup if GPU/CPU transfer is bottleneck

---

**Test 1D: Profile Single Step**
```python
import time
# Add timing to one iteration of main loop
start = time.time()
actions = self.select_actions(states)
t1 = time.time()
next_states, rewards, dones, info = self.env.step(actions)
t2 = time.time()
# ... replay buffer push
t3 = time.time()
# ... training
t4 = time.time()

print(f"Select: {t1-start:.4f}s, Step: {t2-t1:.4f}s, Push: {t3-t2:.4f}s, Train: {t4-t3:.4f}s")
```

---

### Phase 2: Optimize Primary Bottleneck

Based on Phase 1 results, implement the optimization that provides the most speedup.

---

### Phase 3: Verify Episode Count vs Steps

**Sanity check**: Ensure training is actually running the correct number of episodes

```python
# Add to train() method
print(f"Total environment steps: {self.total_steps}")
print(f"Episodes completed: {self.episode}")
print(f"Steps per episode (avg): {self.total_steps / max(1, self.episode)}")
```

**Expected**:
- With 256 parallel envs
- Each episode ~50-200 steps
- For 5000 episodes: ~250,000 - 1,000,000 total steps

---

## Quick Win Tests (5 minutes each)

### Test A: Reduce num_envs
```bash
# Test with fewer parallel environments
./venv/Scripts/python.exe scripts/training/train_dqn_mlp.py --episodes 100 --envs 32
./venv/Scripts/python.exe scripts/training/train_dqn_mlp.py --episodes 100 --envs 64
./venv/Scripts/python.exe scripts/training/train_dqn_mlp.py --episodes 100 --envs 256
```
**Expected**: If 32 envs is much faster, the problem is the per-env loops

---

### Test B: Disable Flood-Fill Globally
```python
# In environment_vectorized.py, force disable:
self.use_flood_fill = False
```
```bash
./venv/Scripts/python.exe scripts/training/train_dqn_mlp.py --episodes 100
```
**Expected**: Same speed (basic DQN doesn't use flood-fill anyway)

---

### Test C: Training Step Frequency
```python
# Test different training frequencies:

# Original: 64 steps per env step
num_train_steps = max(1, self.num_envs // 4)

# Test 1: 16 steps per env step
num_train_steps = max(1, self.num_envs // 16)

# Test 2: 1 step per env step
num_train_steps = 1

# Test 3: Train every 4th env step
if self.total_steps % 4 == 0:
    num_train_steps = 16
```

---

## Profiling

### CPU Profiling
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

trainer.train(verbose=False)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### GPU Profiling
```bash
# Use PyTorch profiler
python -m torch.utils.bottleneck scripts/training/train_dqn_mlp.py --episodes 10
```

---

## Expected Results Summary

| Bottleneck | Expected Speedup | Confidence |
|------------|------------------|------------|
| Replay buffer push (vectorize) | 2-3x | High |
| Training frequency (reduce 64→4) | 10-15x | High |
| Flood-fill BFS (cache/disable) | 2-3x (flood-fill only) | Medium |
| GPU/CPU transfers (batch) | 1.5-2x | Medium |
| Logging overhead | <1.1x | Low |

---

## Action Items

1. **Run Test 1B first** (reduce training frequency) - likely biggest win
2. **Run Test 1C second** (vectorize replay buffer) - easy optimization
3. **Profile with Test 1D** to confirm bottleneck
4. **Optimize based on results**
5. **Re-run 5000 episode training** with optimizations

---

## Notes

- Current training: **14x slower than expected**
- Target: **Get within 2x of expected time** (acceptable overhead)
- Minimum goal: **5x speedup** (from 199 min → 40 min for 5000 episodes)
- Stretch goal: **10x speedup** (from 199 min → 20 min for 5000 episodes)
