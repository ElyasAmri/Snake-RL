# GPU Utilization for Reinforcement Learning

## 1. GPU Acceleration Fundamentals

GPU acceleration leverages parallel processing capabilities to speed up both neural network training and environment execution.

### Why GPU for RL?

**Neural Network Operations**:
- Forward pass: Compute Q-values for batch of states
- Backward pass: Compute gradients and update weights
- Both operations involve matrix multiplications, highly parallelizable

**Parallel Environment Execution**:
- Run N game instances simultaneously
- Collect N experiences per step
- Dramatically increases data collection speed

### Mathematical Foundation

In deep RL, we optimize a loss function L(theta) where theta represents network parameters:

```
theta_{t+1} = theta_t - alpha * nabla_theta L(theta_t)
```

On GPU, this computation is parallelized across:
- **Batch dimension**: Multiple experiences processed simultaneously
- **Network layers**: Parallel matrix operations
- **Multiple environments**: Vectorized execution

---

## 2. Vectorized Environments

Vectorized environments allow running N environments in parallel, dramatically improving sample collection speed.

### Concept

Instead of:
```
for env in environments:
    state, reward, done = env.step(action)
```

Execute simultaneously:
```
states, rewards, dones = vectorized_env.step(actions)
```

All N environments process actions in parallel on GPU.

### Architecture

**Single Environment** (CPU):
- Grid state: (H, W, C)
- One snake, one food, one reward per step
- ~500 FPS

**Vectorized Environment** (GPU):
- Grid states: (N, H, W, C)
- N snakes, N foods, N rewards per step
- ~120,000 FPS with N=256

### Key Components

1. **State Tensor**: All environment states in single tensor
2. **Parallel Updates**: All games updated simultaneously
3. **Auto-reset**: Done environments automatically reset
4. **Batch Actions**: Agent selects N actions at once

### Benefits

- 100-300x speedup over single CPU environment
- Efficient GPU utilization
- Same computational cost for 1 or 256 environments (GPU parallelism)

---

## 3. PyTorch GPU Setup

### Device Management

Basic setup for using CUDA:

```python
import torch

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU
model = DQNNetwork(...).to(device)

# Move data to GPU
state_tensor = torch.FloatTensor(state).to(device)
```

### Key Considerations

1. **Data Transfer Overhead**:
   - CPU-GPU transfer is slow
   - Minimize transfers, keep data on GPU
   - Use pinned memory for faster async transfer

2. **Memory Management**:
   - GPU memory is limited (8-24 GB typically)
   - Monitor usage to avoid OOM errors
   - Clear cache periodically if needed

3. **Batch Processing**:
   - Larger batches better utilize GPU
   - Too large: OOM error
   - Too small: Underutilized GPU

---

## 4. Memory Optimization Techniques

### Pinned Memory

Pinned (page-locked) memory enables faster CPU-GPU transfer:

```python
# Slower (pageable memory)
tensor = torch.zeros(size)

# Faster (pinned memory)
tensor = torch.zeros(size, pin_memory=True)

# Async transfer
tensor_gpu = tensor.to(device, non_blocking=True)
```

**Benefits**:
- 2-3x faster transfer to GPU
- Enables asynchronous transfer
- Important for replay buffer sampling

### Gradient Accumulation

When GPU memory is limited, accumulate gradients over multiple micro-batches:

**Effective batch size = micro_batch_size x accumulation_steps**

**Concept**:
- Forward/backward on micro-batch
- Don't update weights yet
- Accumulate gradients
- After N micro-batches, update weights once

**Benefits**:
- Simulate large batch with limited memory
- Same gradient as full batch (mathematically equivalent)
- Trade computation time for memory

### Mixed Precision Training

Use 16-bit floats instead of 32-bit:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 2x memory reduction
- 2-3x speedup on modern GPUs
- Minimal accuracy loss

---

## 5. Batch Processing Strategies

### Optimal Batch Sizes

**For different GPUs**:
- RTX 3060/3070: 64-128
- RTX 3080/3090: 256-512
- RTX 4090: 512-1024
- A100: 1024-2048

**Trade-offs**:
- Larger batches: Better GPU utilization, more stable gradients, slower iteration
- Smaller batches: Faster iteration, more stochastic, underutilized GPU

### Dynamic Batching

Adjust batch size based on available memory:

```
Start with large batch size
If OOM error: reduce by 50%
Repeat until successful
Use largest feasible batch size
```

### Replay Buffer Sampling

**Efficient sampling strategy**:
1. Store experiences in pinned CPU memory
2. Sample indices on CPU
3. Transfer batch to GPU asynchronously
4. Process while next batch transfers

---

## 6. Performance Benchmarks

### Expected Speedups

| Configuration | FPS (frames/sec) | Training Time (1M steps) | Speedup |
|---------------|------------------|--------------------------|---------|
| Single CPU env | ~500 | ~33 min | 1x |
| 16 CPU envs | ~6,000 | ~3 min | 12x |
| 64 GPU envs | ~35,000 | ~30 sec | 70x |
| 256 GPU envs | ~120,000 | ~8 sec | 240x |

### Factors Affecting Performance

1. **Environment Complexity**:
   - Simple games: Higher FPS
   - Complex physics: Lower FPS

2. **Network Size**:
   - Small networks: Less GPU benefit
   - Large networks: More GPU benefit

3. **Batch Size**:
   - Small batches: Poor GPU utilization
   - Large batches: Good GPU utilization

4. **Transfer Overhead**:
   - Keep data on GPU
   - Minimize CPU-GPU transfers

---

## 7. When GPU Helps vs. Doesn't Help

### GPU Helps

- Neural network forward/backward passes (always faster)
- Batch processing with batch_size >= 32
- Vectorized environment execution (many parallel environments)
- Large replay buffers with batch sampling

### GPU Doesn't Help Much

- Single environment execution
- Very small networks (<1000 parameters)
- CPU-bound game logic (complex Python code)
- Small batch sizes (<16)
- Excessive CPU-GPU data transfers

### Recommendation for Snake

**Use GPU for**:
- DQN network training (batch size 64+)
- Vectorized Snake environments (64+ parallel games)
- Replay buffer operations (batch sampling)

**CPU is fine for**:
- Single game visualization
- Initial prototyping
- Very small experiments

---

## 8. Best Practices

### Data Management

1. **Keep Data on GPU**:
   - Move model to GPU once
   - Keep replay buffer samples on GPU
   - Avoid repeated CPU-GPU transfers

2. **Use Pinned Memory**:
   - For replay buffer storage
   - For async data transfer
   - 2-3x faster transfers

3. **Batch Operations**:
   - Process multiple states at once
   - Vectorize environment operations
   - Maximize parallel execution

### Memory Management

1. **Monitor Usage**:
   - Check GPU memory periodically
   - Identify memory leaks early
   - Clear cache if needed

2. **Optimize Buffer Sizes**:
   - Balance buffer size with memory
   - Consider using CPU for large buffers
   - Transfer only batches to GPU

3. **Delete Unused Tensors**:
   - Explicitly delete large tensors
   - Use context managers
   - Clear computational graphs

### Training Optimization

1. **Start Small, Scale Up**:
   - Prototype on CPU
   - Move to GPU when scaling
   - Increase batch/environment count gradually

2. **Profile Performance**:
   - Measure FPS improvements
   - Identify bottlenecks
   - Optimize critical paths

3. **Balance Computation**:
   - Don't bottleneck on CPU game logic
   - Vectorize environment operations
   - Minimize Python loops

---

## 9. Multi-GPU Training

For very large-scale training, distribute across multiple GPUs.

### Data Parallelism

Replicate model on each GPU, process different batches:

```
GPU 0: Process batch 0-63
GPU 1: Process batch 64-127
GPU 2: Process batch 128-191
GPU 3: Process batch 192-255

Synchronize gradients
Update all models
```

### Environment Parallelism

Run different environments on each GPU:

```
GPU 0: Environments 0-63
GPU 1: Environments 64-127
GPU 2: Environments 128-191
GPU 3: Environments 192-255

Collect all experiences
Train on combined data
```

### Trade-offs

**Benefits**:
- 2-4x further speedup
- Handle larger batch sizes
- More parallel environments

**Challenges**:
- Synchronization overhead
- More complex code
- Diminishing returns beyond 4 GPUs

---

## Summary

GPU acceleration is essential for efficient RL training:

### Key Benefits
- 100-300x speedup over CPU
- Parallel neural network operations
- Vectorized environment execution
- Efficient batch processing

### Implementation Strategy
1. Use PyTorch with CUDA
2. Implement vectorized environments (64-256 parallel)
3. Use pinned memory for transfers
4. Optimize batch sizes (64-512)
5. Keep data on GPU, minimize transfers

### For Snake RL
- Train DQN with 256 parallel environments
- Batch size 128-256
- Expected: 100,000+ FPS
- Training time: Minutes instead of hours

GPU utilization transforms RL from days of training to minutes, making experimentation and iteration much faster.
