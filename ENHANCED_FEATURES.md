# Enhanced Features for Snake RL

## Overview
This document describes the enhanced features implemented to complement the existing flood-fill feature representation. These features improve the agent's ability to make strategic decisions and avoid self-trapping situations.

## Feature Dimensions

### Base Features (11 dimensions)
- **[0-3]**: Danger detection (straight, left, right, back)
- **[4-7]**: Food direction (up, right, down, left)
- **[8-10]**: Current direction (one-hot encoding, 3 bits for 4 directions)

### Flood-Fill Features (3 dimensions) - **use_flood_fill=True**
- **[11-13]**: Flood-fill free space (straight, right, left)
  - Normalized reachable space from each direction
  - Range: [0, 1] where 1 = entire grid accessible

### Enhanced Features (10 dimensions) - **use_enhanced_features=True**
- **[14-16]**: Escape route count (straight, right, left)
  - Number of safe adjacent cells from each direction
  - Normalized by max possible (4 directions)
  - Range: [0, 1]

- **[17-20]**: Tail direction (up, right, down, left)
  - Direction from head to tail
  - Following tail is often safe since it moves away

- **[21]**: Tail reachability (0 or 1)
  - Can the head reach the tail via flood-fill?
  - **High-impact feature** for preventing self-traps

- **[22]**: Distance to tail (normalized)
  - Manhattan distance to tail
  - Normalized by grid diagonal (max distance = 2 * (grid_size - 1))

- **[23]**: Snake length ratio
  - current_length / max_possible_length
  - Helps agent adapt strategy as it grows

## Total Dimensions
- **Base only**: 11 dimensions (use_flood_fill=False, use_enhanced_features=False)
- **With flood-fill**: 14 dimensions (use_flood_fill=True, use_enhanced_features=False)
- **With all features**: 24 dimensions (use_flood_fill=True, use_enhanced_features=True)

## Implementation Details

### 1. Escape Route Detection
**Purpose**: Complements flood-fill by providing immediate escape options

The escape route feature counts how many safe adjacent cells are available from each potential next position. This gives the agent information about maneuverability and helps avoid getting cornered.

**Computation**:
- For each direction (straight, right, left):
  - Check if the next position is safe
  - Count safe adjacent cells (up, down, left, right)
  - Normalize by 4 (max possible adjacent cells)

**Benefits**:
- Immediate feedback on movement flexibility
- Complements long-term flood-fill with short-term escape information
- Helps agent prefer moves with more options

### 2. Tail-Chasing Features
**Purpose**: Enable safe path-following strategy (easiest, high impact)

The tail-chasing features help the agent follow its tail, which is a safe strategy since the tail moves away as the snake advances.

**Components**:
- **Tail direction** (4 features): Direction indicators from head to tail
- **Tail reachability** (1 feature): Binary indicator if tail is reachable via flood-fill
- **Distance to tail** (1 feature): Manhattan distance to tail (normalized)

**Benefits**:
- Following tail prevents self-trapping in most situations
- Tail reachability is critical for avoiding dead-ends
- Simple but highly effective strategy

### 3. Body Awareness Features
**Purpose**: Help agent adapt strategy based on its size

The body awareness features provide information about the snake's current state relative to its maximum size.

**Components**:
- **Snake length ratio** (1 feature): current_length / grid_size^2

**Benefits**:
- Agent can adapt behavior as it grows
- Longer snakes need more caution
- Helps balance risk-taking with safety

## Usage

### With VectorizedSnakeEnv
```python
from core.environment_vectorized import VectorizedSnakeEnv

# Enable all features (24 dimensions)
env = VectorizedSnakeEnv(
    num_envs=256,
    grid_size=10,
    use_flood_fill=True,
    use_enhanced_features=True
)

obs = env.reset()  # Shape: (256, 24)
```

### With FeatureEncoder
```python
from core.state_representations import FeatureEncoder

# Enable all features (24 dimensions)
encoder = FeatureEncoder(
    grid_size=10,
    use_flood_fill=True,
    use_enhanced_features=True
)

obs = encoder.encode(snake, food, direction)  # Shape: (24,)
```

## Backward Compatibility

All existing code continues to work without modification:
- Default behavior unchanged (11-dimensional base features)
- Flood-fill features remain at 14 dimensions when use_flood_fill=True
- Enhanced features only activate when use_enhanced_features=True

## Training Considerations

### When to Use Enhanced Features
- **Always recommended** when using flood-fill features
- Especially useful for larger grid sizes (e.g., 15x15, 20x20)
- Critical for achieving high scores (tail-chasing prevents traps)

### Computational Cost
- Escape routes: Minimal (counts adjacent cells)
- Tail direction: Negligible (simple position comparison)
- Tail reachability: Moderate (BFS per environment)
- Distance to tail: Negligible (Manhattan distance)

**Note**: Tail reachability uses BFS, which adds computational cost. However, this is still much faster than the main flood-fill computation and provides high value.

### Network Architecture Considerations
When using 24-dimensional features:
- Increase hidden layer sizes proportionally (e.g., [256, 256] -> [384, 384])
- Consider using layer normalization for stable training
- The additional features may require more training episodes to fully utilize

## Testing

Run the test suite to verify functionality:
```bash
./venv/Scripts/python.exe tests/test_enhanced_features.py
```

The test suite verifies:
- Correct feature dimensions for all configurations
- Feature value ranges [0, 1]
- Backward compatibility with existing code
- Proper functioning during environment steps

## References

Based on improvements identified in `improvements.txt`:
1. Escape route detection (complements flood-fill)
2. Tail-chasing features (easiest, high impact)
3. Body awareness features (strategy adaptation)

All features are designed to complement the existing flood-fill implementation and improve agent performance.
