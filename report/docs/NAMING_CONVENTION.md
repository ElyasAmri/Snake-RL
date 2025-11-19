# Naming Convention for Snake RL Models and Results

## Model Weights Format

```
YYYYMMDD_algorithm_architecture_features_episodes_descriptor.pt
```

### Components:
- **YYYYMMDD**: Date of training (e.g., 20251119)
- **algorithm**: RL algorithm (dqn, ppo, a2c, reinforce, etc.)
- **architecture**: Network type (mlp, cnn)
- **features**: Feature set used
  - `base` - 11-dim baseline features
  - `floodfill` - 14-dim with flood-fill
  - `selective` - 19-dim with selective features (tail features)
  - `enhanced` - 24-dim with all enhanced features
  - `enhanced_large` - 24-dim with larger network [256,256]
- **episodes**: Training duration (e.g., 1000ep, 500ep)
- **descriptor**: Optional notes (best, v1, v2, etc.)

### Examples:
```
20251119_dqn_mlp_selective_1000ep_best.pt
20251119_ppo_mlp_floodfill_500ep_v2.pt
20251120_a2c_cnn_enhanced_2000ep.pt
```

## Results Data Format

```
YYYYMMDD_comparison_type_episodes_descriptor.json
```

### Components:
- **YYYYMMDD**: Date of comparison
- **comparison**: Type of comparison
  - `comparison_4way` - Four-way comparison
  - `comparison_controlled` - Controlled test
  - `comparison_initial` - Initial exploration
  - `ablation` - Ablation study
  - `baseline` - Baseline test
- **type**: What's being tested (optional, can be part of descriptor)
- **episodes**: Episode count per model
- **descriptor**: Additional info (optimized, v1, features, etc.)

### Examples:
```
20251119_comparison_4way_1000ep_optimized.json
20251119_comparison_controlled_500ep_v1.json
20251120_ablation_tail_features_1000ep.json
```

## Current Models (as of 2025-11-19)

### Model Weights (`results/weights/`):
| Filename | Description | Score | Status |
|----------|-------------|-------|--------|
| `20251119_dqn_mlp_selective_1000ep_best.pt` | Best performer - selective features | 16.26 | Winner |
| `20251119_dqn_mlp_enhanced_large_1000ep.pt` | Enhanced with large network [256,256] | 16.07 | Good |
| `20251119_dqn_mlp_enhanced_1000ep.pt` | All enhanced features [128,128] | 15.41 | Good |
| `20251119_dqn_mlp_floodfill_1000ep_baseline.pt` | Baseline flood-fill only | 14.64 | Baseline |
| `20251119_dqn_mlp_floodfill_500ep_controlled.pt` | Controlled test baseline | 3.94 | Archive |
| `20251119_dqn_mlp_enhanced_500ep_controlled.pt` | Controlled test enhanced | 2.75 | Archive |

### Results Data (`results/data/`):
| Filename | Description |
|----------|-------------|
| `20251119_comparison_4way_1000ep_optimized.json` | Main comparison results (4 models) |
| `20251119_comparison_controlled_500ep.json` | Controlled comparison (baseline vs enhanced) |
| `20251119_comparison_initial_200ep.json` | Initial exploration |

## Training Script Run Names

When running training scripts, use descriptive run names:

```bash
# Good examples:
--run-name 20251119_1000ep
--run-name selective_optimized
--run-name ablation_tail
--run-name grid_size_15

# Avoid:
--run-name final
--run-name test
--run-name new
--run-name v1
--run-name v2
```

## Notes:
- Always include date for traceability
- Use descriptive names that indicate what was tested
- Avoid "final" and "v1/v2" - use descriptive words instead
- Mark best performers with `_best` suffix
- Archive older experiments but keep them for reference
- For iterations, use descriptive terms: `controlled`, `optimized`, `exploratory`, etc.
