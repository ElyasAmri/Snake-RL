"""
Video Presets

Configuration presets for the 8 narrative presentation videos.
"""

from typing import Dict, Any

# Video presets for presentation narrative
PRESETS: Dict[str, Dict[str, Any]] = {
    '01_random': {
        'name': 'Random Baseline',
        'agent_type': 'baseline',
        'baseline': 'random',
        'duration': 30,
        'show_features': False,
        'show_q_values': False,
        'heatmap': None,
        'description': 'Random agent showing unguided exploration'
    },

    '03_basic_trained': {
        'name': 'DQN Basic',
        'agent_type': 'trained',
        'weights_pattern': 'results/weights/dqn_basic_10000ep_*.pt',
        'network': 'dqn',
        'feature_mode': 'basic',
        'duration': 45,
        'show_features': True,
        'show_q_values': True,
        'heatmap': 'danger',
        'description': 'Basic DQN learns food collection and wall avoidance'
    },

    '04_greedy': {
        'name': 'Greedy A* Baseline',
        'agent_type': 'baseline',
        'baseline': 'greedy',
        'duration': 30,
        'show_features': False,
        'show_q_values': False,
        'heatmap': None,
        'show_path': True,
        'description': 'A* pathfinding - efficient but can self-trap'
    },

    '05_basic_trapping': {
        'name': 'DQN Basic (Self-Trapping)',
        'agent_type': 'trained',
        'weights_pattern': 'results/weights/dqn_basic_10000ep_*.pt',
        'network': 'dqn',
        'feature_mode': 'basic',
        'duration': 60,
        'show_features': True,
        'show_q_values': True,
        'heatmap': 'danger',
        'wait_for_trap': True,
        'description': 'Basic DQN without spatial awareness gets trapped'
    },

    '06_flood_fill': {
        'name': 'DQN Flood-Fill',
        'agent_type': 'trained',
        'weights_pattern': 'results/weights/dqn_flood-fill_10000ep_*.pt',
        'network': 'dqn',
        'feature_mode': 'flood-fill',
        'duration': 45,
        'show_features': True,
        'show_q_values': True,
        'heatmap': 'flood_fill',
        'description': 'Flood-fill features prevent self-trapping'
    },

    '07_enhanced': {
        'name': 'DQN Enhanced',
        'agent_type': 'trained',
        'weights_pattern': 'results/weights/dqn_enhanced_10000ep_*.pt',
        'network': 'dqn',
        'feature_mode': 'enhanced',
        'duration': 45,
        'show_features': True,
        'show_q_values': True,
        'heatmap': 'combined',
        'description': 'Enhanced features (24-dim) for best avoidance'
    },

    '08_selective': {
        'name': 'DQN Selective',
        'agent_type': 'trained',
        'weights_pattern': 'results/weights/dqn_selective_10000ep_*.pt',
        'network': 'dqn',
        'feature_mode': 'selective',
        'duration': 45,
        'show_features': True,
        'show_q_values': True,
        'heatmap': 'combined',
        'description': 'Selective features (19-dim) - optimal efficiency'
    }
}

# Fallback weight patterns for when 10000ep weights are not available
FALLBACK_WEIGHTS = {
    'basic': [
        'results/weights/dqn_basic_*5000ep*.pt',
        'results/weights/dqn_basic_*1000ep*.pt',
        'results/weights/dqn_mlp_128x128_*ep*.pt'
    ],
    'flood-fill': [
        'results/weights/dqn_flood-fill_*5000ep*.pt',
        'results/weights/dqn_flood-fill_*1000ep*.pt',
        'results/weights/dqn_mlp_floodfill_*ep*.pt'
    ],
    'selective': [
        'results/weights/dqn_selective_*5000ep*.pt',
        'results/weights/dqn_selective_*1000ep*.pt',
        'results/weights/dqn_mlp_selective_*ep*.pt'
    ],
    'enhanced': [
        'results/weights/dqn_enhanced_*5000ep*.pt',
        'results/weights/dqn_enhanced_*1000ep*.pt',
        'results/weights/dqn_mlp_enhanced_*ep*.pt'
    ]
}


def get_preset(name: str) -> Dict[str, Any]:
    """Get preset configuration by name."""
    if name in PRESETS:
        return PRESETS[name].copy()

    # Try partial match
    for key, preset in PRESETS.items():
        if name in key or name in preset.get('name', '').lower():
            return preset.copy()

    raise KeyError(f"Unknown preset: {name}")


def list_presets() -> None:
    """Print all available presets."""
    print("\nAvailable Video Presets:")
    print("-" * 60)
    for key, preset in PRESETS.items():
        name = preset['name']
        desc = preset.get('description', '')
        print(f"  {key}: {name}")
        print(f"      {desc}")
    print()
