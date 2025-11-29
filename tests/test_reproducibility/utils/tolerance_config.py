"""
Tolerance configuration for reproducibility tests

Different operations have different tolerance requirements:
- Position/action: Must be exact (integers)
- Observations: Small floating point tolerance
- Cross-device: Relaxed tolerance due to GPU RNG differences
"""

import numpy as np
import torch


# Tolerance settings for different comparison types
TOLERANCE = {
    # Exact match required (integers)
    "position": {"atol": 0, "rtol": 0},
    "action": {"atol": 0, "rtol": 0},
    "direction": {"atol": 0, "rtol": 0},

    # Tight tolerance for same-device comparisons
    "observation": {"atol": 1e-6, "rtol": 1e-5},
    "reward": {"atol": 1e-6, "rtol": 1e-5},
    "q_values": {"atol": 1e-5, "rtol": 1e-4},
    "weights": {"atol": 1e-6, "rtol": 1e-5},
    "gradients": {"atol": 1e-5, "rtol": 1e-4},

    # Relaxed tolerance for cross-device comparisons
    # (GPU random number generators may differ between vendors)
    "cross_device_observation": {"atol": 1e-4, "rtol": 1e-3},
    "cross_device_q_values": {"atol": 1e-3, "rtol": 1e-2},
    "cross_device_weights": {"atol": 1e-4, "rtol": 1e-3},
    "cross_device_loss": {"atol": 1e-2, "rtol": 1e-1},
}


def check_within_tolerance(
    value1,
    value2,
    tolerance_type: str = "observation",
    cross_device: bool = False
) -> bool:
    """
    Check if two values are within tolerance

    Args:
        value1: First value (tensor, array, or scalar)
        value2: Second value (tensor, array, or scalar)
        tolerance_type: Type of comparison (key in TOLERANCE dict)
        cross_device: If True, use cross-device tolerances

    Returns:
        True if values are within tolerance
    """
    # Get tolerance settings
    if cross_device and f"cross_device_{tolerance_type}" in TOLERANCE:
        tol = TOLERANCE[f"cross_device_{tolerance_type}"]
    elif tolerance_type in TOLERANCE:
        tol = TOLERANCE[tolerance_type]
    else:
        tol = TOLERANCE["observation"]  # Default

    atol = tol["atol"]
    rtol = tol["rtol"]

    # Convert to numpy arrays
    if isinstance(value1, torch.Tensor):
        value1 = value1.detach().cpu().numpy()
    if isinstance(value2, torch.Tensor):
        value2 = value2.detach().cpu().numpy()

    value1 = np.asarray(value1)
    value2 = np.asarray(value2)

    # Check shapes match
    if value1.shape != value2.shape:
        return False

    # Use numpy's allclose for comparison
    return np.allclose(value1, value2, atol=atol, rtol=rtol)


def get_tolerance(tolerance_type: str, cross_device: bool = False) -> dict:
    """
    Get tolerance settings for a given type

    Args:
        tolerance_type: Type of comparison
        cross_device: If True, return cross-device tolerances

    Returns:
        Dictionary with 'atol' and 'rtol' keys
    """
    if cross_device and f"cross_device_{tolerance_type}" in TOLERANCE:
        return TOLERANCE[f"cross_device_{tolerance_type}"]
    return TOLERANCE.get(tolerance_type, TOLERANCE["observation"])
