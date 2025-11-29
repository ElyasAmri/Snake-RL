"""
Cross-Device Reproducibility Verification Script

Runs a short training sequence and captures deterministic outputs for comparison.
Run on different devices (AMD GPU, NVIDIA GPU, CPU) and compare JSON outputs.

Usage:
    # Generate report on current device
    python scripts/verify_reproducibility.py --seed 67 --output results.json

    # Compare two reports
    python scripts/verify_reproducibility.py --compare results_nvidia.json results_amd.json
"""

import argparse
import json
import hashlib
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import set_seed
from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DQN_MLP


def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """Compute SHA256 hash of tensor data for comparison"""
    data = tensor.detach().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def get_device_info() -> dict:
    """Gather device and environment metadata"""
    info = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    return info


def run_verification(
    seed: int,
    device_str: str = "cuda",
    num_steps: int = 100,
    num_envs: int = 8,
    strict_determinism: bool = False
) -> dict:
    """
    Run deterministic verification sequence

    Args:
        seed: Random seed
        device_str: Device to use ('cuda' or 'cpu')
        num_steps: Number of environment steps
        num_envs: Number of parallel environments
        strict_determinism: Use strict deterministic mode

    Returns:
        Dictionary with hashes and values for comparison
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Set seed with optional strict determinism
    set_seed(seed, strict_determinism=strict_determinism)

    results = {
        "metadata": {
            "seed": seed,
            "device": str(device),
            "num_steps": num_steps,
            "num_envs": num_envs,
            "strict_determinism": strict_determinism,
            "timestamp": datetime.now().isoformat(),
            "device_info": get_device_info()
        },
        "random_generation": {},
        "network": {},
        "environment": {},
        "trajectory": {}
    }

    # 1. Test random number generation
    set_seed(seed, strict_determinism=strict_determinism)

    # CPU random (should be identical across devices)
    cpu_rand = torch.rand(100, device="cpu")
    results["random_generation"]["cpu_rand_hash"] = compute_tensor_hash(cpu_rand)
    results["random_generation"]["cpu_rand_sum"] = float(cpu_rand.sum())

    cpu_randint = torch.randint(0, 100, (100,), device="cpu")
    results["random_generation"]["cpu_randint_hash"] = compute_tensor_hash(cpu_randint)
    results["random_generation"]["cpu_randint_sum"] = int(cpu_randint.sum())

    # GPU random (may differ across vendors)
    if device.type == "cuda":
        set_seed(seed, strict_determinism=strict_determinism)
        gpu_rand = torch.rand(100, device=device)
        results["random_generation"]["gpu_rand_hash"] = compute_tensor_hash(gpu_rand)
        results["random_generation"]["gpu_rand_sum"] = float(gpu_rand.sum())

        gpu_randint = torch.randint(0, 100, (100,), device=device)
        results["random_generation"]["gpu_randint_hash"] = compute_tensor_hash(gpu_randint)
        results["random_generation"]["gpu_randint_sum"] = int(gpu_randint.sum())

    # 2. Test network initialization
    set_seed(seed, strict_determinism=strict_determinism)
    model = DQN_MLP(input_dim=11, output_dim=3, hidden_dims=(128, 128)).to(device)

    weight_hashes = {}
    weight_sums = {}
    for name, param in model.named_parameters():
        weight_hashes[name] = compute_tensor_hash(param.data)
        weight_sums[name] = float(param.data.sum())

    results["network"]["weight_hashes"] = weight_hashes
    results["network"]["weight_sums"] = weight_sums

    # Sample forward pass
    set_seed(seed, strict_determinism=strict_determinism)
    dummy_input = torch.rand(32, 11, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    results["network"]["forward_input_hash"] = compute_tensor_hash(dummy_input)
    results["network"]["forward_output_hash"] = compute_tensor_hash(output)
    results["network"]["forward_output_sum"] = float(output.sum())

    # 3. Test environment
    set_seed(seed, strict_determinism=strict_determinism)
    env = VectorizedSnakeEnv(
        num_envs=num_envs,
        grid_size=10,
        action_space_type='relative',
        state_representation='feature',
        device=device
    )

    obs = env.reset(seed=seed)
    results["environment"]["initial_obs_hash"] = compute_tensor_hash(obs)
    results["environment"]["initial_obs_sum"] = float(obs.sum())

    # 4. Run trajectory
    trajectory_rewards = []
    trajectory_obs_hashes = []
    trajectory_dones = []

    for step in range(num_steps):
        # Deterministic action selection (argmax Q-values)
        with torch.no_grad():
            q_values = model(obs)
            actions = q_values.argmax(dim=1)

        obs, rewards, dones, info = env.step(actions)

        trajectory_rewards.append(rewards.cpu().tolist())
        trajectory_dones.append(dones.cpu().tolist())

        # Sample observations at intervals
        if step % 10 == 0:
            trajectory_obs_hashes.append({
                "step": step,
                "hash": compute_tensor_hash(obs),
                "sum": float(obs.sum())
            })

    results["trajectory"]["rewards"] = trajectory_rewards
    results["trajectory"]["dones"] = trajectory_dones
    results["trajectory"]["obs_samples"] = trajectory_obs_hashes
    results["trajectory"]["final_obs_hash"] = compute_tensor_hash(obs)
    results["trajectory"]["total_reward"] = sum(sum(r) for r in trajectory_rewards)

    # 5. Test gradient computation
    set_seed(seed, strict_determinism=strict_determinism)
    dummy_obs = torch.rand(32, 11, device=device)
    dummy_target = torch.rand(32, 3, device=device)

    model.zero_grad()
    output = model(dummy_obs)
    loss = torch.nn.functional.mse_loss(output, dummy_target)
    loss.backward()

    grad_hashes = {}
    grad_sums = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_hashes[name] = compute_tensor_hash(param.grad)
            grad_sums[name] = float(param.grad.sum())

    results["network"]["gradient_hashes"] = grad_hashes
    results["network"]["gradient_sums"] = grad_sums
    results["network"]["loss_value"] = float(loss.item())

    return results


def compare_reports(report1_path: str, report2_path: str) -> dict:
    """
    Compare two reproducibility reports

    Args:
        report1_path: Path to first report JSON
        report2_path: Path to second report JSON

    Returns:
        Comparison results
    """
    with open(report1_path) as f:
        report1 = json.load(f)
    with open(report2_path) as f:
        report2 = json.load(f)

    comparison = {
        "report1": report1_path,
        "report2": report2_path,
        "metadata_diff": {},
        "matches": {},
        "differences": {},
        "summary": {}
    }

    # Compare metadata
    m1 = report1["metadata"]
    m2 = report2["metadata"]
    comparison["metadata_diff"] = {
        "seed_match": m1["seed"] == m2["seed"],
        "device1": m1["device"],
        "device2": m2["device"],
        "gpu1": m1["device_info"].get("gpu_name", "N/A"),
        "gpu2": m2["device_info"].get("gpu_name", "N/A")
    }

    # Compare sections
    sections = ["random_generation", "network", "environment", "trajectory"]
    total_checks = 0
    matches = 0

    for section in sections:
        s1 = report1.get(section, {})
        s2 = report2.get(section, {})

        comparison["matches"][section] = {}
        comparison["differences"][section] = {}

        def compare_values(key, v1, v2):
            nonlocal total_checks, matches
            total_checks += 1

            if isinstance(v1, str) and isinstance(v2, str):
                # Hash comparison (exact match)
                match = v1 == v2
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numeric comparison with tolerance
                if isinstance(v1, int) and isinstance(v2, int):
                    match = v1 == v2
                else:
                    match = abs(v1 - v2) < 1e-5
            elif isinstance(v1, list) and isinstance(v2, list):
                # List comparison
                if len(v1) != len(v2):
                    match = False
                else:
                    match = all(
                        abs(a - b) < 1e-5 if isinstance(a, (int, float)) else a == b
                        for a, b in zip(
                            [x for sublist in v1 for x in (sublist if isinstance(sublist, list) else [sublist])],
                            [x for sublist in v2 for x in (sublist if isinstance(sublist, list) else [sublist])]
                        )
                    )
            else:
                match = v1 == v2

            if match:
                matches += 1
                comparison["matches"][section][key] = True
            else:
                comparison["differences"][section][key] = {
                    "value1": v1 if not isinstance(v1, list) else f"list[{len(v1)}]",
                    "value2": v2 if not isinstance(v2, list) else f"list[{len(v2)}]"
                }

        # Compare all keys
        all_keys = set(s1.keys()) | set(s2.keys())
        for key in all_keys:
            if key in s1 and key in s2:
                v1, v2 = s1[key], s2[key]
                if isinstance(v1, dict) and isinstance(v2, dict):
                    for subkey in set(v1.keys()) | set(v2.keys()):
                        if subkey in v1 and subkey in v2:
                            compare_values(f"{key}.{subkey}", v1[subkey], v2[subkey])
                else:
                    compare_values(key, v1, v2)

    comparison["summary"] = {
        "total_checks": total_checks,
        "matches": matches,
        "differences": total_checks - matches,
        "match_percentage": round(100 * matches / total_checks, 2) if total_checks > 0 else 0
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Verify cross-device reproducibility")
    parser.add_argument("--seed", type=int, default=67, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of steps")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of environments")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict determinism mode")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"),
                        help="Compare two report files")
    args = parser.parse_args()

    if args.compare:
        # Compare mode
        print(f"Comparing {args.compare[0]} vs {args.compare[1]}...")
        comparison = compare_reports(args.compare[0], args.compare[1])

        print("\n" + "=" * 60)
        print("REPRODUCIBILITY COMPARISON REPORT")
        print("=" * 60)

        print(f"\nDevices compared:")
        print(f"  Report 1: {comparison['metadata_diff']['device1']} "
              f"({comparison['metadata_diff']['gpu1']})")
        print(f"  Report 2: {comparison['metadata_diff']['device2']} "
              f"({comparison['metadata_diff']['gpu2']})")
        print(f"  Seeds match: {comparison['metadata_diff']['seed_match']}")

        print(f"\nSummary:")
        print(f"  Total checks: {comparison['summary']['total_checks']}")
        print(f"  Matches: {comparison['summary']['matches']}")
        print(f"  Differences: {comparison['summary']['differences']}")
        print(f"  Match rate: {comparison['summary']['match_percentage']}%")

        if comparison['summary']['differences'] > 0:
            print("\nDifferences found:")
            for section, diffs in comparison['differences'].items():
                if diffs:
                    print(f"\n  [{section}]")
                    for key, diff in diffs.items():
                        print(f"    {key}:")
                        print(f"      Report 1: {diff['value1']}")
                        print(f"      Report 2: {diff['value2']}")

        # Save comparison report
        comparison_output = args.output or "comparison_report.json"
        with open(comparison_output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {comparison_output}")

    else:
        # Generate mode
        print(f"Running reproducibility verification...")
        print(f"  Seed: {args.seed}")
        print(f"  Device: {args.device}")
        print(f"  Steps: {args.num_steps}")
        print(f"  Environments: {args.num_envs}")
        print(f"  Strict determinism: {args.strict}")

        results = run_verification(
            seed=args.seed,
            device_str=args.device,
            num_steps=args.num_steps,
            num_envs=args.num_envs,
            strict_determinism=args.strict
        )

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            device_name = results["metadata"]["device_info"].get("gpu_name", "cpu")
            device_name = device_name.replace(" ", "_").replace("/", "_")[:20]
            output_path = f"reproducibility_{device_name}_{args.seed}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        print(f"\nKey hashes:")
        print(f"  CPU rand: {results['random_generation']['cpu_rand_hash'][:16]}...")
        print(f"  Network weights: {list(results['network']['weight_hashes'].values())[0][:16]}...")
        print(f"  Initial obs: {results['environment']['initial_obs_hash'][:16]}...")
        print(f"  Final obs: {results['trajectory']['final_obs_hash'][:16]}...")
        print(f"  Total reward: {results['trajectory']['total_reward']:.2f}")


if __name__ == "__main__":
    main()
