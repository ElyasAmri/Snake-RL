"""
Resume PPO-MLP Curriculum Training from Stage 3 Checkpoint

This script loads the Stage 3 checkpoint and trains only Stage 4 (Co-Evolution).
One-time use script to complete the curriculum after Stage 4 stall.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import time

from train_ppo_two_snake_mlp import (
    PPOConfig, TwoSnakePPOAgent, PPOBuffer
)
from train_ppo_two_snake_mlp_curriculum import (
    CurriculumStage, CurriculumPPOTrainer
)
from core.utils import set_seed, get_device


def main():
    print("\n" + "="*70)
    print("RESUME PPO-MLP CURRICULUM FROM STAGE 3")
    print("="*70)
    print("This will load Stage 3 checkpoint and train Stage 4 only")
    print("="*70 + "\n")

    # Configuration (same as original training)
    # Using defaults except for num_envs and rollout_steps
    config = PPOConfig(
        num_envs=128,
        rollout_steps=2048
    )

    # Set seed and device
    set_seed(42)
    device = get_device()

    # Create trainer
    print("Initializing curriculum trainer...")
    trainer = CurriculumPPOTrainer(config)

    # Load Stage 3 checkpoint
    print("\nLoading Stage 3 checkpoint...")
    checkpoint_dir = Path("results/weights/ppo_two_snake_mlp_curriculum_fixed")

    # Agent1 checkpoint
    agent1_path = checkpoint_dir / "big_256x256_Stage3_Frozen_20251124_170515.pt"
    print(f"Loading Agent1 from: {agent1_path}")
    checkpoint1 = torch.load(agent1_path, map_location=device)
    trainer.agent1.actor.load_state_dict(checkpoint1['actor'])
    trainer.agent1.critic.load_state_dict(checkpoint1['critic'])
    trainer.agent1.actor_optimizer.load_state_dict(checkpoint1['actor_optimizer'])
    trainer.agent1.critic_optimizer.load_state_dict(checkpoint1['critic_optimizer'])

    # Agent2 checkpoint
    agent2_path = checkpoint_dir / "small_128x128_Stage3_Frozen_20251124_170515.pt"
    print(f"Loading Agent2 from: {agent2_path}")
    checkpoint2 = torch.load(agent2_path, map_location=device)
    trainer.agent2.actor.load_state_dict(checkpoint2['actor'])
    trainer.agent2.critic.load_state_dict(checkpoint2['critic'])
    trainer.agent2.actor_optimizer.load_state_dict(checkpoint2['actor_optimizer'])
    trainer.agent2.critic_optimizer.load_state_dict(checkpoint2['critic_optimizer'])

    print("Checkpoints loaded successfully")

    # Set initial step count (approximate - Stage 3 completed around 70K steps)
    # With fixed step counting: 5000 (Stage0) + 7500 (Stage1) + 47219 (Stage2) + 10000 (Stage3) = ~70K
    trainer.total_steps = 70000
    trainer.stage_steps = 0

    # Define Stage 4 only
    stage4 = CurriculumStage(
        stage_id=4,
        name="Stage4_CoEvolution",
        opponent_type="learning",
        target_food=8,
        min_steps=20000,
        win_rate_threshold=None,  # No threshold - runs for full duration
        description="Full co-evolution training (target: 8 food)",
        agent2_trains=True  # Both agents train
    )

    print(f"\n" + "="*70)
    print(f"STAGE 4: {stage4.name}")
    print("="*70)
    print(f"Opponent: {stage4.opponent_type}")
    print(f"Target food: {stage4.target_food}")
    print(f"Min steps: {stage4.min_steps:,}")
    print(f"Win rate threshold: {stage4.win_rate_threshold}")
    print(f"Description: {stage4.description}")
    print(f"Agent2 trains: {stage4.agent2_trains}")
    print(f"Starting from total_steps: {trainer.total_steps:,}")
    print("="*70 + "\n")

    # Train Stage 4
    start_time = time.time()
    trainer.train_curriculum_stage(stage4)
    elapsed = time.time() - start_time

    print(f"\n" + "="*70)
    print("STAGE 4 COMPLETE")
    print("="*70)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Steps: {trainer.stage_steps:,}")
    print(f"Total steps: {trainer.total_steps:,}")
    print("="*70)

    # Save final checkpoint
    print("\nSaving final Stage 4 checkpoint...")
    trainer.save_checkpoint("Stage4_CoEvolution")

    print("\n" + "="*70)
    print("CURRICULUM TRAINING COMPLETED")
    print("="*70)
    print("All 5 stages finished successfully!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
