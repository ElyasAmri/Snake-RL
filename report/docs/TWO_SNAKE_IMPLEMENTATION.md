# Two-Snake Competitive Environment - Implementation Guide

## Overview
This document provides a complete implementation guide for the two-snake competitive environment where:
- Snake 1 (Big): 256x256 hidden layers
- Snake 2 (Small): 128x128 hidden layers
- Win condition: First to collect N food (default 10)
- Training: Curriculum learning with 5 stages

## Files to Create

### 1. Core Environment: `core/environment_two_snake_vectorized.py`

Key features needed:
- Extend VectorizedSnakeEnv structure
- Track 2 snakes per environment (snakes1, snakes2)
- Track 2 scores (food_collected1, food_collected2)
- Single food (competitive race)
- Win condition: first_to_target_food OR last_alive
- Both lose on timeout without winner

State tensors needed:
```python
# Per environment (num_envs parallel games):
self.snakes1: (num_envs, max_length, 2)  # Snake 1 bodies
self.snakes2: (num_envs, max_length, 2)  # Snake 2 bodies
self.lengths1: (num_envs,)
self.lengths2: (num_envs,)
self.directions1: (num_envs,)  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
self.directions2: (num_envs,)
self.foods: (num_envs, 2)  # Single food per environment
self.food_counts1: (num_envs,)  # Food collected by snake 1
self.food_counts2: (num_envs,)  # Food collected by snake 2
self.alive1: (num_envs,) # Boolean
self.alive2: (num_envs,) # Boolean
self.round_winners: (num_envs,)  # 0=none, 1=snake1, 2=snake2, 3=both_lose
```

Key methods:
```python
def step(actions1, actions2):
    # 1. Update both snakes simultaneously
    # 2. Check collisions (5 types):
    #    - snake1 wall, snake1 self, snake1 vs snake2 body
    #    - snake2 wall, snake2 self
    #    - head-to-head (both die)
    # 3. Check food collection (first to reach)
    # 4. Update rewards based on:
    #    - Food collected (+10 for self, -5 for opponent)
    #    - Death (-50)
    #    - Win round (+100)
    #    - Step alive (+0.01)
    # 5. Check win condition (first to target_food)
    # 6. Auto-reset completed rounds
    return obs1, obs2, rewards1, rewards2, dones, info
```

Collision detection priority:
1. Wall collisions (both snakes)
2. Self collisions (both snakes)
3. Head-to-head collision (if heads at same cell -> both die)
4. Snake1 hits snake2 body -> snake1 dies
5. Snake2 hits snake1 body -> snake2 dies

### 2. State Representation: `core/state_representations_competitive.py`

Implement CompetitiveFeatureEncoder with 35-dim output:

**Self-awareness (14 dims):**
- Danger from walls/self in 4 directions (4)
- Food direction (up, right, down, left) (4)
- Current direction one-hot (3)
- Flood-fill free space (straight, right, left) (3)

**Opponent-awareness (15 dims):**
- Danger from opponent body in 4 directions (4)
- Opponent head position relative (up, right, down, left) (4)
- Opponent current direction (3)
- Opponent length normalized (1)
- Manhattan distance to opponent head normalized (1)
- Opponent threat level (1) = is_longer + is_closer_to_food
- Can reach opponent via flood-fill (1)

**Competitive metrics (6 dims):**
- Length difference normalized: (len_self - len_opponent) / max_length (1)
- Food count difference: (food_self - food_opponent) / target_food (1)
- Food proximity advantage: (dist_opponent_food - dist_self_food) / grid_diagonal (1)
- Space control: (flood_self - flood_opponent) / total_cells (1)
- Steps since last food normalized (1)
- Round progress: food_self / target_food (1)

Agent-centric: Each snake gets its own 35-dim observation treating itself as "self" and opponent as "opponent"

### 3. Scripted Opponents: `scripts/baselines/scripted_opponents.py`

```python
class ScriptedAgent:
    def select_action(self, obs, env, snake_id):
        pass

class StaticAgent(ScriptedAgent):
    # Always go straight (for stage 0)
    def select_action(self, obs, env, snake_id):
        return 0  # STRAIGHT

class RandomAgent(ScriptedAgent):
    # Random valid actions (for stage 1)
    def select_action(self, obs, env, snake_id):
        return torch.randint(0, 3, (env.num_envs,), device=env.device)

class GreedyFoodAgent(ScriptedAgent):
    # Always move toward food using BFS shortest path (for stage 2)
    def select_action(self, obs, env, snake_id):
        # Get snake head and food positions
        # Compute shortest path action
        # Return action tensor
        pass

class DefensiveAgent(ScriptedAgent):
    # Avoid opponent, seek food when safe (for stage 3)
    pass
```

### 4. Curriculum Training: `scripts/training/train_curriculum_two_snake.py`

Structure:
```python
from dataclasses import dataclass

@dataclass
class CurriculumStage:
    name: str
    opponent_type: str  # 'static', 'random', 'greedy', 'frozen', 'learning'
    min_steps: int
    win_rate_threshold: float  # To advance to next stage

CURRICULUM = [
    CurriculumStage("Stage0_Static", "static", 50000, 0.70),
    CurriculumStage("Stage1_Random", "random", 50000, 0.60),
    CurriculumStage("Stage2_Greedy", "greedy", 100000, 0.55),
    CurriculumStage("Stage3_Frozen", "frozen", 100000, 0.50),
    CurriculumStage("Stage4_Competitive", "learning", 500000, None),
]

class CurriculumTrainer:
    def __init__(self):
        # Big snake (256x256)
        self.agent1 = DQNTrainer(
            input_dim=35,
            hidden_dims=[256, 256],
            output_dim=3,
            ...
        )

        # Small snake (128x128)
        self.agent2 = DQNTrainer(
            input_dim=35,
            hidden_dims=[128, 128],
            output_dim=3,
            ...
        )

        self.env = VectorizedTwoSnakeEnv(num_envs=128, target_food=10)
        self.current_stage = 0

    def train(self):
        for stage_idx, stage in enumerate(CURRICULUM):
            print(f"Starting {stage.name}")
            self.current_stage = stage_idx

            while not self.should_advance_stage(stage):
                # Get observations
                obs1, obs2 = self.env.get_current_obs()

                # Select actions
                if stage.opponent_type == 'learning':
                    action1 = self.agent1.select_action(obs1)
                    action2 = self.agent2.select_action(obs2)
                else:
                    action1 = self.agent1.select_action(obs1)
                    action2 = self.get_scripted_action(stage.opponent_type, obs2)

                # Environment step
                next_obs1, next_obs2, r1, r2, dones, info = self.env.step(action1, action2)

                # Store transitions
                self.agent1.replay_buffer.push(obs1, action1, r1, next_obs1, dones)
                if stage.opponent_type == 'learning':
                    self.agent2.replay_buffer.push(obs2, action2, r2, next_obs2, dones)

                # Train
                if self.agent1.replay_buffer.size() > min_buffer_size:
                    self.agent1.train_step()
                    if stage.opponent_type == 'learning':
                        self.agent2.train_step()

                # Track metrics
                self.update_metrics(info)

            # Save checkpoint
            self.save_checkpoint(stage.name)
            print(f"Completed {stage.name}")

    def should_advance_stage(self, stage):
        # Check if min steps met AND win rate threshold met
        if self.total_steps < stage.min_steps:
            return False
        if stage.win_rate_threshold is None:  # Final stage
            return False
        return self.recent_win_rate() >= stage.win_rate_threshold

    def recent_win_rate(self):
        # Calculate win rate over last 100 rounds
        return self.wins / (self.wins + self.losses)
```

### 5. Visualizer: `scripts/visualizer/visualize_two_snake.py`

Extend single-snake visualizer:

```python
class TwoSnakeVisualizer:
    def __init__(self, env, agent1=None, agent2=None):
        self.env = env
        self.agent1 = agent1  # Big snake
        self.agent2 = agent2  # Small snake

        # Pygame setup
        self.screen = pygame.display.set_mode((width, height))

        # Colors
        self.SNAKE1_COLOR = (0, 255, 0)  # Green (big network)
        self.SNAKE1_HEAD = (0, 200, 0)
        self.SNAKE2_COLOR = (0, 100, 255)  # Blue (small network)
        self.SNAKE2_HEAD = (0, 80, 200)
        self.FOOD_COLOR = (255, 0, 0)

    def render(self, env_idx=0):
        # Draw grid
        # Draw snake 1 (green, thicker body)
        # Draw snake 2 (blue, normal body)
        # Draw food
        # Draw HUD:
        #   - Snake 1 score: X/10 (256x256)
        #   - Snake 2 score: Y/10 (128x128)
        #   - Current winner or "In progress"
        pygame.display.flip()
```

Modes:
- Random vs Random
- Trained Big vs Random
- Trained Big vs Trained Small
- Training (live curriculum visualization)

### 6. Testing: `scripts/testing/test_two_snake.py`

```python
def test_environment_initialization():
    env = VectorizedTwoSnakeEnv(num_envs=4, grid_size=10)
    obs1, obs2 = env.reset()
    assert obs1.shape == (4, 35)
    assert obs2.shape == (4, 35)

def test_collision_detection():
    # Test wall collision
    # Test self collision
    # Test opponent collision
    # Test head-to-head collision
    pass

def test_food_collection():
    # Test snake 1 eats food
    # Test snake 2 eats food
    # Test food respawn
    pass

def test_win_conditions():
    # Test first to N food
    # Test last alive
    # Test timeout (both lose)
    pass

def test_state_features():
    # Validate 35-dim features
    # Check value ranges [0, 1]
    pass

def test_performance():
    # Benchmark steps/sec
    # Should be >1000 steps/sec on GPU
    pass
```

## Implementation Order

1. **Start with environment** (`core/environment_two_snake_vectorized.py`)
   - ~600 lines
   - Most critical component
   - Test with random agents first

2. **Add state representation** (`core/state_representations_competitive.py`)
   - ~250 lines
   - Enables intelligent agents

3. **Create scripted opponents** (`scripts/baselines/scripted_opponents.py`)
   - ~200 lines
   - Needed for curriculum stages 0-2

4. **Build curriculum trainer** (`scripts/training/train_curriculum_two_snake.py`)
   - ~700 lines
   - Orchestrates training

5. **Add visualizer** (`scripts/visualizer/visualize_two_snake.py`)
   - ~400 lines
   - For debugging and demos

6. **Create tests** (`scripts/testing/test_two_snake.py`)
   - ~300 lines
   - Validation

## Estimated Timeline

- Environment: 3-4 hours
- State representation: 1-2 hours
- Scripted opponents: 1-2 hours
- Curriculum trainer: 2-3 hours
- Visualizer: 2 hours
- Testing: 1 hour

**Total: 10-14 hours** of focused development

## Expected Training Time

- Stage 0 (Static): ~5-10 minutes
- Stage 1 (Random): ~5-10 minutes
- Stage 2 (Greedy): ~10-20 minutes
- Stage 3 (Frozen): ~10-20 minutes
- Stage 4 (Competitive): ~1-2 hours

**Total: 2-3 hours** end-to-end training

## Next Steps

Once single-snake training completes (~60-90 minutes from now):

1. Check training results
2. Begin two-snake implementation
3. Test environment with random agents
4. Run curriculum training
5. Analyze big vs small network performance
6. Create visualizations and report findings

## Key Design Notes

- **Memory**: Reduce num_envs from 256 to 128 (2x snakes = 2x memory)
- **GPU**: All tensors on CUDA for speed
- **Vectorization**: Maintain parallel execution across all 128 games
- **Agent-centric**: Each snake sees world from its own perspective
- **Competitive rewards**: Zero-sum style (one snake's gain = other's relative loss)
- **Curriculum**: Progressive difficulty prevents early policy collapse
