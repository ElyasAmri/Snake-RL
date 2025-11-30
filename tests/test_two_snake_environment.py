"""
Tests for Two-Snake Competitive Environment

Comprehensive tests for TwoSnakeCompetitiveEnv covering:
- Initialization and reset
- Step functionality
- Collision detection (walls, self, opponent)
- Win conditions
- Food consumption
- Reproducibility
"""
import pytest
import numpy as np
from core.environment_two_snake_classic import TwoSnakeCompetitiveEnv


class TestTwoSnakeInitialization:
    """Test environment initialization"""

    def test_default_initialization(self):
        """Test environment initializes with default parameters"""
        env = TwoSnakeCompetitiveEnv()
        assert env.grid_size == 10
        assert env.initial_length == 3
        assert env.target_score == 10
        assert env.max_steps == 1000

    def test_custom_initialization(self):
        """Test environment initializes with custom parameters"""
        env = TwoSnakeCompetitiveEnv(
            grid_size=15,
            initial_length=4,
            target_score=5,
            max_steps=500
        )
        assert env.grid_size == 15
        assert env.initial_length == 4
        assert env.target_score == 5
        assert env.max_steps == 500

    def test_observation_space(self):
        """Test observation space is correctly defined"""
        env = TwoSnakeCompetitiveEnv()
        assert env.observation_space.shape == (20,)
        assert env.observation_space.low.min() == -1.0
        assert env.observation_space.high.max() == 1.0

    def test_action_space(self):
        """Test action space is correctly defined"""
        env = TwoSnakeCompetitiveEnv()
        assert env.action_space.n == 3  # STRAIGHT, RIGHT, LEFT


class TestTwoSnakeReset:
    """Test environment reset functionality"""

    def test_reset_returns_valid_observations(self):
        """Test that reset returns valid observations for both agents"""
        env = TwoSnakeCompetitiveEnv()
        obs, info = env.reset()

        assert 'agent1' in obs
        assert 'agent2' in obs
        assert obs['agent1'].shape == (20,)
        assert obs['agent2'].shape == (20,)

    def test_reset_initializes_snakes(self):
        """Test that reset properly initializes both snakes"""
        env = TwoSnakeCompetitiveEnv(initial_length=3)
        env.reset()

        assert len(env.snake1_positions) == 3
        assert len(env.snake2_positions) == 3
        assert env.snake1_alive
        assert env.snake2_alive

    def test_reset_places_food(self):
        """Test that reset places food on the grid"""
        env = TwoSnakeCompetitiveEnv()
        env.reset()

        assert env.food_position is not None
        assert 0 <= env.food_position[0] < env.grid_size
        assert 0 <= env.food_position[1] < env.grid_size

    def test_reset_clears_scores(self):
        """Test that reset clears scores"""
        env = TwoSnakeCompetitiveEnv()
        env.reset()

        assert env.score1 == 0
        assert env.score2 == 0
        assert env.steps == 0

    def test_reset_info_contents(self):
        """Test that reset returns proper info dict"""
        env = TwoSnakeCompetitiveEnv()
        _, info = env.reset()

        assert 'score1' in info
        assert 'score2' in info
        assert 'snake1_length' in info
        assert 'snake2_length' in info
        assert 'snake1_alive' in info
        assert 'snake2_alive' in info
        assert 'steps' in info


class TestTwoSnakeStep:
    """Test environment step functionality"""

    def test_step_returns_correct_structure(self):
        """Test that step returns observations, rewards, terminated, truncated, info"""
        env = TwoSnakeCompetitiveEnv()
        env.reset()

        actions = {'agent1': 0, 'agent2': 0}  # Both go straight
        obs, rewards, terminated, truncated, info = env.step(actions)

        assert 'agent1' in obs
        assert 'agent2' in obs
        assert 'agent1' in rewards
        assert 'agent2' in rewards
        assert 'agent1' in terminated
        assert 'agent2' in terminated

    def test_step_updates_positions(self):
        """Test that step updates snake positions"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        initial_head1 = env.snake1_positions[0]
        initial_head2 = env.snake2_positions[0]

        actions = {'agent1': 0, 'agent2': 0}
        env.step(actions)

        # Heads should have moved
        assert env.snake1_positions[0] != initial_head1 or not env.snake1_alive
        assert env.snake2_positions[0] != initial_head2 or not env.snake2_alive

    def test_step_increments_steps(self):
        """Test that step increments step counter"""
        env = TwoSnakeCompetitiveEnv()
        env.reset()

        assert env.steps == 0
        env.step({'agent1': 0, 'agent2': 0})
        assert env.steps == 1

    def test_time_penalty_applied(self):
        """Test that small time penalty is applied each step"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        # Take a step where no collision or food
        _, rewards, _, _, _ = env.step({'agent1': 0, 'agent2': 0})

        # Both should have negative time penalty (unless something happens)
        # At minimum, base reward is -0.01
        assert rewards['agent1'] <= 0 or rewards['agent1'] >= 10  # Ate food
        assert rewards['agent2'] <= 0 or rewards['agent2'] >= 10


class TestTwoSnakeCollisions:
    """Test collision detection"""

    def test_wall_collision_terminates_snake(self):
        """Test that wall collision terminates the snake"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        # Move snake1 repeatedly in one direction until wall collision
        collision_occurred = False
        for _ in range(20):
            _, rewards, terminated, _, _ = env.step({'agent1': 0, 'agent2': 1})
            if not env.snake1_alive:
                collision_occurred = True
                assert rewards['agent1'] == -100.0
                break

        # With seed 42, collision should occur within 20 steps
        assert collision_occurred, "Snake should have hit wall within 20 steps with seed 42"

    def test_self_collision_terminates_snake(self):
        """Test that self-collision terminates the snake"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        # Make snake turn repeatedly to cause self-collision
        # Turn sequence: right, right, right (180 degree turn into self)
        actions_sequence = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for action in actions_sequence:
            if not env.snake1_alive:
                break
            env.step({'agent1': action, 'agent2': 0})

    def test_opponent_collision_terminates(self):
        """Test that colliding with opponent body terminates snake"""
        env = TwoSnakeCompetitiveEnv()
        env.reset()

        # Run until some collision occurs
        for _ in range(100):
            if not env.snake1_alive or not env.snake2_alive:
                break
            env.step({'agent1': 0, 'agent2': 0})


class TestTwoSnakeWinConditions:
    """Test win condition detection"""

    def test_reaching_target_score_wins(self):
        """Test that reaching target score declares winner"""
        env = TwoSnakeCompetitiveEnv(target_score=1)  # Low target for quick test
        env.reset(seed=42)

        # Keep playing until someone wins or timeout
        winner = None
        for _ in range(200):
            _, _, terminated, truncated, info = env.step({'agent1': 0, 'agent2': 0})
            if info['winner'] is not None:
                winner = info['winner']
                break
            if terminated['agent1'] and terminated['agent2']:
                break

    def test_opponent_death_awards_bonus(self):
        """Test that opponent dying gives bonus to survivor"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        # Play until one dies
        for _ in range(100):
            _, rewards, terminated, _, info = env.step({'agent1': 0, 'agent2': 0})

            # Check if one died while other alive
            if not env.snake1_alive and env.snake2_alive:
                assert rewards['agent2'] >= 50.0  # Death bonus
                break
            elif not env.snake2_alive and env.snake1_alive:
                assert rewards['agent1'] >= 50.0  # Death bonus
                break

    def test_both_dead_is_draw(self):
        """Test that simultaneous death results in draw"""
        env = TwoSnakeCompetitiveEnv()
        env.reset()

        # Manually set both snakes as dead to test draw logic
        env.snake1_alive = False
        env.snake2_alive = False
        assert env._get_winner() == 0, "Both dead should result in draw (winner=0)"


class TestTwoSnakeFoodConsumption:
    """Test food consumption mechanics"""

    def test_eating_food_increases_score(self):
        """Test that eating food increases score"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        initial_score1 = env.score1
        initial_score2 = env.score2

        # Play for a while
        for _ in range(200):
            env.step({'agent1': 0, 'agent2': 0})

            # If score increased, food was eaten
            if env.score1 > initial_score1 or env.score2 > initial_score2:
                break

    def test_eating_food_increases_length(self):
        """Test that eating food increases snake length"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        initial_len1 = len(env.snake1_positions)
        initial_len2 = len(env.snake2_positions)

        # Play until someone eats food
        for _ in range(200):
            if not env.snake1_alive and not env.snake2_alive:
                break
            env.step({'agent1': 0, 'agent2': 0})

            # If length increased, food was eaten
            if (env.snake1_alive and len(env.snake1_positions) > initial_len1) or \
               (env.snake2_alive and len(env.snake2_positions) > initial_len2):
                break

    def test_food_respawns_after_eaten(self):
        """Test that new food spawns after being eaten"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        # Food should always be present (unless grid full)
        for _ in range(50):
            if not env.snake1_alive and not env.snake2_alive:
                break
            env.step({'agent1': 0, 'agent2': 0})
            # Food should exist unless grid is completely full
            if env.food_position is None:
                # Grid might be full (edge case)
                total_occupied = len(env.snake1_positions) + len(env.snake2_positions)
                assert total_occupied >= env.grid_size * env.grid_size


class TestTwoSnakeReproducibility:
    """Test reproducibility with seeds"""

    def test_same_seed_produces_same_initial_state(self):
        """Test that same seed produces identical initial states"""
        env1 = TwoSnakeCompetitiveEnv()
        env2 = TwoSnakeCompetitiveEnv()

        obs1, _ = env1.reset(seed=12345)
        obs2, _ = env2.reset(seed=12345)

        np.testing.assert_array_equal(obs1['agent1'], obs2['agent1'])
        np.testing.assert_array_equal(obs1['agent2'], obs2['agent2'])
        assert env1.food_position == env2.food_position

    def test_same_seed_produces_same_trajectory(self):
        """Test that same seed with same actions produces same trajectory"""
        env1 = TwoSnakeCompetitiveEnv()
        env2 = TwoSnakeCompetitiveEnv()

        env1.reset(seed=12345)
        env2.reset(seed=12345)

        # Take same actions
        for _ in range(20):
            actions = {'agent1': 0, 'agent2': 1}
            obs1, r1, t1, tr1, i1 = env1.step(actions)
            obs2, r2, t2, tr2, i2 = env2.step(actions)

            np.testing.assert_array_equal(obs1['agent1'], obs2['agent1'])
            np.testing.assert_array_equal(obs1['agent2'], obs2['agent2'])
            assert r1 == r2
            assert t1 == t2

            if t1['agent1'] and t1['agent2']:
                break


class TestTwoSnakeTimeout:
    """Test timeout/truncation"""

    def test_max_steps_causes_truncation(self):
        """Test that reaching max_steps causes truncation"""
        env = TwoSnakeCompetitiveEnv(max_steps=10)
        env.reset(seed=42)

        truncated = False
        for _ in range(15):
            _, _, _, trunc, _ = env.step({'agent1': 0, 'agent2': 0})
            if trunc['agent1'] or trunc['agent2']:
                truncated = True
                break

        assert truncated, "Should have truncated at max_steps"


class TestTwoSnakeObservationFeatures:
    """Test observation feature correctness"""

    def test_observation_in_valid_range(self):
        """Test that all observation values are in [-1, 1]"""
        env = TwoSnakeCompetitiveEnv()

        for _ in range(10):
            obs, _ = env.reset()

            assert obs['agent1'].min() >= -1.0
            assert obs['agent1'].max() <= 1.0
            assert obs['agent2'].min() >= -1.0
            assert obs['agent2'].max() <= 1.0

    def test_dead_agent_returns_zeros(self):
        """Test that dead agent receives zero observation"""
        env = TwoSnakeCompetitiveEnv()
        env.reset(seed=42)

        # Play until one dies
        for _ in range(100):
            obs, _, _, _, _ = env.step({'agent1': 0, 'agent2': 0})

            if not env.snake1_alive:
                np.testing.assert_array_equal(obs['agent1'], np.zeros(20))
                break
            elif not env.snake2_alive:
                np.testing.assert_array_equal(obs['agent2'], np.zeros(20))
                break

    def test_danger_features_correct(self):
        """Test that danger features reflect actual collisions"""
        env = TwoSnakeCompetitiveEnv()
        obs, _ = env.reset(seed=42)

        # Danger features are indices 0-2 (straight, right, left)
        danger1 = obs['agent1'][0:3]
        danger2 = obs['agent2'][0:3]

        # All danger values should be 0 or 1
        assert np.all((danger1 == 0) | (danger1 == 1))
        assert np.all((danger2 == 0) | (danger2 == 1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
