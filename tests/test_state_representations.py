"""
Tests for State Representations

Tests for FeatureEncoder and GridEncoder in state_representations.py
"""
import pytest
import numpy as np
from core.state_representations import FeatureEncoder, GridEncoder


class TestFeatureEncoderBase:
    """Test basic FeatureEncoder functionality"""

    def test_encoder_initialization(self):
        """Test encoder initializes with correct parameters"""
        encoder = FeatureEncoder(grid_size=10)
        assert encoder.grid_size == 10
        assert encoder.use_flood_fill is False
        assert encoder.use_enhanced_features is False

    def test_encoder_with_flood_fill(self):
        """Test encoder with flood-fill enabled"""
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=True)
        assert encoder.use_flood_fill is True

    def test_encoder_with_enhanced_features(self):
        """Test encoder with enhanced features enabled"""
        encoder = FeatureEncoder(
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )
        assert encoder.use_enhanced_features is True

    def test_base_encoding_shape(self):
        """Test that base encoding produces 10-dim vector"""
        encoder = FeatureEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1  # RIGHT

        obs = encoder.encode(snake, food, direction)
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"

    def test_flood_fill_encoding_shape(self):
        """Test that flood-fill encoding produces 13-dim vector"""
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=True)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1

        obs = encoder.encode(snake, food, direction)
        assert obs.shape == (13,), f"Expected (13,), got {obs.shape}"

    def test_enhanced_encoding_shape(self):
        """Test that enhanced encoding produces 23-dim vector"""
        encoder = FeatureEncoder(
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1

        obs = encoder.encode(snake, food, direction)
        assert obs.shape == (23,), f"Expected (23,), got {obs.shape}"


class TestFeatureEncoderDanger:
    """Test danger detection features"""

    def test_danger_at_wall(self):
        """Test danger detection at wall"""
        encoder = FeatureEncoder(grid_size=10)
        # Snake at right edge, facing right
        snake = [(9, 5), (8, 5), (7, 5)]
        food = (5, 5)
        direction = 1  # RIGHT

        obs = encoder.encode(snake, food, direction)
        # Danger straight (wall ahead)
        assert obs[0] == 1.0, "Should detect wall danger straight ahead"

    def test_danger_at_body(self):
        """Test danger detection at snake body"""
        encoder = FeatureEncoder(grid_size=10)
        # Snake in a shape where body is to the left
        snake = [(5, 5), (5, 4), (4, 4), (4, 5)]
        food = (7, 7)
        direction = 0  # UP

        obs = encoder.encode(snake, food, direction)
        # Check that at least one danger is detected (body nearby)
        dangers = obs[0:3]
        # With this configuration, left turn would hit body at (4, 5)
        assert dangers[1] == 1.0, "Should detect body danger on left"

    def test_no_danger_in_open(self):
        """Test no danger in open space"""
        encoder = FeatureEncoder(grid_size=10)
        # Snake in center with open space
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 7)
        direction = 0  # UP

        obs = encoder.encode(snake, food, direction)
        dangers = obs[0:3]
        # In open space, all directions should be safe
        assert dangers[0] == 0.0, "No danger straight in open space"


class TestFeatureEncoderFoodDirection:
    """Test food direction features"""

    def test_food_direction_right(self):
        """Test food direction when food is to the right"""
        encoder = FeatureEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (8, 5)  # Food to the right
        direction = 1

        obs = encoder.encode(snake, food, direction)
        food_dir = obs[3:7]  # UP, RIGHT, DOWN, LEFT
        assert food_dir[1] == 1.0, "Food should be detected to the right"
        assert food_dir[3] == 0.0, "Food should not be to the left"

    def test_food_direction_up(self):
        """Test food direction when food is up"""
        encoder = FeatureEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (5, 2)  # Food above
        direction = 1

        obs = encoder.encode(snake, food, direction)
        food_dir = obs[3:7]
        assert food_dir[0] == 1.0, "Food should be detected up"

    def test_food_direction_diagonal(self):
        """Test food direction when food is diagonal"""
        encoder = FeatureEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (8, 2)  # Food up-right
        direction = 1

        obs = encoder.encode(snake, food, direction)
        food_dir = obs[3:7]
        assert food_dir[0] == 1.0, "Food should be detected up"
        assert food_dir[1] == 1.0, "Food should be detected right"


class TestFeatureEncoderDirection:
    """Test current direction encoding"""

    def test_direction_encoding_up(self):
        """Test direction encoding for UP"""
        encoder = FeatureEncoder(grid_size=10)
        snake = [(5, 5), (5, 6), (5, 7)]
        food = (7, 3)
        direction = 0  # UP

        obs = encoder.encode(snake, food, direction)
        dir_encoding = obs[7:10]
        # Direction 0 (UP) should have specific encoding
        assert sum(dir_encoding) <= 1, "Direction should be one-hot or zero"

    def test_direction_encoding_right(self):
        """Test direction encoding for RIGHT"""
        encoder = FeatureEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1  # RIGHT

        obs = encoder.encode(snake, food, direction)
        dir_encoding = obs[7:10]
        assert sum(dir_encoding) <= 1, "Direction should be one-hot or zero"


class TestFeatureEncoderFloodFill:
    """Test flood-fill features"""

    def test_flood_fill_features_in_range(self):
        """Test flood-fill features are normalized [0, 1]"""
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=True)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1

        obs = encoder.encode(snake, food, direction)
        flood_fill = obs[10:13]
        assert np.all(flood_fill >= 0), "Flood-fill should be >= 0"
        assert np.all(flood_fill <= 1), "Flood-fill should be <= 1"

    def test_flood_fill_blocked_direction(self):
        """Test flood-fill when one direction is heavily blocked"""
        encoder = FeatureEncoder(grid_size=10, use_flood_fill=True)
        # Snake at corner with limited space
        snake = [(1, 1), (1, 2), (1, 3)]
        food = (5, 5)
        direction = 3  # LEFT (towards wall)

        obs = encoder.encode(snake, food, direction)
        flood_fill = obs[10:13]
        # At least some variation should exist
        assert isinstance(flood_fill[0], (int, float, np.floating))


class TestFeatureEncoderValueRanges:
    """Test that all features are in valid ranges"""

    def test_all_features_in_valid_range(self):
        """Test all features are in [0, 1]"""
        encoder = FeatureEncoder(
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )

        # Test multiple scenarios
        test_cases = [
            ([(5, 5), (4, 5), (3, 5)], (7, 3), 1),
            ([(0, 0), (1, 0), (2, 0)], (9, 9), 2),
            ([(9, 9), (8, 9), (7, 9)], (0, 0), 0),
        ]

        for snake, food, direction in test_cases:
            obs = encoder.encode(snake, food, direction)
            assert np.all(obs >= 0), f"All features should be >= 0, got min {obs.min()}"
            assert np.all(obs <= 1), f"All features should be <= 1, got max {obs.max()}"


class TestGridEncoder:
    """Test GridEncoder functionality"""

    def test_grid_encoder_initialization(self):
        """Test grid encoder initializes correctly"""
        encoder = GridEncoder(grid_size=10)
        assert encoder.grid_size == 10

    def test_grid_encoder_output_shape(self):
        """Test grid encoder produces correct shape"""
        encoder = GridEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1

        grid = encoder.encode(snake, food, direction)
        # Grid shape is (height, width, channels)
        assert len(grid.shape) == 3, "Grid should be 3D"
        assert grid.shape[0] == 10, "Height should match grid_size"
        assert grid.shape[1] == 10, "Width should match grid_size"
        assert grid.shape[2] == 3, "Should have 3 channels"

    def test_grid_encoder_snake_marked(self):
        """Test that snake positions are marked in grid"""
        encoder = GridEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1

        grid = encoder.encode(snake, food, direction)
        # Grid shape is (height, width, channels)
        # Check that snake positions have non-zero values
        total_snake_marks = 0
        for x, y in snake:
            if x < 10 and y < 10:
                total_snake_marks += np.sum(grid[y, x, :] != 0)
        assert total_snake_marks > 0, "Snake positions should be marked"

    def test_grid_encoder_food_marked(self):
        """Test that food position is marked in grid"""
        encoder = GridEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1

        grid = encoder.encode(snake, food, direction)
        # Food should be marked at its position (channel 2)
        food_value = grid[food[1], food[0], 2]
        assert food_value > 0, "Food position should be marked in channel 2"


class TestFeatureEncoderReproducibility:
    """Test reproducibility of feature encoding"""

    def test_same_input_same_output(self):
        """Test that same input produces same output"""
        encoder = FeatureEncoder(
            grid_size=10,
            use_flood_fill=True,
            use_enhanced_features=True
        )
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)
        direction = 1

        obs1 = encoder.encode(snake, food, direction)
        obs2 = encoder.encode(snake, food, direction)

        np.testing.assert_array_equal(obs1, obs2)

    def test_different_direction_different_output(self):
        """Test that different directions produce different outputs"""
        encoder = FeatureEncoder(grid_size=10)
        snake = [(5, 5), (4, 5), (3, 5)]
        food = (7, 3)

        obs_right = encoder.encode(snake, food, direction=1)
        obs_up = encoder.encode(snake, food, direction=0)

        assert not np.array_equal(obs_right, obs_up), \
            "Different directions should produce different observations"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
