"""
Feature Visualizer

Creates gradient heatmap overlays, feature panels, and Q-value charts
for presentation videos using matplotlib.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io


# Feature names for each mode
# Note: Index 3 (Danger: Back) is always 1 (body behind head), so we skip it in display
FEATURE_NAMES = {
    'basic': [
        'Danger: Straight',
        'Danger: Left',
        'Danger: Right',
        '(skip)',  # Danger: Back - always 1
        'Food: Up',
        'Food: Right',
        'Food: Down',
        'Food: Left',
        'Dir: Straight',
        'Dir: Left',
        'Dir: Right'
    ],
    'flood-fill': [
        'Danger: Straight',
        'Danger: Left',
        'Danger: Right',
        '(skip)',  # Danger: Back - always 1
        'Food: Up',
        'Food: Right',
        'Food: Down',
        'Food: Left',
        'Dir: Straight',
        'Dir: Left',
        'Dir: Right',
        'FloodFill: Straight',
        'FloodFill: Right',
        'FloodFill: Left'
    ],
    'selective': [
        'Danger: Straight',
        'Danger: Left',
        'Danger: Right',
        '(skip)',  # Danger: Back - always 1
        'Food: Up',
        'Food: Right',
        'Food: Down',
        'Food: Left',
        'Dir: Straight',
        'Dir: Left',
        'Dir: Right',
        'FloodFill: Straight',
        'FloodFill: Right',
        'FloodFill: Left',
        'Tail: Up',
        'Tail: Right',
        'Tail: Down',
        'Tail: Left',
        'Tail: Reachable'
    ],
    'enhanced': [
        'Danger: Straight',
        'Danger: Left',
        'Danger: Right',
        '(skip)',  # Danger: Back - always 1
        'Food: Up',
        'Food: Right',
        'Food: Down',
        'Food: Left',
        'Dir: Straight',
        'Dir: Left',
        'Dir: Right',
        'FloodFill: Straight',
        'FloodFill: Right',
        'FloodFill: Left',
        'Escape: Straight',
        'Escape: Right',
        'Escape: Left',
        'Tail: Up',
        'Tail: Right',
        'Tail: Down',
        'Tail: Left',
        'Tail: Reachable',
        'Tail: Distance',
        'Snake: Length Ratio'
    ]
}

# Indices to skip in feature panel display (always constant values)
SKIP_INDICES = {3}  # Danger: Back is always 1


class FeatureVisualizer:
    """
    Creates feature visualization components for presentation videos.

    Generates:
    - Gradient heatmap overlays for danger/flood-fill
    - Feature panel with values and importance bars
    - Q-value bar charts
    """

    def __init__(
        self,
        grid_size: int = 10,
        game_area_size: int = 1080
    ):
        """
        Initialize feature visualizer.

        Args:
            grid_size: Number of cells in grid
            game_area_size: Pixel size of game area
        """
        self.grid_size = grid_size
        self.game_area_size = game_area_size
        self.cell_size = game_area_size // grid_size

        # Color schemes
        self.danger_color = (255, 50, 50)  # Red
        self.safe_color = (50, 200, 100)   # Green
        self.flood_color = (50, 150, 255)  # Blue
        self.tail_color = (255, 200, 50)   # Yellow

    def create_danger_overlay(
        self,
        features: np.ndarray,
        head_pos: Tuple[int, int],
        direction: int
    ) -> np.ndarray:
        """
        Create gradient overlay showing danger in each direction.

        Args:
            features: Feature vector (danger features are indices 0-3)
            head_pos: (x, y) position of snake head
            direction: Current direction (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)

        Returns:
            RGBA overlay image (H, W, 4)
        """
        overlay = np.zeros((self.game_area_size, self.game_area_size, 4), dtype=np.uint8)

        # Danger values: straight, left, right, back (indices 0-3)
        danger_straight = features[0] if len(features) > 0 else 0
        danger_left = features[1] if len(features) > 1 else 0
        danger_right = features[2] if len(features) > 2 else 0
        danger_back = features[3] if len(features) > 3 else 0

        # Direction deltas: UP, RIGHT, DOWN, LEFT
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # Map relative directions to absolute
        straight_dir = direction
        left_dir = (direction - 1) % 4
        right_dir = (direction + 1) % 4
        back_dir = (direction + 2) % 4

        dangers = [
            (straight_dir, danger_straight),
            (left_dir, danger_left),
            (right_dir, danger_right),
            (back_dir, danger_back)
        ]

        hx, hy = head_pos
        for abs_dir, danger_val in dangers:
            if danger_val > 0.5:  # Dangerous
                dx, dy = deltas[abs_dir]
                nx, ny = hx + dx, hy + dy

                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self._fill_cell_gradient(overlay, nx, ny, self.danger_color, danger_val * 0.7)

        return overlay

    def create_flood_fill_overlay(
        self,
        features: np.ndarray,
        head_pos: Tuple[int, int],
        direction: int
    ) -> np.ndarray:
        """
        Create gradient overlay showing flood-fill values.

        Args:
            features: Feature vector (flood-fill at indices 11-13 for flood-fill mode)
            head_pos: (x, y) position of snake head
            direction: Current direction

        Returns:
            RGBA overlay image (H, W, 4)
        """
        overlay = np.zeros((self.game_area_size, self.game_area_size, 4), dtype=np.uint8)

        # Check if flood-fill features exist
        if len(features) < 14:
            return overlay

        # Flood-fill values: straight, right, left (indices 11-13)
        ff_straight = features[11]
        ff_right = features[12]
        ff_left = features[13]

        # Direction deltas
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        straight_dir = direction
        left_dir = (direction - 1) % 4
        right_dir = (direction + 1) % 4

        flood_fills = [
            (straight_dir, ff_straight),
            (right_dir, ff_right),
            (left_dir, ff_left)
        ]

        hx, hy = head_pos
        for abs_dir, ff_val in flood_fills:
            dx, dy = deltas[abs_dir]
            nx, ny = hx + dx, hy + dy

            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                # Higher flood-fill = safer (more space)
                # Use blue-green gradient
                if ff_val > 0.1:
                    color = self._interpolate_color(
                        (100, 50, 50),   # Low space - reddish
                        (50, 200, 150),  # High space - greenish
                        ff_val
                    )
                    self._fill_cell_gradient(overlay, nx, ny, color, 0.5 + ff_val * 0.3)

        return overlay

    def create_combined_overlay(
        self,
        features: np.ndarray,
        head_pos: Tuple[int, int],
        direction: int,
        feature_mode: str = 'basic'
    ) -> np.ndarray:
        """
        Create combined overlay with danger and flood-fill.

        Args:
            features: Feature vector
            head_pos: (x, y) position of snake head
            direction: Current direction
            feature_mode: Feature mode for determining which features to show

        Returns:
            RGBA overlay image (H, W, 4)
        """
        # Start with danger overlay
        overlay = self.create_danger_overlay(features, head_pos, direction)

        # Add flood-fill if available
        if feature_mode in ('flood-fill', 'floodfill', 'selective', 'enhanced'):
            ff_overlay = self.create_flood_fill_overlay(features, head_pos, direction)
            # Blend flood-fill where danger is not present
            mask = overlay[:, :, 3] < 50
            overlay[mask] = ff_overlay[mask]

        return overlay

    def _fill_cell_gradient(
        self,
        overlay: np.ndarray,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        alpha: float
    ):
        """Fill a grid cell with gradient color."""
        px = x * self.cell_size
        py = y * self.cell_size

        # Create radial gradient within cell
        for i in range(self.cell_size):
            for j in range(self.cell_size):
                # Distance from cell center (normalized)
                cx, cy = self.cell_size / 2, self.cell_size / 2
                dist = np.sqrt((i - cx)**2 + (j - cy)**2) / (self.cell_size / 2)
                dist = min(dist, 1.0)

                # Fade from center
                cell_alpha = alpha * (1 - dist * 0.5)

                if py + j < self.game_area_size and px + i < self.game_area_size:
                    overlay[py + j, px + i, 0] = color[0]
                    overlay[py + j, px + i, 1] = color[1]
                    overlay[py + j, px + i, 2] = color[2]
                    overlay[py + j, px + i, 3] = int(cell_alpha * 255)

    def _interpolate_color(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        t: float
    ) -> Tuple[int, int, int]:
        """Interpolate between two colors."""
        t = max(0, min(1, t))
        return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))

    def create_feature_panel(
        self,
        features: np.ndarray,
        feature_mode: str,
        importance: Optional[np.ndarray] = None,
        width: int = 840,
        height: int = 720
    ) -> np.ndarray:
        """
        Create feature panel showing all feature values with importance bars.

        Args:
            features: Feature vector
            feature_mode: 'basic', 'flood-fill', 'selective', 'enhanced'
            importance: Optional importance scores for each feature
            width: Panel width in pixels
            height: Panel height in pixels

        Returns:
            RGB image (H, W, 3)
        """
        # Get feature names
        all_names = FEATURE_NAMES.get(feature_mode, FEATURE_NAMES['basic'])
        n_all = min(len(features), len(all_names))

        # Filter out skipped features (like Danger: Back which is always 1)
        display_indices = [i for i in range(n_all) if i not in SKIP_INDICES]
        names = [all_names[i] for i in display_indices]
        display_features = [features[i] for i in display_indices]
        n_features = len(display_indices)

        # Create figure
        fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor='#1a1a1a')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1a1a1a')

        # Title (show original dimension count)
        ax.set_title(f'Feature Vector ({n_all}-dim)', color='white', fontsize=14, pad=10)

        # Create horizontal bar chart
        y_pos = np.arange(n_features)
        colors = []

        for idx, i in enumerate(display_indices):
            val = features[i]
            if i < 4:  # Danger features
                colors.append('#ff5050' if val > 0.5 else '#404040')
            elif i < 8:  # Food direction
                colors.append('#50ff50' if val > 0.5 else '#404040')
            elif i < 11:  # Direction
                colors.append('#ffff50' if val > 0.5 else '#404040')
            elif i < 14:  # Flood-fill
                colors.append('#5080ff' if val > 0.3 else '#404040')
            else:  # Tail/enhanced features
                colors.append('#ff8050' if val > 0.3 else '#404040')

        bars = ax.barh(y_pos, display_features, color=colors, height=0.7)

        # Add importance overlay if provided
        if importance is not None:
            display_importance = [importance[i] for i in display_indices if i < len(importance)]
            for bar, imp in zip(bars, display_importance):
                bar.set_alpha(0.4 + imp * 0.6)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, color='white', fontsize=8)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Value', color='white')

        # Add value text
        for idx, val in enumerate(display_features):
            ax.text(val + 0.02, idx, f'{val:.2f}', va='center', color='white', fontsize=7)

        # Style
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.invert_yaxis()

        fig.tight_layout()

        # Convert to numpy array
        return self._fig_to_array(fig)

    def create_q_value_chart(
        self,
        q_values: np.ndarray,
        selected_action: int,
        width: int = 840,
        height: int = 360,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create Q-value bar chart with normalized bars and actual values as labels.

        Args:
            q_values: Q-values for each action [straight, left, right]
            selected_action: Index of selected action
            width: Chart width in pixels
            height: Chart height in pixels
            normalize: If True, normalize bars to 0-1 range (softmax-style)

        Returns:
            RGB image (H, W, 3)
        """
        action_names = ['STRAIGHT', 'LEFT', 'RIGHT']

        fig = Figure(figsize=(width/100, height/100), dpi=100, facecolor='#1a1a1a')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1a1a1a')

        # Title
        ax.set_title('Q-Values / Action Probabilities', color='white', fontsize=14, pad=10)

        # Normalize Q-values using softmax for display
        if normalize:
            # Softmax normalization for consistent bar heights
            exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
            display_values = exp_q / exp_q.sum()
        else:
            display_values = q_values

        # Colors - highlight selected action
        colors = ['#505050'] * 3
        colors[selected_action] = '#50ff50'

        # Create bars with normalized heights
        x_pos = np.arange(3)
        bars = ax.bar(x_pos, display_values, color=colors, width=0.6)

        # Add actual Q-value labels on top of bars
        for i, (bar, actual_val) in enumerate(zip(bars, q_values)):
            # Position label on top of bar
            bar_height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, bar_height + 0.02,
                   f'{actual_val:.1f}', ha='center', color='white', fontsize=11, fontweight='bold')

        # Labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(action_names, color='white', fontsize=12)
        ax.set_ylabel('Probability', color='white')

        # Style
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        # Fixed y-axis limits (0 to 1 for normalized)
        ax.set_ylim(0, 1.15)  # Extra space for labels

        fig.tight_layout()

        return self._fig_to_array(fig)

    def compute_feature_importance(
        self,
        model,
        features: np.ndarray,
        device: str = 'cuda'
    ) -> np.ndarray:
        """
        Compute feature importance using gradient-based attribution.

        Args:
            model: Neural network model
            features: Feature vector
            device: Device for computation

        Returns:
            Importance scores for each feature (0-1)
        """
        import torch

        # Convert to tensor
        obs_tensor = torch.FloatTensor(features).unsqueeze(0)
        obs_tensor = obs_tensor.to(device)
        obs_tensor.requires_grad = True

        # Forward pass
        output = model(obs_tensor)

        # Get selected action
        action = output.argmax(dim=1)

        # Backward pass on selected action's Q-value
        output[0, action].backward()

        # Importance = absolute gradient, normalized
        importance = obs_tensor.grad.abs().squeeze().cpu().numpy()

        if importance.max() > 0:
            importance = importance / importance.max()

        return importance

    def _fig_to_array(self, fig: Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy RGB array."""
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Get buffer
        buf = canvas.buffer_rgba()
        width, height = canvas.get_width_height()

        # Convert to numpy
        image = np.frombuffer(buf, dtype=np.uint8)
        image = image.reshape(height, width, 4)

        # Close figure to free memory
        plt.close(fig)

        # Return RGB (drop alpha)
        return image[:, :, :3]
