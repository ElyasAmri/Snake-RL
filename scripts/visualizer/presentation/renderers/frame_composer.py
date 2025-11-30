"""
Frame Composer

Handles layout management and frame composition for presentation videos.
Combines game surface, heatmap overlays, and side panels into final 1080p frames.
"""

import numpy as np
from typing import Optional, Tuple


class FrameComposer:
    """
    Composes final video frames from multiple visual components.

    Layout (1920x1080):
    +--------------------------------------+
    |  Game Area (1080x1080)  | Side Panel |
    |  +--------------------+ | (840x1080) |
    |  | Grid + Snake       | | +--------+ |
    |  | + Gradient Overlay | | |Features| |
    |  | + Numeric Values   | | | Panel  | |
    |  +--------------------+ | +--------+ |
    |  Score: 5  Episode: 3   | |Q-Values| |
    |                         | +--------+ |
    +--------------------------------------+
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        game_area_size: int = 1080
    ):
        """
        Initialize frame composer.

        Args:
            width: Total frame width (default 1920 for 16:9)
            height: Total frame height (default 1080 for Full HD)
            game_area_size: Size of square game area (default 1080)
        """
        self.width = width
        self.height = height
        self.game_area_size = game_area_size
        self.side_panel_width = width - game_area_size  # 840px

    def compose(
        self,
        game_surface: np.ndarray,
        heatmap_overlay: Optional[np.ndarray] = None,
        feature_panel: Optional[np.ndarray] = None,
        q_value_chart: Optional[np.ndarray] = None,
        overlay_alpha: float = 0.4
    ) -> np.ndarray:
        """
        Compose final frame from components.

        Args:
            game_surface: Game rendering (H, W, 3) RGB
            heatmap_overlay: Optional heatmap (H, W, 4) RGBA with alpha channel
            feature_panel: Optional feature panel (H, W, 3) RGB
            q_value_chart: Optional Q-value chart (H, W, 3) RGB
            overlay_alpha: Alpha for heatmap blending (0-1)

        Returns:
            Composed frame (height, width, 3) RGB
        """
        # Create base frame (black background)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Resize game surface if needed
        game = self._resize_to_fit(game_surface, (self.game_area_size, self.game_area_size))

        # Apply heatmap overlay if provided
        if heatmap_overlay is not None:
            heatmap = self._resize_to_fit(heatmap_overlay, (self.game_area_size, self.game_area_size))
            game = self.alpha_blend(game, heatmap, overlay_alpha)

        # Place game in frame
        frame[0:self.game_area_size, 0:self.game_area_size] = game

        # Place side panel components if provided
        if feature_panel is not None or q_value_chart is not None:
            panel_x = self.game_area_size

            if feature_panel is not None:
                # Feature panel takes top 2/3 of side panel
                panel_height = int(self.height * 0.66)
                panel = self._resize_to_fit(feature_panel, (panel_height, self.side_panel_width))
                frame[0:panel_height, panel_x:panel_x + self.side_panel_width] = panel

                if q_value_chart is not None:
                    # Q-value chart takes bottom 1/3
                    chart_height = self.height - panel_height
                    chart = self._resize_to_fit(q_value_chart, (chart_height, self.side_panel_width))
                    frame[panel_height:self.height, panel_x:panel_x + self.side_panel_width] = chart
            elif q_value_chart is not None:
                # Only Q-value chart, takes full side panel
                chart = self._resize_to_fit(q_value_chart, (self.height, self.side_panel_width))
                frame[0:self.height, panel_x:panel_x + self.side_panel_width] = chart

        return frame

    def compose_simple(
        self,
        game_surface: np.ndarray,
        heatmap_overlay: Optional[np.ndarray] = None,
        overlay_alpha: float = 0.4
    ) -> np.ndarray:
        """
        Compose frame with game only (no side panel).

        Useful for baseline videos without feature visualization.

        Args:
            game_surface: Game rendering (H, W, 3) RGB
            heatmap_overlay: Optional heatmap (H, W, 4) RGBA
            overlay_alpha: Alpha for heatmap blending

        Returns:
            Composed frame (height, width, 3) RGB - game centered
        """
        # Create base frame (black background)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Resize game surface
        game = self._resize_to_fit(game_surface, (self.game_area_size, self.game_area_size))

        # Apply heatmap overlay if provided
        if heatmap_overlay is not None:
            heatmap = self._resize_to_fit(heatmap_overlay, (self.game_area_size, self.game_area_size))
            game = self.alpha_blend(game, heatmap, overlay_alpha)

        # Center game in frame
        x_offset = (self.width - self.game_area_size) // 2
        frame[0:self.game_area_size, x_offset:x_offset + self.game_area_size] = game

        return frame

    def alpha_blend(
        self,
        base: np.ndarray,
        overlay: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Alpha blend overlay onto base image.

        Args:
            base: Base image (H, W, 3) RGB
            overlay: Overlay image (H, W, 3) or (H, W, 4) RGBA
            alpha: Global alpha multiplier (0-1)

        Returns:
            Blended image (H, W, 3) RGB
        """
        if overlay.shape[2] == 4:
            # RGBA overlay - use per-pixel alpha
            overlay_rgb = overlay[:, :, :3]
            overlay_alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0 * alpha
        else:
            # RGB overlay - use global alpha
            overlay_rgb = overlay
            overlay_alpha = alpha

        # Blend: result = base * (1 - alpha) + overlay * alpha
        base_float = base.astype(np.float32)
        overlay_float = overlay_rgb.astype(np.float32)

        blended = base_float * (1 - overlay_alpha) + overlay_float * overlay_alpha

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _resize_to_fit(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize image to fit target size.

        Args:
            image: Input image
            target_size: (height, width) tuple

        Returns:
            Resized image
        """
        import cv2

        target_h, target_w = target_size
        current_h, current_w = image.shape[:2]

        if current_h == target_h and current_w == target_w:
            return image

        # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
        if current_h > target_h or current_w > target_w:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    def create_gradient_overlay(
        self,
        values: np.ndarray,
        colormap: str = 'danger',
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> np.ndarray:
        """
        Create gradient overlay from value grid.

        Args:
            values: Grid of values (H, W) normalized 0-1
            colormap: 'danger' (red), 'safe' (green-blue), 'info' (yellow)
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization

        Returns:
            RGBA overlay image (H, W, 4)
        """
        h, w = values.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        # Normalize values
        normalized = np.clip((values - min_val) / (max_val - min_val + 1e-8), 0, 1)

        if colormap == 'danger':
            # Red gradient - higher = more danger
            overlay[:, :, 0] = (normalized * 255).astype(np.uint8)  # R
            overlay[:, :, 1] = 0  # G
            overlay[:, :, 2] = 0  # B
            overlay[:, :, 3] = (normalized * 200).astype(np.uint8)  # A

        elif colormap == 'safe':
            # Green-blue gradient - higher = safer
            overlay[:, :, 0] = 0  # R
            overlay[:, :, 1] = (normalized * 200).astype(np.uint8)  # G
            overlay[:, :, 2] = ((1 - normalized) * 100).astype(np.uint8)  # B
            overlay[:, :, 3] = (normalized * 180).astype(np.uint8)  # A

        elif colormap == 'info':
            # Yellow gradient
            overlay[:, :, 0] = (normalized * 255).astype(np.uint8)  # R
            overlay[:, :, 1] = (normalized * 200).astype(np.uint8)  # G
            overlay[:, :, 2] = 0  # B
            overlay[:, :, 3] = (normalized * 150).astype(np.uint8)  # A

        return overlay
