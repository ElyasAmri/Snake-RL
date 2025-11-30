"""
Presentation Video Recording System

Creates high-quality 1080p MP4 videos for presentations with:
- Game visualization
- Feature heatmap overlays
- Feature vector side panels
- Q-value charts
"""

from .recorder import PresentationRecorder
from .presets import PRESETS

__all__ = ['PresentationRecorder', 'PRESETS']
