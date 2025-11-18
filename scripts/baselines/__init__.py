"""
Baseline Agents for Snake

Non-learning deterministic and random agents for comparison
"""

from scripts.baselines.shortest_path import ShortestPathAgent
from scripts.baselines.random_agent import RandomAgent

__all__ = ['ShortestPathAgent', 'RandomAgent']
