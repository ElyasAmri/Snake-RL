"""
Pytest configuration for reproducibility tests
"""

import pytest


def pytest_addoption(parser):
    """Add command-line options for cross-device report comparison"""
    parser.addoption(
        "--report1",
        action="store",
        default=None,
        help="Path to first reproducibility report JSON"
    )
    parser.addoption(
        "--report2",
        action="store",
        default=None,
        help="Path to second reproducibility report JSON"
    )
