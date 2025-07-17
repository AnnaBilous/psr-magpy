"""Utility functions for tests."""

import sys
from pathlib import Path
import logging
import tempfile
import pytest
import numpy as np

# Add project root to PYTHONPATH first
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure astropy logging
import astropy

astropy.conf.logging_level = "WARNING"

# Import project utilities
from src.psr_magpy.utils.logging_config import setup_logging, LogLevels


def setup_test_logger(test_name: str) -> logging.Logger:
    """Set up logging for tests.

    Parameters
    ----------
    test_name : str
        Name of the test module

    Returns
    -------
    logging.Logger
        Configured logger
    """
    setup_logging(
        log_level="INFO",
        handler_levels={
            "console": LogLevels.CONSOLE,
            "file": None,  # No file logging for tests
        },
    )
    return logging.getLogger(test_name)


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_fits_file():
    """Provide temporary file path for FITS testing."""
    with tempfile.NamedTemporaryFile(suffix=".fits") as f:
        yield Path(f.name)


@pytest.fixture
def rng():
    """Provide numpy random number generator with fixed seed."""
    return np.random.default_rng(42)
