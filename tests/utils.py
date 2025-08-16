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


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_directories():
    """Clean up test directories at the end of test session."""
    yield  # Run all tests
    
    # Clean up test directories after all tests complete
    import shutil
    
    test_dirs = ["test_experiment", "test_series"]
    for dir_name in test_dirs:
        test_dir = Path(dir_name)
        if test_dir.exists():
            try:
                shutil.rmtree(test_dir)
                print(f"Cleaned up test directory: {test_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up {test_dir}: {e}")
