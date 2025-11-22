"""Utility functions for tests."""

import sys
from pathlib import Path
import logging
import tempfile
import pytest
import numpy as np

# Add project root to PYTHONPATH first
project_root = Path(__file__).parent.parent.parent
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
def test_data_dir():
    """Provide test data directory inside tests/data.
    
    This directory is cleaned up after all tests complete.
    """
    tests_dir = Path(__file__).parent.parent
    data_dir = tests_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    yield data_dir


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
    """Clean up test directories at the end of test session.
    
    This fixture automatically cleans up:
    1. tests/data directory (all test output)
    2. Any test directories created in the project root when tests don't properly set output_dir
    """
    yield  # Run all tests
    
    # Clean up test directories after all tests complete
    import shutil
    
    # Get paths
    tests_dir = Path(__file__).parent.parent
    data_dir = tests_dir / "data"
    project_root = Path(__file__).parent.parent.parent
    
    cleaned = []
    
    # Clean up tests/data directory
    if data_dir.exists():
        try:
            shutil.rmtree(data_dir)
            cleaned.append(str(data_dir))
            print(f"Cleaned up test data directory: {data_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {data_dir}: {e}")
    
    # Patterns for test directories that might be created in project root
    # These match common test directory names when output_dir is not set
    test_dir_patterns = [
        "test_experiment*",
        "test_series*",
        "test_*_series*",
        "test_*_experiment*",
        "simple_series*",  # From test_experiment_config.py
        "integration_test*",  # From test_experiment_series.py::test_full_series_workflow
    ]
    
    # Also clean up grids directory if created in project root
    # (created by experiment_runner.py when grid_file_path is not set)
    grids_dir = project_root / "grids"
    if grids_dir.exists() and grids_dir.is_dir():
        try:
            # Only clean if it contains test files (check for test_* pattern)
            grid_files = list(grids_dir.glob("test_*.fits"))
            if grid_files:
                shutil.rmtree(grids_dir)
                cleaned.append(str(grids_dir))
                print(f"Cleaned up grids directory: {grids_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {grids_dir}: {e}")
    
    for pattern in test_dir_patterns:
        # Search in project root only (not in subdirectories)
        for test_dir in project_root.glob(pattern):
            if test_dir.is_dir() and test_dir.exists():
                try:
                    shutil.rmtree(test_dir)
                    cleaned.append(str(test_dir))
                    print(f"Cleaned up test directory: {test_dir}")
                except Exception as e:
                    print(f"Warning: Could not clean up {test_dir}: {e}")
    
    if cleaned:
        print(f"Cleaned up {len(cleaned)} test directory(ies)")
    else:
        print("No test directories to clean up")
