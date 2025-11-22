"""Root-level conftest for test fixtures.

This file makes fixtures from tests/fixtures/conftest.py available to all tests.
"""

import pytest
from pathlib import Path

# Import all fixtures from fixtures/conftest.py
from tests.fixtures.conftest import (
    setup_test_logger,
    temp_dir,
    temp_fits_file,
    rng,
    cleanup_test_directories,
)


@pytest.fixture
def test_data_dir():
    """Provide test data directory inside tests/data.
    
    This directory is cleaned up after all tests complete.
    """
    tests_dir = Path(__file__).parent
    data_dir = tests_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    yield data_dir

