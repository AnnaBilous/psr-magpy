"""IO utilities for psr_magpy."""

from typing import Optional, Dict, Any, Union
from pathlib import Path

from psr_magpy.utils.logging_config import get_logger

# pylint: disable=logging-fstring-interpolation
logger = get_logger(__name__, console_format_brief=True)


def get_io_factory():
    """Get IO factory instance."""
    from psr_magpy.io.factory import IOFactory

    return IOFactory()


def get_config_io():
    """Get config IO instance."""
    from psr_magpy.io.config_io import ConfigIO

    return ConfigIO()


def get_data_io():
    """Get data IO instance."""
    from psr_magpy.io.data_io import DataIO

    return DataIO()


__all__ = ["get_io_factory", "get_config_io", "get_data_io"]
