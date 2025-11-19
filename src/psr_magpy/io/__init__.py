"""IO utilities for psr_magpy."""

from .dispatch import (
    save_data,
    load_data,
    save_config,
    load_config,
    detect_format,
)

__all__ = [
    "save_data",
    "load_data",
    "save_config",
    "load_config",
    "detect_format",
]
