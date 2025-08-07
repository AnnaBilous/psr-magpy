"""Base classes for I/O operations with validation, recovery, and progress tracking."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union, Optional
from contextlib import contextmanager
import logging
import numpy as np

from .validator import (
    DataIOValidator,
    ConfigIOValidator,
    ValidationLevel,
    RecoveryStrategy,
)
from .memory_monitor import MemoryMonitor, MemoryThreshold
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class IOUtils:
    """Common I/O utilities."""

    # FITS type mapping for format detection
    FITS_TYPE_MAPPING = {
        float: "D",
        int: "I",
        str: "10A",
        list: "PE",  # Variable-length array of doubles
        np.ndarray: "D",
        np.integer: "I",  # numpy integers
        np.floating: "D",  # numpy floats
    }

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists for the given path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def convert_for_yaml(obj: Any) -> Any:
        """Recursively convert objects for YAML compatibility."""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: IOUtils.convert_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [IOUtils.convert_for_yaml(item) for item in obj]
        else:
            return obj

    @staticmethod
    def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
        """Validate file path."""
        path = Path(path)

        if must_exist and not path.exists():
            error_msg = f"File does not exist: {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        return path

    @staticmethod
    def get_fits_format(value: Any) -> str:
        """Determine FITS format for a value."""
        logger.debug(
            f"Determining FITS format for value of type {type(value).__name__}"
        )

        # Handle None values
        if value is None:
            logger.debug("Value is None, using default format 'D'")
            return "D"

        # Handle arrays with shape
        if isinstance(value, np.ndarray) and value.size > 0:
            shape = value.shape
            if len(shape) > 1:
                format_str = f"{shape[1]}D"
                logger.debug(f"Array with shape {shape}, using format '{format_str}'")
                return format_str

        # Use type mapping
        for type_class, format_str in IOUtils.FITS_TYPE_MAPPING.items():
            if isinstance(value, type_class):
                logger.debug(
                    f"Found matching type {type_class.__name__}, using format '{format_str}'"
                )
                return format_str

        # Default case
        logger.warning(
            f"No format mapping found for type {type(value).__name__}, using default"
        )
        return "D"

    @staticmethod
    def safe_file_operation(operation_name: str, file_path: Union[str, Path]):
        """Context manager for safe file operations with error handling."""

        @contextmanager
        def _safe_operation():
            file_path_obj = Path(file_path)
            logger.debug(f"Starting {operation_name} for {file_path_obj}")

            try:
                yield file_path_obj
                logger.debug(
                    f"Successfully completed {operation_name} for {file_path_obj}"
                )
            except Exception as e:
                error_msg = f"Failed to {operation_name} {file_path_obj}: {e}"
                logger.error(error_msg)
                logger.debug(f"{operation_name} error details", exc_info=True)
                raise

        return _safe_operation()

    @staticmethod
    def get_file_size_mb(path: Union[str, Path]) -> float:
        """Get file size in megabytes."""
        path = Path(path)
        if not path.exists():
            return 0.0
        return path.stat().st_size / (1024 * 1024)

    @staticmethod
    def check_disk_space(path: Union[str, Path], required_mb: float = 100.0) -> bool:
        """Check if sufficient disk space is available."""
        import shutil

        path = Path(path)
        total, used, free = shutil.disk_usage(path.parent)
        free_mb = free / (1024 * 1024)
        logger.debug(
            f"Available disk space: {free_mb:.1f} MB, required: {required_mb:.1f} MB"
        )
        return free_mb >= required_mb

    @staticmethod
    def backup_file(path: Union[str, Path], backup_suffix: str = ".backup") -> Path:
        """Create a backup of the file if it exists."""
        path = Path(path)
        if path.exists():
            backup_path = path.with_suffix(path.suffix + backup_suffix)
            import shutil

            shutil.copy2(path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        return path


class DataIO(ABC):
    """Abstract base class for data I/O operations."""

    def __init__(
        self,
        validator: Optional[DataIOValidator] = None,
        memory_monitor: Optional[MemoryMonitor] = None,
        progress_tracker=None,
    ):
        """Initialize DataIO with optional validator, memory monitor, and progress tracker."""
        self.validator = validator or DataIOValidator()
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.progress_tracker = progress_tracker

        logger.debug(
            f"Initialized {self.__class__.__name__} with validator and memory monitor"
        )

    @abstractmethod
    def save(self, data: Any, path: Union[str, Path], **kwargs) -> None:
        """Save data to file."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path], **kwargs) -> Any:
        """Load data from file."""
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data before saving."""
        pass

    def estimate_size(self, data: Any) -> int:
        """Estimate the size of data for progress tracking."""
        if hasattr(data, "__len__"):
            return len(data)
        elif hasattr(data, "n_field_lines"):
            return data.n_field_lines
        elif hasattr(data, "nline"):
            return data.nline
        else:
            return 1  # Default to 1 for unknown data types

    def validate_with_recovery(
        self, data: Any, path: Union[str, Path], operation: str = "save", **kwargs
    ) -> bool:
        """Validate data and path with recovery options."""
        # Validate data
        data_result = self.validator.validate_data(data, operation=operation, **kwargs)
        if not data_result.is_valid:
            logger.error(f"Data validation failed: {data_result.errors}")
            return False

        # Validate path
        path_result = self.validator.validate_path(path, operation, **kwargs)
        if not path_result.is_valid:
            logger.error(f"Path validation failed: {path_result.errors}")

            # Attempt recovery
            recovery_result = self.validator.attempt_recovery(
                operation, path, data, **kwargs
            )
            if recovery_result.success:
                logger.info(
                    f"Recovery successful using {recovery_result.strategy_used.value}"
                )
                return True
            else:
                logger.error(f"Recovery failed: {recovery_result.error_message}")
                return False

        # Memory safety check
        data_size_mb = self.memory_monitor.estimate_data_size(data)
        memory_result = self.memory_monitor.check_memory_safety(
            data_size_mb, operation, **kwargs
        )

        if not memory_result.is_safe:
            logger.error(f"Memory safety check failed: {memory_result.warnings}")
            return False

        # Log warnings
        if data_result.warnings:
            for warning in data_result.warnings:
                logger.warning(f"Data validation warning: {warning}")

        if path_result.warnings:
            for warning in path_result.warnings:
                logger.warning(f"Path validation warning: {warning}")

        if memory_result.warnings:
            for warning in memory_result.warnings:
                logger.warning(f"Memory warning: {warning}")

        return True

    def _track_io_progress(
        self, operation: str, path: Union[str, Path], data: Any = None
    ):
        """Context manager for tracking I/O progress."""
        if self.progress_tracker is not None:
            data_size = self.estimate_size(data) if data is not None else None
            return self.progress_tracker.track_io_operation(operation, path, data_size)
        else:
            # Return a dummy context manager if no progress tracker
            @contextmanager
            def dummy_context():
                yield None

            return dummy_context()


class ConfigIO(ABC):
    """Abstract base class for configuration I/O operations."""

    def __init__(
        self,
        validator: Optional[ConfigIOValidator] = None,
        memory_monitor: Optional[MemoryMonitor] = None,
        progress_tracker=None,
    ):
        """Initialize ConfigIO with optional validator, memory monitor, and progress tracker."""
        self.validator = validator or ConfigIOValidator()
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.progress_tracker = progress_tracker

        logger.debug(
            f"Initialized {self.__class__.__name__} with validator and memory monitor"
        )

    @abstractmethod
    def save(self, config: Any, path: Union[str, Path], **kwargs) -> None:
        """Save configuration to file."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path], **kwargs) -> Any:
        """Load configuration from file."""
        pass

    @abstractmethod
    def validate(self, config: Any) -> bool:
        """Validate configuration before saving."""
        pass

    def print_config(self, config: Any) -> None:
        """Print configuration to screen."""
        print("=== Configuration ===")
        print(f"Type: {type(config).__name__}")
        print(f"Config: {config}")

    def validate_with_recovery(
        self, config: Any, path: Union[str, Path], operation: str = "save", **kwargs
    ) -> bool:
        """Validate configuration and path with recovery options."""
        # Validate config (skip for load operations since we don't have config yet)
        if operation != "load":
            config_result = self.validator.validate_data(config, **kwargs)
            if not config_result.is_valid:
                logger.error(f"Configuration validation failed: {config_result.errors}")
                return False

        # Validate path
        path_result = self.validator.validate_path(path, operation, **kwargs)
        if not path_result.is_valid:
            logger.error(f"Path validation failed: {path_result.errors}")

            # Attempt recovery
            recovery_result = self.validator.attempt_recovery(
                operation, path, config, **kwargs
            )
            if recovery_result.success:
                logger.info(
                    f"Recovery successful using {recovery_result.strategy_used.value}"
                )
                return True
            else:
                logger.error(f"Recovery failed: {recovery_result.error_message}")
                return False

        # Memory safety check (config files are typically small)
        memory_result = self.memory_monitor.check_memory_safety(
            1.0, operation, **kwargs
        )  # 1MB default for configs

        if not memory_result.is_safe:
            logger.error(f"Memory safety check failed: {memory_result.warnings}")
            return False

        # Log warnings
        if operation != "load" and config_result.warnings:
            for warning in config_result.warnings:
                logger.warning(f"Configuration validation warning: {warning}")

        if path_result.warnings:
            for warning in path_result.warnings:
                logger.warning(f"Path validation warning: {warning}")

        if memory_result.warnings:
            for warning in memory_result.warnings:
                logger.warning(f"Memory warning: {warning}")

        return True

    def _track_io_progress(
        self, operation: str, path: Union[str, Path], config: Any = None
    ):
        """Context manager for tracking I/O progress."""
        if self.progress_tracker is not None:
            # Config files are typically small, so we don't need detailed progress
            return self.progress_tracker.track_io_operation(operation, path, None)
        else:
            # Return a dummy context manager if no progress tracker
            @contextmanager
            def dummy_context():
                yield None

            return dummy_context()
