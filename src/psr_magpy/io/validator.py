"""I/O validation and error recovery system.

This module provides a comprehensive validation and error recovery system
for I/O operations, with standardized patterns and recovery strategies.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import os
import shutil
import tempfile

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class ValidationLevel(Enum):
    """Validation levels for I/O operations."""

    BASIC = "basic"  # Essential checks only
    STANDARD = "standard"  # Normal validation
    STRICT = "strict"  # Comprehensive validation


class RecoveryStrategy(Enum):
    """Recovery strategies for I/O failures."""

    FAIL_FAST = "fail_fast"  # Raise exception immediately
    RETRY_ONCE = "retry_once"  # Retry operation once
    RETRY_MULTIPLE = "retry_multiple"  # Retry multiple times
    USE_BACKUP = "use_backup"  # Try backup file
    CREATE_NEW = "create_new"  # Create new file
    SKIP = "skip"  # Skip operation


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

    def __post_init__(self):
        """Initialize with empty lists if None."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.details is None:
            self.details = {}


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""

    success: bool
    recovered_data: Optional[Any]
    strategy_used: RecoveryStrategy
    error_message: Optional[str]
    details: Dict[str, Any]


class IOValidator(ABC):
    """Base class for I/O validation and error recovery."""

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
        max_retries: int = 3,
        backup_suffix: str = ".backup",
    ):
        """Initialize validator with configuration."""
        self.validation_level = validation_level
        self.recovery_strategy = recovery_strategy
        self.max_retries = max_retries
        self.backup_suffix = backup_suffix

        logger.debug(
            f"Initialized IOValidator: level={validation_level.value}, "
            f"recovery={recovery_strategy.value}, max_retries={max_retries}"
        )

    def validate_data(self, data: Any, **kwargs) -> ValidationResult:
        """Validate data before I/O operations."""
        errors = []
        warnings = []
        details = {}

        try:
            # Basic validation
            basic_result = self._validate_basic(data, **kwargs)
            errors.extend(basic_result.errors)
            warnings.extend(basic_result.warnings)
            details.update(basic_result.details)

            # Standard validation
            if self.validation_level in [
                ValidationLevel.STANDARD,
                ValidationLevel.STRICT,
            ]:
                standard_result = self._validate_standard(data, **kwargs)
                errors.extend(standard_result.errors)
                warnings.extend(standard_result.warnings)
                details.update(standard_result.details)

            # Strict validation
            if self.validation_level == ValidationLevel.STRICT:
                strict_result = self._validate_strict(data, **kwargs)
                errors.extend(strict_result.errors)
                warnings.extend(strict_result.warnings)
                details.update(strict_result.details)

            is_valid = len(errors) == 0

            if is_valid:
                logger.debug("Data validation passed")
            else:
                logger.warning(f"Data validation failed: {len(errors)} errors")

            return ValidationResult(
                is_valid=is_valid, errors=errors, warnings=warnings, details=details
            )

        except Exception as e:
            error_msg = f"Validation failed with exception: {e}"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False, errors=[error_msg], warnings=warnings, details=details
            )

    def validate_path(
        self, path: Union[str, Path], operation: str = "unknown", **kwargs
    ) -> ValidationResult:
        """Validate file path for I/O operations."""
        errors = []
        warnings = []
        details = {}

        try:
            path_obj = Path(path)
            details["path"] = str(path_obj)
            details["operation"] = operation

            # Basic path validation
            if not path_obj.parent.exists():
                errors.append(f"Parent directory does not exist: {path_obj.parent}")

            if operation == "save":
                # Save operation validation
                if path_obj.exists() and not kwargs.get("overwrite", False):
                    errors.append(
                        f"File already exists and overwrite=False: {path_obj}"
                    )

                # Check disk space
                required_mb = kwargs.get("required_mb", 100.0)
                if not self._check_disk_space(path_obj, required_mb):
                    errors.append(f"Insufficient disk space: {required_mb} MB required")

            elif operation == "load":
                # Load operation validation
                if not path_obj.exists():
                    errors.append(f"File does not exist: {path_obj}")
                elif not path_obj.is_file():
                    errors.append(f"Path is not a file: {path_obj}")
                elif path_obj.stat().st_size == 0:
                    warnings.append(f"File is empty: {path_obj}")

            # File permissions
            if operation == "save":
                if not os.access(path_obj.parent, os.W_OK):
                    errors.append(
                        f"No write permission for directory: {path_obj.parent}"
                    )
            elif operation == "load":
                if not os.access(path_obj, os.R_OK):
                    errors.append(f"No read permission for file: {path_obj}")

            is_valid = len(errors) == 0

            if is_valid:
                logger.debug(f"Path validation passed for {operation}: {path_obj}")
            else:
                logger.warning(
                    f"Path validation failed for {operation}: {len(errors)} errors"
                )

            return ValidationResult(
                is_valid=is_valid, errors=errors, warnings=warnings, details=details
            )

        except Exception as e:
            error_msg = f"Path validation failed with exception: {e}"
            logger.error(error_msg)
            return ValidationResult(
                is_valid=False, errors=[error_msg], warnings=warnings, details=details
            )

    def attempt_recovery(
        self,
        operation: str,
        original_path: Union[str, Path],
        data: Optional[Any] = None,
        **kwargs,
    ) -> RecoveryResult:
        """Attempt to recover from I/O failure."""
        logger.info(f"Attempting recovery for {operation} operation")

        try:
            if self.recovery_strategy == RecoveryStrategy.FAIL_FAST:
                return RecoveryResult(
                    success=False,
                    recovered_data=None,
                    strategy_used=RecoveryStrategy.FAIL_FAST,
                    error_message="Fail-fast strategy: no recovery attempted",
                    details={"operation": operation},
                )

            elif self.recovery_strategy == RecoveryStrategy.RETRY_ONCE:
                return self._retry_operation(
                    operation, original_path, data, max_retries=1, **kwargs
                )

            elif self.recovery_strategy == RecoveryStrategy.RETRY_MULTIPLE:
                return self._retry_operation(
                    operation,
                    original_path,
                    data,
                    max_retries=self.max_retries,
                    **kwargs,
                )

            elif self.recovery_strategy == RecoveryStrategy.USE_BACKUP:
                return self._recover_from_backup(
                    operation, original_path, data, **kwargs
                )

            elif self.recovery_strategy == RecoveryStrategy.CREATE_NEW:
                return self._create_new_file(operation, original_path, data, **kwargs)

            elif self.recovery_strategy == RecoveryStrategy.SKIP:
                return RecoveryResult(
                    success=True,
                    recovered_data=None,
                    strategy_used=RecoveryStrategy.SKIP,
                    error_message=None,
                    details={"operation": operation, "skipped": True},
                )

            else:
                error_msg = f"Unknown recovery strategy: {self.recovery_strategy}"
                logger.error(error_msg)
                return RecoveryResult(
                    success=False,
                    recovered_data=None,
                    strategy_used=self.recovery_strategy,
                    error_message=error_msg,
                    details={"operation": operation},
                )

        except Exception as e:
            error_msg = f"Recovery attempt failed: {e}"
            logger.error(error_msg)
            return RecoveryResult(
                success=False,
                recovered_data=None,
                strategy_used=self.recovery_strategy,
                error_message=error_msg,
                details={"operation": operation},
            )

    @abstractmethod
    def _validate_basic(self, data: Any, **kwargs) -> ValidationResult:
        """Perform basic validation - must be implemented by subclasses."""
        pass

    @abstractmethod
    def _validate_standard(self, data: Any, **kwargs) -> ValidationResult:
        """Perform standard validation - must be implemented by subclasses."""
        pass

    @abstractmethod
    def _validate_strict(self, data: Any, **kwargs) -> ValidationResult:
        """Strict validation - must be implemented by subclasses."""
        pass

    def _check_disk_space(self, path: Path, required_mb: float) -> bool:
        """Check if sufficient disk space is available."""
        try:
            total, used, free = shutil.disk_usage(path.parent)
            free_mb = free / (1024 * 1024)
            logger.debug(
                f"Available disk space: {free_mb:.1f} MB, required: {required_mb:.1f} MB"
            )
            return free_mb >= required_mb
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Assume sufficient space if check fails

    def _retry_operation(
        self,
        operation: str,
        path: Union[str, Path],
        data: Optional[Any],
        max_retries: int,
        **kwargs,
    ) -> RecoveryResult:
        """Retry operation with exponential backoff."""
        path_obj = Path(path)

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Retry attempt {attempt + 1}/{max_retries} for {operation}"
                )

                # Add small delay between retries
                if attempt > 0:
                    import time

                    time.sleep(0.1 * (2**attempt))  # Exponential backoff

                # Validate before retry
                validation = self.validate_path(path_obj, operation, **kwargs)
                if not validation.is_valid:
                    logger.warning(
                        f"Validation failed on retry {attempt + 1}: {validation.errors}"
                    )
                    continue

                # Attempt operation (this would be implemented by the calling code)
                # For now, we just return success
                return RecoveryResult(
                    success=True,
                    recovered_data=data,
                    strategy_used=RecoveryStrategy.RETRY_MULTIPLE,
                    error_message=None,
                    details={"operation": operation, "attempts": attempt + 1},
                )

            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return RecoveryResult(
                        success=False,
                        recovered_data=None,
                        strategy_used=RecoveryStrategy.RETRY_MULTIPLE,
                        error_message=f"All {max_retries} retry attempts failed",
                        details={"operation": operation, "attempts": max_retries},
                    )

        return RecoveryResult(
            success=False,
            recovered_data=None,
            strategy_used=RecoveryStrategy.RETRY_MULTIPLE,
            error_message="Retry operation failed",
            details={"operation": operation},
        )

    def _recover_from_backup(
        self, operation: str, path: Union[str, Path], data: Optional[Any], **kwargs
    ) -> RecoveryResult:
        """Attempt to recover from backup file."""
        path_obj = Path(path)
        backup_path = path_obj.with_suffix(path_obj.suffix + self.backup_suffix)

        if backup_path.exists():
            logger.info(f"Attempting recovery from backup: {backup_path}")
            try:
                # Validate backup file
                validation = self.validate_path(backup_path, "load")
                if validation.is_valid:
                    return RecoveryResult(
                        success=True,
                        recovered_data=None,  # Would be loaded from backup
                        strategy_used=RecoveryStrategy.USE_BACKUP,
                        error_message=None,
                        details={
                            "operation": operation,
                            "backup_path": str(backup_path),
                        },
                    )
                else:
                    logger.warning(
                        f"Backup file validation failed: {validation.errors}"
                    )
            except Exception as e:
                logger.warning(f"Backup recovery failed: {e}")

        return RecoveryResult(
            success=False,
            recovered_data=None,
            strategy_used=RecoveryStrategy.USE_BACKUP,
            error_message="No valid backup file available",
            details={"operation": operation},
        )

    def _create_new_file(
        self, operation: str, path: Union[str, Path], data: Optional[Any], **kwargs
    ) -> RecoveryResult:
        """Create new file with unique name."""
        path_obj = Path(path)

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=path_obj.suffix,
                prefix=path_obj.stem + "_recovered_",
                dir=path_obj.parent,
                delete=False,
            ) as tmp_file:
                new_path = Path(tmp_file.name)

            logger.info(f"Created new file for recovery: {new_path}")

            return RecoveryResult(
                success=True,
                recovered_data=data,
                strategy_used=RecoveryStrategy.CREATE_NEW,
                error_message=None,
                details={"operation": operation, "new_path": str(new_path)},
            )

        except Exception as e:
            error_msg = f"Failed to create new file: {e}"
            logger.error(error_msg)
            return RecoveryResult(
                success=False,
                recovered_data=None,
                strategy_used=RecoveryStrategy.CREATE_NEW,
                error_message=error_msg,
                details={"operation": operation},
            )


class DataIOValidator(IOValidator):
    """Validator for data I/O operations."""

    def _validate_basic(self, data: Any, **kwargs) -> ValidationResult:
        """Perform basic validation for data objects."""
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # For load operations, data can be None (we don't have it yet)
        operation = kwargs.get("operation", "unknown")
        if operation == "load" and data is None:
            details["data_type"] = "None (load operation)"
            details["has_validate"] = "False"
            return ValidationResult(True, errors, warnings, details)

        # Check if data is None (for save operations)
        if data is None:
            errors.append("Data is None")
            return ValidationResult(False, errors, warnings, details)

        # Check if data has required attributes
        if hasattr(data, "validate"):
            try:
                if not data.validate():
                    errors.append("Data validation failed")
            except Exception as e:
                errors.append(f"Data validation error: {e}")

        details["data_type"] = type(data).__name__
        details["has_validate"] = str(hasattr(data, "validate"))

        return ValidationResult(len(errors) == 0, errors, warnings, details)

    def _validate_standard(self, data: Any, **kwargs) -> ValidationResult:
        """Perform standard validation for data objects."""
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # Check data size
        if hasattr(data, "estimate_size"):
            try:
                size_bytes = data.estimate_size()
                size_mb = size_bytes / (1024 * 1024)
                details["size_mb"] = size_mb

                if size_mb > 1000:  # 1 GB
                    warnings.append(f"Large data size: {size_mb:.1f} MB")
            except Exception as e:
                warnings.append(f"Could not estimate data size: {e}")

        # Check for common data issues
        if hasattr(data, "n_field_lines") and data.n_field_lines == 0:
            warnings.append("Data contains no field lines")

        return ValidationResult(True, errors, warnings, details)

    def _validate_strict(self, data: Any, **kwargs) -> ValidationResult:
        """Strict validation for data objects."""
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # Deep validation of data structure
        if hasattr(data, "_properties"):
            try:
                for prop_name, prop_value in data._properties.items():
                    if prop_value is None:
                        warnings.append(f"Property {prop_name} is None")
                    elif hasattr(prop_value, "__len__") and len(prop_value) == 0:
                        warnings.append(f"Property {prop_name} is empty")
            except Exception as e:
                warnings.append(f"Could not validate properties: {e}")

        return ValidationResult(True, errors, warnings, details)


class ConfigIOValidator(IOValidator):
    """Validator for configuration I/O operations."""

    def _validate_basic(self, config: Any, **kwargs: Any) -> ValidationResult:  # type: ignore[override]
        """Perform basic validation for configuration objects."""
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # Check if config is None
        if config is None:
            errors.append("Configuration is None")
            return ValidationResult(False, errors, warnings, details)

        # Check if config is a Pydantic model
        if hasattr(config, "model_validate"):
            try:
                config.model_validate(config.model_dump())
            except Exception as e:
                errors.append(f"Configuration validation failed: {e}")

        details["config_type"] = type(config).__name__
        details["is_pydantic"] = str(hasattr(config, "model_validate"))

        return ValidationResult(len(errors) == 0, errors, warnings, details)

    def _validate_standard(self, config: Any, **kwargs: Any) -> ValidationResult:  # type: ignore[override]
        """Perform standard validation for configuration objects."""
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # Check required fields
        if hasattr(config, "name") and not config.name:
            warnings.append("Configuration name is empty")

        # Check nested objects
        if hasattr(config, "field") and config.field is None:
            errors.append("Field configuration is None")

        if hasattr(config, "grid") and config.grid is None:
            errors.append("Grid configuration is None")

        return ValidationResult(len(errors) == 0, errors, warnings, details)

    def _validate_strict(self, config: Any, **kwargs: Any) -> ValidationResult:  # type: ignore[override]
        """Strict validation for configuration objects."""
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # Validate all nested configurations
        if hasattr(config, "field") and config.field:
            try:
                field_validator = ConfigIOValidator()
                field_result = field_validator.validate_data(config.field)
                if not field_result.is_valid:
                    errors.extend([f"Field config: {e}" for e in field_result.errors])
                warnings.extend([f"Field config: {w}" for w in field_result.warnings])
            except Exception as e:
                warnings.append(f"Could not validate field configuration: {e}")

        return ValidationResult(len(errors) == 0, errors, warnings, details)
