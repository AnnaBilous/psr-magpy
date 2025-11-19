"""Functional dispatch for I/O operations.

Replaces factory pattern with simple type-based dispatch.
"""

from typing import Any, Type, Union
from pathlib import Path

from .data_io import FieldLineSetIO, FieldComponentIO, GridIO
from .config_io import ExperimentConfigIO, ExperimentSeriesConfigIO
from .validator import (
    ValidationLevel,
    RecoveryStrategy,
    DataIOValidator,
    ConfigIOValidator,
)
from .memory_monitor import MemoryMonitor, MemoryThreshold
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError

logger = get_logger(__name__, console_format_brief=True)


def _create_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
    **kwargs,
) -> DataIOValidator:
    """Create data IO validator."""
    return DataIOValidator(
        validation_level=validation_level,
        recovery_strategy=recovery_strategy,
        **kwargs,
    )


def _create_config_validator(
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
    **kwargs,
) -> ConfigIOValidator:
    """Create config IO validator."""
    return ConfigIOValidator(
        validation_level=validation_level,
        recovery_strategy=recovery_strategy,
        **kwargs,
    )


def _create_memory_monitor(
    memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
    safety_margin_mb: float = 100.0,
) -> MemoryMonitor:
    """Create memory monitor."""
    return MemoryMonitor(
        threshold=memory_threshold,
        safety_margin_mb=safety_margin_mb,
        enable_checks=True,
        enable_warnings=True,
    )


def _get_data_io_class(data: Any):
    """Get appropriate data IO class for given data."""
    from psr_magpy.line.field_line_set import FieldLineSet
    from psr_magpy.field.multipole import MultipoleField
    from psr_magpy.grid.grid import (
        FootpointGrid,
        SphericalGrid,
        HealpixGrid,
        RandomGrid,
        CustomGrid,
    )

    if isinstance(data, FieldLineSet):
        return FieldLineSetIO
    elif isinstance(data, MultipoleField):
        return FieldComponentIO
    elif isinstance(data, (FootpointGrid, SphericalGrid, HealpixGrid, RandomGrid, CustomGrid)):
        return GridIO
    else:
        data_type = type(data).__name__
        raise PSRMagPyIOError(f"No IO handler registered for data type: {data_type}")


def _get_config_io_class(config_type: Type):
    """Get appropriate config IO class for given config type."""
    from psr_magpy.experiment.experiment_config import ExperimentConfig
    from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig

    # Check by type name since ExperimentConfig is a dataclass and ExperimentSeriesConfig is a Pydantic model
    if config_type == ExperimentConfig or config_type.__name__ == "ExperimentConfig":
        return ExperimentConfigIO
    elif config_type == ExperimentSeriesConfig or config_type.__name__ == "ExperimentSeriesConfig":
        return ExperimentSeriesConfigIO
    else:
        raise PSRMagPyIOError(
            f"No IO handler registered for config type: {config_type.__name__}"
        )


def _create_data_io_instance(
    data: Any,
    io_class: Type,
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
    memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
    safety_margin_mb: float = 100.0,
    progress_tracker=None,
    **kwargs,
):
    """Create data IO instance with appropriate parameters."""
    validator = _create_validator(validation_level, recovery_strategy, **kwargs)
    memory_monitor = _create_memory_monitor(memory_threshold, safety_margin_mb)

    # Filter out parameters that shouldn't go to constructor
    io_kwargs = {k: v for k, v in kwargs.items() if k not in ("template", "overwrite")}

    # Handle different constructor signatures
    if io_class == FieldLineSetIO:
        return io_class(
            field_line_set=data,
            validator=validator,
            memory_monitor=memory_monitor,
            progress_tracker=progress_tracker,
            **io_kwargs,
        )
    elif io_class == FieldComponentIO:
        return io_class(
            field=data,
            validator=validator,
            memory_monitor=memory_monitor,
            progress_tracker=progress_tracker,
            **io_kwargs,
        )
    else:  # GridIO
        return io_class(
            data=data,
            validator=validator,
            memory_monitor=memory_monitor,
            progress_tracker=progress_tracker,
            **io_kwargs,
        )


def save_data(
    data: Any,
    path: Union[str, Path],
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
    memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
    safety_margin_mb: float = 100.0,
    progress_tracker=None,
    **kwargs,
) -> None:
    """Save data using type-based dispatch.
    
    Parameters
    ----------
    data : Any
        Data object to save (FieldLineSet, MultipoleField, or Grid)
    path : Union[str, Path]
        Path to save file
    validation_level : ValidationLevel, optional
        Validation level for save operation
    recovery_strategy : RecoveryStrategy, optional
        Recovery strategy for validation failures
    memory_threshold : MemoryThreshold, optional
        Memory threshold for monitoring
    safety_margin_mb : float, optional
        Safety margin in MB for memory monitoring
    progress_tracker : optional
        Progress tracker for I/O operations
    **kwargs
        Additional arguments passed to IO class
    """
    io_class = _get_data_io_class(data)
    io_instance = _create_data_io_instance(
        data,
        io_class,
        validation_level=validation_level,
        recovery_strategy=recovery_strategy,
        memory_threshold=memory_threshold,
        safety_margin_mb=safety_margin_mb,
        progress_tracker=progress_tracker,
        **kwargs,
    )
    io_instance.save(data, path, **kwargs)


def load_data(
    data_type: Type,
    path: Union[str, Path],
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
    memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
    safety_margin_mb: float = 100.0,
    progress_tracker=None,
    **kwargs,
) -> Any:
    """Load data using type-based dispatch.
    
    Parameters
    ----------
    data_type : Type
        Type of data to load (FieldLineSet, MultipoleField, or Grid class)
    path : Union[str, Path]
        Path to load file from
    validation_level : ValidationLevel, optional
        Validation level for load operation
    recovery_strategy : RecoveryStrategy, optional
        Recovery strategy for validation failures
    memory_threshold : MemoryThreshold, optional
        Memory threshold for monitoring
    safety_margin_mb : float, optional
        Safety margin in MB for memory monitoring
    progress_tracker : optional
        Progress tracker for I/O operations
    **kwargs
        Additional arguments passed to IO class
        
    Returns
    -------
    Any
        Loaded data object
    """
    # Create dummy instance to determine IO class
    from psr_magpy.line.field_line_set import FieldLineSet
    from psr_magpy.field.multipole import MultipoleField
    from psr_magpy.grid.grid import (
        FootpointGrid,
        SphericalGrid,
        HealpixGrid,
        RandomGrid,
        CustomGrid,
    )

    # Map type to IO class
    if issubclass(data_type, FieldLineSet):
        io_class = FieldLineSetIO
    elif issubclass(data_type, MultipoleField):
        io_class = FieldComponentIO
    elif issubclass(data_type, (FootpointGrid, SphericalGrid, HealpixGrid, RandomGrid, CustomGrid)):
        io_class = GridIO
    else:
        raise PSRMagPyIOError(f"No IO handler registered for data type: {data_type.__name__}")

    validator = _create_validator(validation_level, recovery_strategy, **kwargs)
    memory_monitor = _create_memory_monitor(memory_threshold, safety_margin_mb)

    # Filter out parameters that shouldn't go to constructor
    io_kwargs = {k: v for k, v in kwargs.items() if k != "template"}

    # Create IO instance
    if io_class == FieldLineSetIO:
        io_instance = io_class(
            validator=validator,
            memory_monitor=memory_monitor,
            progress_tracker=progress_tracker,
            **io_kwargs,
        )
    elif io_class == FieldComponentIO:
        io_instance = io_class(
            validator=validator,
            memory_monitor=memory_monitor,
            progress_tracker=progress_tracker,
            **io_kwargs,
        )
    else:  # GridIO
        io_instance = io_class(
            validator=validator,
            memory_monitor=memory_monitor,
            progress_tracker=progress_tracker,
            **io_kwargs,
        )

    return io_instance.load(path, **kwargs)


def save_config(
    config: Any,
    path: Union[str, Path],
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
    memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
    safety_margin_mb: float = 100.0,
    progress_tracker=None,
    **kwargs,
) -> None:
    """Save configuration using type-based dispatch.
    
    Parameters
    ----------
    config : Any
        Configuration object to save
    path : Union[str, Path]
        Path to save file
    validation_level : ValidationLevel, optional
        Validation level for save operation
    recovery_strategy : RecoveryStrategy, optional
        Recovery strategy for validation failures
    memory_threshold : MemoryThreshold, optional
        Memory threshold for monitoring
    safety_margin_mb : float, optional
        Safety margin in MB for memory monitoring
    progress_tracker : optional
        Progress tracker for I/O operations
    **kwargs
        Additional arguments passed to IO class
    """
    config_type = type(config)
    io_class = _get_config_io_class(config_type)
    
    validator = _create_config_validator(validation_level, recovery_strategy, **kwargs)
    memory_monitor = _create_memory_monitor(memory_threshold, safety_margin_mb)

    io_instance = io_class(
        validator=validator,
        memory_monitor=memory_monitor,
    )
    io_instance.save(config, path, **kwargs)


def load_config(
    config_type: Type,
    path: Union[str, Path],
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
    memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
    safety_margin_mb: float = 100.0,
    progress_tracker=None,
    **kwargs,
) -> Any:
    """Load configuration using type-based dispatch.
    
    Parameters
    ----------
    config_type : Type
        Type of configuration to load
    path : Union[str, Path]
        Path to load file from
    validation_level : ValidationLevel, optional
        Validation level for load operation
    recovery_strategy : RecoveryStrategy, optional
        Recovery strategy for validation failures
    memory_threshold : MemoryThreshold, optional
        Memory threshold for monitoring
    safety_margin_mb : float, optional
        Safety margin in MB for memory monitoring
    progress_tracker : optional
        Progress tracker for I/O operations
    **kwargs
        Additional arguments passed to IO class
        
    Returns
    -------
    Any
        Loaded configuration object
    """
    io_class = _get_config_io_class(config_type)
    
    validator = _create_config_validator(validation_level, recovery_strategy, **kwargs)
    memory_monitor = _create_memory_monitor(memory_threshold, safety_margin_mb)

    io_instance = io_class(
        validator=validator,
        memory_monitor=memory_monitor,
    )
    return io_instance.load(path, **kwargs)


def detect_format(path: Union[str, Path]) -> str:
    """Detect file format from path extension.
    
    Parameters
    ----------
    path : Union[str, Path]
        File path
        
    Returns
    -------
    str
        Detected format ('fits', 'yaml', or 'unknown')
    """
    path = Path(path)
    suffix = path.suffix.lower()

    format_map = {
        ".fits": "fits",
        ".fit": "fits",
        ".yaml": "yaml",
        ".yml": "yaml",
    }

    return format_map.get(suffix, "unknown")

