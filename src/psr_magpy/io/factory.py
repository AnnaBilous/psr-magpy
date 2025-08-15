"""I/O factory and registry for creating appropriate I/O instances."""

from typing import Type, Dict, Any, Union, Optional
from pathlib import Path

from .base import DataIO, ConfigIO
from .data_io import FitsIO, FieldLineSetIO, FieldComponentIO, GridIO
from .config_io import YamlIO, ExperimentConfigIO, ExperimentSeriesConfigIO
from .validator import (
    ValidationLevel,
    RecoveryStrategy,
    DataIOValidator,
    ConfigIOValidator,
)
from .memory_monitor import MemoryMonitor, MemoryThreshold
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class IOFactory:
    """Factory for creating I/O instances."""

    # Registry for data types to I/O classes
    _data_registry: Dict[Type, Type[DataIO]] = {}
    _config_registry: Dict[Type, Type[ConfigIO]] = {}

    @classmethod
    def register_data_io(cls, data_type: Type, io_class: Type[DataIO]) -> None:
        """Register a data type with its I/O class."""
        cls._data_registry[data_type] = io_class
        logger.debug(f"Registered {data_type.__name__} with {io_class.__name__}")

    @classmethod
    def register_config_io(cls, config_type: Type, io_class: Type[ConfigIO]) -> None:
        """Register a config type with its I/O class."""
        cls._config_registry[config_type] = io_class
        logger.debug(f"Registered {config_type.__name__} with {io_class.__name__}")

    @classmethod
    def create_data_io(
        cls,
        data_type: Type,
        format: str = "fits",
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
        memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
        safety_margin_mb: float = 100.0,
        progress_tracker=None,
        **kwargs,
    ) -> DataIO:
        """Create a data I/O instance with validation and memory monitoring configuration."""
        if data_type not in cls._data_registry:
            error_msg = f"No I/O class registered for data type: {data_type.__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Filter out template parameter from kwargs (it's passed to load method, not constructor)
        io_kwargs = {k: v for k, v in kwargs.items() if k != 'template'}

        # Create validator with specified configuration
        validator = DataIOValidator(
            validation_level=validation_level,
            recovery_strategy=recovery_strategy,
            **io_kwargs,
        )

        # Create memory monitor with specified configuration
        memory_monitor = MemoryMonitor(
            threshold=memory_threshold,
            safety_margin_mb=safety_margin_mb,
            enable_checks=True,
            enable_warnings=True,
        )

        io_class = cls._data_registry[data_type]
        # Create instance with validator, memory monitor, and progress tracker
        try:
            return io_class(
                validator=validator,
                memory_monitor=memory_monitor,
                progress_tracker=progress_tracker,
                **io_kwargs,
            )
        except TypeError:
            # Fallback: create without validator/memory_monitor if constructor doesn't support it
            return io_class(**io_kwargs)

    @classmethod
    def create_data_io_with_data(
        cls,
        data: Any,
        format: str = "fits",
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
        memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
        safety_margin_mb: float = 100.0,
        progress_tracker=None,
        **kwargs,
    ) -> DataIO:
        """Create a data I/O instance with data, validation, and memory monitoring configuration."""
        data_type = type(data)
        if data_type not in cls._data_registry:
            error_msg = f"No I/O class registered for data type: {data_type.__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Create validator with specified configuration
        # Filter out overwrite parameter from kwargs for validator constructor
        validator_kwargs = {k: v for k, v in kwargs.items() if k != "overwrite"}
        validator = DataIOValidator(
            validation_level=validation_level,
            recovery_strategy=recovery_strategy,
            **validator_kwargs,
        )

        # Create memory monitor with specified configuration
        memory_monitor = MemoryMonitor(
            threshold=memory_threshold,
            safety_margin_mb=safety_margin_mb,
            enable_checks=True,
            enable_warnings=True,
        )

        io_class = cls._data_registry[data_type]
        # Filter out overwrite parameter from kwargs for I/O class constructors
        io_kwargs = {k: v for k, v in kwargs.items() if k != "overwrite"}

        # Create instance with validator, memory monitor, and progress tracker
        try:
            return io_class(
                data=data,
                validator=validator,
                memory_monitor=memory_monitor,
                progress_tracker=progress_tracker,
                **io_kwargs,
            )
        except TypeError:
            # Fallback: try different parameter names
            try:
                return io_class(
                    field_line_set=data,
                    validator=validator,
                    memory_monitor=memory_monitor,
                    progress_tracker=progress_tracker,
                    **io_kwargs,
                )
            except TypeError:
                try:
                    return io_class(
                        field=data,
                        validator=validator,
                        memory_monitor=memory_monitor,
                        progress_tracker=progress_tracker,
                        **io_kwargs,
                    )
                except TypeError:
                    # Final fallback: create without validator/memory_monitor
                    try:
                        return io_class(data=data, **io_kwargs)
                    except TypeError:
                        try:
                            return io_class(field_line_set=data, **io_kwargs)
                        except TypeError:
                            return io_class(field=data, **io_kwargs)

    @classmethod
    def create_config_io(
        cls,
        config_type: Type,
        format: str = "yaml",
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_ONCE,
        memory_threshold: MemoryThreshold = MemoryThreshold.MODERATE,
        safety_margin_mb: float = 100.0,
        progress_tracker=None,
        **kwargs,
    ) -> ConfigIO:
        """Create a config I/O instance with validation and memory monitoring configuration."""
        if config_type not in cls._config_registry:
            # Try to register ExperimentSeriesConfig lazily to avoid circular imports
            if config_type.__name__ == "ExperimentSeriesConfig":
                try:
                    from psr_magpy.experiment.experiment_series import (
                        ExperimentSeriesConfig,
                    )

                    cls.register_config_io(
                        ExperimentSeriesConfig, ExperimentSeriesConfigIO
                    )
                    logger.info(
                        "Successfully registered ExperimentSeriesConfig I/O class"
                    )
                except Exception as e:
                    logger.error(f"Failed to register ExperimentSeriesConfig: {e}")

            if config_type not in cls._config_registry:
                error_msg = (
                    f"No I/O class registered for config type: {config_type.__name__}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Create validator with specified configuration
        # Filter out overwrite parameter from kwargs for validator constructor
        validator_kwargs = {k: v for k, v in kwargs.items() if k != "overwrite"}
        validator = ConfigIOValidator(
            validation_level=validation_level,
            recovery_strategy=recovery_strategy,
            **validator_kwargs,
        )

        # Create memory monitor with specified configuration
        memory_monitor = MemoryMonitor(
            threshold=memory_threshold,
            safety_margin_mb=safety_margin_mb,
            enable_checks=True,
            enable_warnings=True,
        )

        io_class = cls._config_registry[config_type]
        # Create instance with validator, memory monitor, and progress tracker
        try:
            return io_class(
                validator=validator,
                memory_monitor=memory_monitor,
                progress_tracker=progress_tracker,
                **kwargs,
            )
        except TypeError:
            # Fallback: create without validator/memory_monitor if constructor doesn't support it
            return io_class(**kwargs)

    @classmethod
    def detect_format(cls, path: Union[str, Path]) -> str:
        """Detect file format from path extension."""
        path = Path(path)
        suffix = path.suffix.lower()

        format_map = {
            ".fits": "fits",
            ".fit": "fits",
            ".yaml": "yaml",
            ".yml": "yaml",
        }

        return format_map.get(suffix, "unknown")

    @classmethod
    def save_data(
        cls, data: Any, path: Union[str, Path], progress_tracker=None, **kwargs
    ) -> None:
        """Save data using appropriate I/O class."""
        format = cls.detect_format(path)

        io_instance = cls.create_data_io_with_data(
            data, format, progress_tracker=progress_tracker, **kwargs
        )
        io_instance.save(data, path, **kwargs)

    @classmethod
    def load_data(
        cls, data_type: Type, path: Union[str, Path], progress_tracker=None, **kwargs
    ) -> Any:
        """Load data using appropriate I/O class."""
        format = cls.detect_format(path)

        io_instance = cls.create_data_io(
            data_type, format, progress_tracker=progress_tracker, **kwargs
        )
        return io_instance.load(path, **kwargs)

    @classmethod
    def save_config(
        cls, config: Any, path: Union[str, Path], progress_tracker=None, **kwargs
    ) -> None:
        """Save configuration using appropriate I/O class."""
        config_type = type(config)
        format = cls.detect_format(path)

        io_instance = cls.create_config_io(
            config_type, format, progress_tracker=progress_tracker, **kwargs
        )
        io_instance.save(config, path, **kwargs)

    @classmethod
    def load_config(
        cls, config_type: Type, path: Union[str, Path], progress_tracker=None, **kwargs
    ) -> Any:
        """Load configuration using appropriate I/O class."""
        format = cls.detect_format(path)

        io_instance = cls.create_config_io(
            config_type, format, progress_tracker=progress_tracker, **kwargs
        )
        return io_instance.load(path, **kwargs)


# Auto-register known I/O classes when module is imported
def _register_default_io_classes():
    """Register default I/O classes."""
    logger.debug("Registering default I/O classes")

    try:
        # Import data types
        from psr_magpy.line.field_line_set import FieldLineSet
        from psr_magpy.field.multipole import MultipoleField
        from psr_magpy.grid.grid import (
            FootpointGrid,
            SphericalGrid,
            HealpixGrid,
            RandomGrid,
            CustomGrid,
        )

        # Import config types
        from psr_magpy.experiment.experiment_config import ExperimentConfig

        # Register data I/O classes
        IOFactory.register_data_io(FieldLineSet, FieldLineSetIO)
        IOFactory.register_data_io(MultipoleField, FieldComponentIO)
        IOFactory.register_data_io(FootpointGrid, GridIO)
        IOFactory.register_data_io(SphericalGrid, GridIO)
        IOFactory.register_data_io(HealpixGrid, GridIO)
        IOFactory.register_data_io(RandomGrid, GridIO)
        IOFactory.register_data_io(CustomGrid, GridIO)

        # Register config I/O classes
        IOFactory.register_config_io(ExperimentConfig, ExperimentConfigIO)

        logger.info("Successfully registered basic I/O classes")

    except ImportError as e:
        logger.warning(f"Could not import some classes for I/O registration: {e}")
    except Exception as e:
        logger.error(f"Error registering I/O classes: {e}")


# Register default classes when module is imported
_register_default_io_classes()
