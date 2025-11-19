"""Central logging configuration for psr-magpy."""

import logging
import logging.handlers
import sys
import os

from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from colorlog import ColoredFormatter

# Try to import colorlog for colored console output
try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


class LogLevels:
    """Logging levels for different handlers."""

    CRITICAL = logging.CRITICAL  # 50
    ERROR = logging.ERROR  # 40
    WARNING = logging.WARNING  # 30
    INFO = logging.INFO  # 20
    DEBUG = logging.DEBUG  # 10
    NOTSET = logging.NOTSET  # 0

    # Default levels for different handlers
    CONSOLE = logging.INFO
    FILE = logging.DEBUG


def get_log_dir(base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Get or create the log directory.

    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory for logs. If None, uses the current directory.

    Returns
    -------
    Path
        Path to the log directory
    """
    if base_dir is None:
        # Use current directory by default
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir


def setup_logging(
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    handler_levels: Optional[Dict[str, int]] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    capture_warnings: bool = True,
    propagate: bool = True,
    use_colors: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console_format_brief: bool = True,
    debug_mode: bool = False,
    modules_debug: Optional[List[str]] = None,
    show_process_id: bool = False,
    show_filename: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Parameters
    ----------
    log_level : int or str, optional
        Base logging level, by default logging.INFO
    log_file : str or Path, optional
        Path to log file. If None and log_dir is provided,
        a default filename will be used.
    log_dir : str or Path, optional
        Directory for log files. If None, logs will only go to console.
    handler_levels : dict, optional
        Dictionary mapping handler names to log levels
    log_format : str, optional
        Format string for log messages
    date_format : str, optional
        Format string for dates in log messages
    capture_warnings : bool, optional
        Whether to capture warnings from the warnings module
    propagate : bool, optional
        Whether to propagate log messages to parent loggers
    use_colors : bool, optional
        Whether to use colored output in console (if colorlog is available)
    max_file_size : int, optional
        Maximum size of log file before rotation (in bytes)
    backup_count : int, optional
        Number of backup log files to keep
    console_format_brief : bool, optional
        Use a more concise format for console output
    debug_mode : bool, optional
        Enable debug mode (sets all loggers to DEBUG level)
    modules_debug : list, optional
        List of module names to set to DEBUG level (if debug_mode is False)
    show_process_id : bool, optional
        Whether to include process ID in log messages
    show_filename : bool, optional
        Whether to include the Python filename in log messages

    Returns
    -------
    logging.Logger
        Root logger
    """
    # Override log level if debug mode is enabled
    if debug_mode:
        log_level = logging.DEBUG
        if handler_levels is None:
            handler_levels = {"console": logging.DEBUG, "file": logging.DEBUG}
        else:
            handler_levels["console"] = logging.DEBUG
            handler_levels["file"] = logging.DEBUG

    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Set default handler levels
    if handler_levels is None:
        handler_levels = {"console": LogLevels.CONSOLE, "file": LogLevels.FILE}

    # Set default formats
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Use a more concise format for console if requested
    if console_format_brief:
        format_parts = []
        if show_process_id:
            format_parts.append("[%(process)d]")
        if show_filename:
            format_parts.append("[%(filename)s]")

        prefix = " ".join(format_parts)
        if prefix:
            prefix += " "

        console_format = f"{prefix}%(levelname)-8s %(message)s"
    else:
        format_parts = ["%(asctime)s"]
        if show_process_id:
            format_parts.append("[%(process)d]")
        if show_filename:
            format_parts.append("[%(filename)s]")

        prefix = " ".join(format_parts)
        console_format = f"{prefix} - %(name)s - %(levelname)s - %(message)s"

    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create formatters
    file_formatter = logging.Formatter(log_format, date_format)

    # Use colored formatter for console if available and requested
    if HAS_COLORLOG and use_colors:
        color_formatter: Union[logging.Formatter, "ColoredFormatter"] = colorlog.ColoredFormatter(
            "%(log_color)s" + console_format,
            datefmt=date_format,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        color_formatter = logging.Formatter(console_format, date_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(handler_levels.get("console", LogLevels.CONSOLE))
    console_handler.setFormatter(color_formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file or log_dir is specified
    if log_file is not None or log_dir is not None:
        if log_file is None:
            # Create default log filename with timestamp
            log_dir = get_log_dir(log_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"psr_magpy_{timestamp}.log"
        else:
            log_file = Path(log_file)
            # Ensure parent directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use RotatingFileHandler instead of FileHandler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(handler_levels.get("file", LogLevels.FILE))
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Capture warnings from the warnings module
    if capture_warnings:
        logging.captureWarnings(True)

    # Set propagation for the psr_magpy logger
    psr_magpy_logger = logging.getLogger("psr_magpy")
    psr_magpy_logger.propagate = propagate

    # Set specific modules to DEBUG level if requested
    if modules_debug and not debug_mode:
        for module_name in modules_debug:
            logging.getLogger(module_name).setLevel(logging.DEBUG)

    # Log configuration details
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured with base level: {logging.getLevelName(log_level)}"
    )
    if log_file:
        logger.info(f"Log file: {log_file}")

    # Log handler levels
    for handler in root_logger.handlers:
        handler_name = (
            "console"
            if isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.handlers.RotatingFileHandler)
            else "file"
        )
        logger.debug(
            f"{handler_name} handler level: {logging.getLevelName(handler.level)}"
        )

    return root_logger


def get_logger(
    name: str, show_process_id: bool = False, show_filename: bool = True, **kwargs
) -> logging.Logger:
    """
    Get a logger with the given name, setting up logging if needed.

    Parameters
    ----------
    name : str
        Logger name
    show_process_id : bool, optional
        Whether to include process ID in log messages, by default False
    show_filename : bool, optional
        Whether to include the Python filename in log messages, by default True
    **kwargs
        Additional arguments to pass to setup_logging if it hasn't been called yet

    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)

    # Check for environment variable to enable debug mode
    debug_env = os.environ.get("PSR_MAGPY_DEBUG", "").lower()
    if debug_env in ("1", "true", "yes") and "debug_mode" not in kwargs:
        kwargs["debug_mode"] = True

    # If root logger has no handlers, set up logging
    if not logging.getLogger().handlers:
        kwargs["show_process_id"] = show_process_id
        kwargs["show_filename"] = show_filename
        setup_logging(**kwargs)

    return logger


def set_debug_for_modules(module_names: List[str], enable: bool = True) -> None:
    """
    Set debug level for specific modules.

    Parameters
    ----------
    module_names : list of str
        List of module names to set debug level for
    enable : bool, optional
        Whether to enable (True) or disable (False) debug mode
    """
    level = logging.DEBUG if enable else logging.INFO
    for module_name in module_names:
        logging.getLogger(module_name).setLevel(level)

    logger = logging.getLogger(__name__)
    action = "Enabled" if enable else "Disabled"
    logger.info(f"{action} debug logging for modules: {', '.join(module_names)}")
