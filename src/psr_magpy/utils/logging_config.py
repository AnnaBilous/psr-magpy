"""Central logging configuration for psr-magpy."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import os
import sys

class LogLevels:
    """Logging levels for different handlers."""
    CRITICAL = logging.CRITICAL  # 50
    ERROR = logging.ERROR        # 40
    WARNING = logging.WARNING    # 30
    INFO = logging.INFO          # 20
    DEBUG = logging.DEBUG        # 10
    NOTSET = logging.NOTSET      # 0
    
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
        
    Returns
    -------
    logging.Logger
        Root logger
    """
    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Set default handler levels
    if handler_levels is None:
        handler_levels = {
            'console': LogLevels.CONSOLE,
            'file': LogLevels.FILE
        }
    
    # Set default formats
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(handler_levels.get('console', LogLevels.CONSOLE))
    console_handler.setFormatter(formatter)
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
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(handler_levels.get('file', LogLevels.FILE))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Capture warnings from the warnings module
    if capture_warnings:
        logging.captureWarnings(True)
    
    # Set propagation for the psr_magpy logger
    psr_magpy_logger = logging.getLogger('psr_magpy')
    psr_magpy_logger.propagate = propagate
    
    # Log configuration details
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with base level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    # Log handler levels
    for handler in root_logger.handlers:
        handler_name = 'console' if isinstance(handler, logging.StreamHandler) else 'file'
        logger.debug(f"{handler_name} handler level: {logging.getLevelName(handler.level)}")
    
    return root_logger

def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get a logger with the given name, setting up logging if needed.
    
    Parameters
    ----------
    name : str
        Logger name
    **kwargs
        Additional arguments to pass to setup_logging if it hasn't been called yet
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If root logger has no handlers, set up logging
    if not logging.getLogger().handlers:
        setup_logging(**kwargs)
    
    return logger 