"""Central logging configuration."""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Union, Dict
from datetime import datetime

class LogLevels:
    """Logging levels for different handlers."""
    CONSOLE = logging.INFO
    FILE = logging.DEBUG
    MEMORY = logging.WARNING  # For memory-related operations
    CALCULATION = logging.INFO  # For calculation progress
    IO = logging.INFO  # For I/O operations

def setup_logging(
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    handler_levels: Optional[Dict[str, int]] = None
) -> None:
    """Configure logging for the entire application.
    
    Parameters
    ----------
    log_level : Union[str, int]
        Base logging level (e.g., 'DEBUG', 'INFO', etc.)
    log_file : Optional[Union[str, Path]]
        Path to log file. If None, only console logging is enabled
    log_format : str
        Format string for log messages
    max_bytes : int
        Maximum size of each log file before rotation
    backup_count : int
        Number of backup files to keep
    handler_levels : Optional[Dict[str, int]]
        Custom logging levels for different handlers
        e.g., {'console': logging.INFO, 'file': logging.DEBUG}
    """
    # Convert string level to numeric if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Use default handler levels if none provided
    if handler_levels is None:
        handler_levels = {
            'console': LogLevels.CONSOLE,
            'file': LogLevels.FILE
        }
    
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(handler_levels.get('console', log_level))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(handler_levels.get('file', log_level))
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Set root logger level to the most verbose of all handlers
    root_logger.setLevel(min(handler.level for handler in root_logger.handlers))
    
    # Suppress external library logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('astropy').setLevel(logging.WARNING)
    
    # Log configuration details
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with base level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    # Log handler levels
    for handler in root_logger.handlers:
        handler_name = 'console' if isinstance(handler, logging.StreamHandler) else 'file'
        logger.debug(f"{handler_name} handler level: {logging.getLevelName(handler.level)}") 