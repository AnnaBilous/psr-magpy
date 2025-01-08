import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S',
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (default: logging.INFO)
        log_file: Optional file path to save logs
        log_format: Format string for log messages
        date_format: Format string for timestamps
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    
    Raises:
        PermissionError: If log file cannot be created due to permissions
        OSError: If there are other file-related issues
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use RotatingFileHandler instead of FileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            handlers.append(file_handler)
            
        except PermissionError as e:
            print(f"Permission denied creating log file: {e}", file=sys.stderr)
            raise
        except OSError as e:
            print(f"Error creating log file: {e}", file=sys.stderr)
            raise
    
    # Configure logging
    try:
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True  # Python 3.8+: Override existing configuration
        )
    except Exception as e:
        print(f"Failed to configure logging: {e}", file=sys.stderr)
        raise

def cleanup_logging() -> None:
    """Clean up logging handlers."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler) 