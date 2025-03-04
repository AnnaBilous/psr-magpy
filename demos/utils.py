"""Utility functions for demo scripts."""
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple

# Configure astropy logging first
import astropy
astropy.conf.logging_level = 'WARNING'

from setup_directories import setup_output_directories

def setup_demo(script_path: str, log_level: str = 'DEBUG') -> Tuple[Path, Path, Path, logging.Logger]:
    """Set up environment for demo scripts.
    
    This function:
    1. Adds project root to PYTHONPATH
    2. Creates output directories
    3. Sets up logging
    
    Parameters
    ----------
    script_path : str
        Path to the demo script (__file__)
    log_level : str, optional
        Base logging level, by default 'DEBUG'
        
    Returns
    -------
    Tuple[Path, Path, Path, logging.Logger]
        fits_dir: Directory for FITS files
        plot_dir: Directory for plots
        log_dir: Directory for logs
        logger: Configured logger for the demo
    """
    # Add project root to PYTHONPATH first
    project_root = Path(script_path).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Now we can import from src
    from src.utils.logging_config import setup_logging, LogLevels
    
    # Create output directories
    _, fits_dir, plot_dir, log_dir = setup_output_directories(script_path)
    
    # Setup logging
    script_name = Path(script_path).stem
    log_file = log_dir / f"{script_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        handler_levels={
            'console': LogLevels.CONSOLE,
            'file': LogLevels.FILE
        }
    )
    
    logger = logging.getLogger(script_name)
    logger.info(f"Starting {script_name} demo")
    
    return fits_dir, plot_dir, log_dir, logger 