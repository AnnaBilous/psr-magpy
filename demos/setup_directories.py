from pathlib import Path
from typing import Tuple

def setup_output_directories(script_path: Path ) -> Tuple[Path, Path, Path, Path]:
    """
    Create and return output directories relative to the given script location.
    
    Args:
        script_path: Path to the script calling this function
    
    Returns:
        tuple containing:
            - output_dir: Main output directory
            - fits_dir: Directory for FITS files
            - plot_dir: Directory for plots
            - log_dir: Directory for log files
    """
    script_dir = Path(script_path).parent
    output_dir = script_dir / "output"
    fits_dir = output_dir / "fits"
    plot_dir = output_dir / "plots"
    log_dir = output_dir / "logs"

    for directory in [fits_dir, plot_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        
    return output_dir, fits_dir, plot_dir, log_dir

if __name__ == "__main__":
    # Test the function when run directly
    setup_output_directories(__file__) 