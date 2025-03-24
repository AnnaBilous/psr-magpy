from pathlib import Path

def setup_output_directories(path: str):
    """Create all necessary directories for the project."""


    # Create output directories
    output_dir = Path(path)
    fits_dir = output_dir / "fits"
    plot_dir = output_dir / "plots"
    log_dir = output_dir / "logs"

    # Create all directories
    directories = [
   
        fits_dir, plot_dir, log_dir
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    return fits_dir, plot_dir, log_dir

if __name__ == "__main__":
    setup_output_directories()
