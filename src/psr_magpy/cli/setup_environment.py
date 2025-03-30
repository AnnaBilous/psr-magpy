"""Set up the environment for running psr-magpy."""

import argparse
import sys
from pathlib import Path
import logging


def find_project_root(start_path=None):
    """
    Find the project root directory.

    Parameters
    ----------
    start_path : str or Path, optional
        Starting path for the search. If None, uses the current directory.

    Returns
    -------
    Path
        Path to the project root directory
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    # Look for pyproject.toml or setup.py
    current = start_path
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent

    # If we can't find the project root, use the starting directory
    return start_path


def setup_directories(base_dir=None, subdirs=None):
    """
    Set up directories for outputs.

    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory for outputs. If None, uses the current directory.
    subdirs : list, optional
        List of subdirectories to create. If None, creates 'fits', 'plots', and 'logs'.

    Returns
    -------
    dict
        Dictionary mapping directory names to paths
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    if subdirs is None:
        subdirs = ["fits", "plots", "logs"]

    # Create base directory
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    directories = {"base": base_dir}
    for subdir in subdirs:
        dir_path = base_dir / subdir
        dir_path.mkdir(exist_ok=True)
        directories[subdir] = dir_path

    return directories


def setup_environment(output_dir=None, log_level="INFO", add_to_path=True):
    """
    Set up the environment for running psr-magpy.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory for outputs. If None, uses the current directory.
    log_level : str, optional
        Logging level, by default 'INFO'
    add_to_path : bool, optional
        Whether to add the project root to sys.path, by default True

    Returns
    -------
    dict
        Dictionary containing setup information
    """
    # Find project root
    project_root = find_project_root()

    # Add project root to sys.path if needed
    if add_to_path and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Set up output directories
    if output_dir is None:
        output_dir = project_root / "output"

    directories = setup_directories(output_dir)

    # Set up logging
    from src.psr_magpy.utils.logging_config import setup_logging

    log_file = directories["logs"] / f"psr_magpy_{Path.cwd().name}.log"
    root_logger = setup_logging(log_level=log_level, log_file=log_file)

    # Get a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Environment set up with output directory: {output_dir}")
    logger.info(f"Project root: {project_root}")

    return {"project_root": project_root, "directories": directories, "logger": logger}


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Set up the environment for psr-magpy")
    parser.add_argument("--output-dir", help="Directory for outputs")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    env = setup_environment(args.output_dir, args.log_level)

    print(f"Environment set up successfully:")
    print(f"  - Project root: {env['project_root']}")
    print(f"  - Output directory: {env['directories']['base']}")
    print(f"  - Log file: {env['directories']['logs']}")


if __name__ == "__main__":
    main()
