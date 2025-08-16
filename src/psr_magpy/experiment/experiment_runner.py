"""Runner for executing individual experiments."""

import time
from typing import Optional
from pathlib import Path

from psr_magpy.line import FieldLineCalculator
from psr_magpy.utils.logging_config import get_logger
from .experiment_config import ExperimentConfig
from .experiment_result import ExperimentResult

logger = get_logger(__name__, console_format_brief=True)


class ExperimentRunner:
    """Runner for executing individual experiments."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize experiment runner."""
        self.config = config
        logger.debug(f"Initialized runner for experiment: {config.name}")

    def run(self) -> ExperimentResult:
        """Run the experiment."""
        start_time = time.time()
        logger.info(f"Starting experiment: {self.config.name}")

        try:
            # Create field
            field = self.config.field.create_field()

            # Create grid
            grid = self.config.grid.create_grid()
            if grid is None:
                error_msg = "Failed to create grid"
                logger.error(error_msg)
                return ExperimentResult(
                    experiment_name=self.config.name,
                    success=False,
                    duration=time.time() - start_time,
                    error_message=error_msg,
                )
            logger.info(f"Grid created with {grid.nline} points")

            # Auto-save grids if saving results to disk
            if self.config.tracing.save_results_to_disk:
                # Save RandomGrid and CustomGrid (always saved when results saved)
                if (self.config.grid.random is not None or self.config.grid.custom is not None):
                    self._save_grid(grid)
                # Save subset grids for SphericalGrid and HealpixGrid
                elif grid.partial and (self.config.grid.spherical is not None or self.config.grid.healpix is not None):
                    self._save_subset_grid(grid)

            # Create calculator
            calculator = FieldLineCalculator(
                field=field,
                grid=grid,
                params=self.config.tracing.create_tracing_parameters(),
            )

            # Run calculations
            logger.info("Starting field line calculations...")
            results = calculator.get_field_lines(self.config.template)

            # Clear references to free memory
            calculator = None
            field = None
            grid = None

            # Log results
            logger.info(f"Experiment completed in {time.time() - start_time:.2f}s")
            logger.info(f"Results: {results}")

            # Check for errors in results
            error_msg = None
            if results is not None:
                if hasattr(results, "reversed_error_warning"):
                    error_msg = results.reversed_error_warning
                elif hasattr(results, "dotprod_error"):
                    error_msg = results.dotprod_error

            # Create result object
            result = ExperimentResult(
                experiment_name=self.config.name,
                success=True,  # Always true for reversed error
                duration=time.time() - start_time,
                results=results,
                error_message=error_msg,  # Include error message even if success is true
            )

            # Log error if present
            if error_msg:
                logger.warning(f"Experiment completed with warning: {error_msg}")

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Experiment {self.config.name} failed: {error_msg}")
            return ExperimentResult(
                experiment_name=self.config.name,
                success=False,
                duration=time.time() - start_time,
                error_message=error_msg,
            )

    def _save_subset_grid(self, grid) -> None:
        """Automatically save subset grid to the grids directory."""
        from pathlib import Path
        
        # Create grids directory in current working directory
        grids_dir = Path.cwd() / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate grid filename
        grid_filename = f"{self.config.name}_subset_grid.fits"
        grid_path = grids_dir / grid_filename
        
        # Save the grid
        logger.info(f"Saving subset grid to {grid_path}")
        grid.save_to_fits(str(grid_path))
        
        # Update the config with the grid file path
        self._update_grid_config_with_path(str(grid_path))
        
        logger.info(f"Subset grid saved and config updated with path: {grid_path}")

    def _save_grid(self, grid) -> None:
        """Automatically save grid to the grids directory."""
        from pathlib import Path
        
        # Create grids directory in current working directory
        grids_dir = Path.cwd() / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate grid filename
        grid_filename = f"{self.config.name}_grid.fits"
        grid_path = grids_dir / grid_filename
        
        # Save the grid
        logger.info(f"Saving grid to {grid_path}")
        grid.save_to_fits(str(grid_path))
        
        # Update the config with the grid file path
        self._update_grid_config_with_path(str(grid_path))
        
        logger.info(f"Grid saved and config updated with path: {grid_path}")

    def _update_grid_config_with_path(self, grid_path: str) -> None:
        """Update the grid config with the saved file path."""
        # Update the appropriate grid config based on type
        if self.config.grid.spherical is not None:
            self.config.grid.spherical.grid_file_path = grid_path
            self.config.grid.spherical.subset = True
        elif self.config.grid.healpix is not None:
            self.config.grid.healpix.grid_file_path = grid_path
            self.config.grid.healpix.subset = True
        elif self.config.grid.random is not None:
            self.config.grid.random.grid_file_path = grid_path
        elif self.config.grid.custom is not None:
            self.config.grid.custom.grid_file_path = grid_path


def run_experiment(config: ExperimentConfig) -> Optional[object]:
    """Convenience function to run an experiment."""
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_from_file(config_path: str) -> Optional[object]:
    """Load and run experiment from config file."""
    config = ExperimentConfig.load(config_path)
    return run_experiment(config)
