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

            return ExperimentResult(
                experiment_name=self.config.name,
                success=True,
                duration=time.time() - start_time,
                results=results,
                error_message=error_msg,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Experiment {self.config.name} failed: {error_msg}")
            return ExperimentResult(
                experiment_name=self.config.name,
                success=False,
                duration=time.time() - start_time,
                error_message=error_msg,
            )


def run_experiment(config: ExperimentConfig) -> Optional[object]:
    """Convenience function to run an experiment."""
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_from_file(config_path: str) -> Optional[object]:
    """Load and run experiment from config file."""
    config = ExperimentConfig.load(config_path)
    return run_experiment(config)
