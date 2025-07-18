import time
from pathlib import Path
from typing import Optional

from psr_magpy.line.calculator import FieldLineCalculator
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

from .simple_config import ExperimentConfig

logger = get_logger(__name__, console_format_brief=True)


class ExperimentRunner:
    """Simple experiment runner for field line calculations."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize with experiment configuration."""
        self.config = config
        logger.debug(f"Initialized runner for experiment: {config.name}")
    
    def run(self) -> Optional[object]:
        """Run the complete experiment pipeline."""
        start_time = time.time()
        logger.info(f"Starting experiment: {self.config.name}")
        
        try:
            # Create field
            logger.debug("Creating magnetic field")
            field = self.config.field.create_field()
            
            # Create grid
            logger.debug("Creating grid")
            grid = self.config.grid.create_grid()
            logger.info(f"Grid created with {grid.nline} points")
            
            # Create calculator
            logger.debug("Creating field line calculator")
            calculator = FieldLineCalculator(field, grid, self.config.tracing)
            
            # Use template from config
            logger.debug("Using template from config")
            template = self.config.template
            
            # Run calculation
            logger.info("Starting field line calculations...")
            results = calculator.get_field_lines(template)
            
            # Log completion
            duration = time.time() - start_time
            logger.info(f"Experiment completed in {duration:.2f}s")
            
            if results is not None:
                logger.info(f"Results: {results}")
                results.print_summary()
            else:
                logger.info("Results saved to disk")
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Experiment failed after {duration:.2f}s: {e}")
            raise


def run_experiment(config: ExperimentConfig) -> Optional[object]:
    """Convenience function to run an experiment."""
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_from_file(config_path: str) -> Optional[object]:
    """Load and run experiment from config file."""
    config = ExperimentConfig.load(config_path)
    return run_experiment(config) 