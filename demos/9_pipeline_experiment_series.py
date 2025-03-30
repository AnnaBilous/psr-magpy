#!/usr/bin/env python
"""Demo for running experiment series using the enhanced pipeline architecture."""

from pathlib import Path
import logging
from typing import List, Optional

# Import directly from the package
from psr_magpy.cli.setup_environment import setup_environment

# Set up environment
env = setup_environment(output_dir=Path(__file__).parent / "output")
logger = env["logger"]
fits_dir = env["directories"]["fits"]
plot_dir = env["directories"]["plots"]
log_dir = env["directories"]["logs"]

# Import pipeline components
from psr_magpy.experiment.experiment_pipeline import (
    Pipeline,
    PipelineStage,
    ExperimentContext,
    create_experiment_pipeline,
)

# Import configuration components
from psr_magpy.experiment.experiment_config import GridConfig, OutputConfig, ExperimentConfig
from psr_magpy.line.calculator import TracingConfig
from psr_magpy.experiment.field_factory_config import (
    FieldComponentFactory,
    MultipoleFieldFactory,
)
from psr_magpy.experiment.experiment_series_config import (
    ExperimentSeriesConfig,
    analyze_experiment_log,
)


class ExperimentSeriesContext:
    """Context object for experiment series pipeline execution."""
    
    def __init__(self, config: ExperimentSeriesConfig, clear_log: bool = False):
        """Initialize experiment series context.
        
        Args:
            config: Experiment series configuration
            clear_log: Whether to clear the log file before starting
        """
        self.config = config
        self.clear_log = clear_log
        self.experiments: List[ExperimentConfig] = []
        self.results = []
        self.log_file: Optional[str] = None
        self.config_path: Optional[str] = None
        self.summary = None


class SaveSeriesConfigStage(PipelineStage[ExperimentSeriesContext, ExperimentSeriesContext]):
    """Save experiment series configuration to file."""
    
    def process(self, context: ExperimentSeriesContext) -> ExperimentSeriesContext:
        """Save configuration and update the context."""
        config_path = f"{fits_dir}/{context.config.name}_series_config.json"
        context.config.save(config_path)
        context.config_path = config_path
        logger.info(f"Series configuration saved to: {config_path}")
        return context


class ExpandExperimentsStage(PipelineStage[ExperimentSeriesContext, ExperimentSeriesContext]):
    """Expand series configuration to individual experiments."""
    
    def process(self, context: ExperimentSeriesContext) -> ExperimentSeriesContext:
        """Expand experiments and update the context."""
        context.experiments = context.config.expand_to_experiments()
        logger.info(f"Generated {len(context.experiments)} experiment configurations")
        
        # Print details of each experiment
        for i, exp in enumerate(context.experiments):
            logger.info(f"Experiment {i+1}:")
            for comp in exp.field_components:
                logger.info(
                    f"  - {comp.type}: B={comp.B_magnitude}, "
                    f"lat={comp.moment_orientation_latitude_degree}°, "
                    f"lon={comp.moment_orientation_longitude_degree}°"
                )
        
        return context


class SetupSeriesLogFileStage(PipelineStage[ExperimentSeriesContext, ExperimentSeriesContext]):
    """Set up log file for the experiment series."""
    
    def process(self, context: ExperimentSeriesContext) -> ExperimentSeriesContext:
        """Set up log file and update the context."""
        log_file = f"{log_dir}/{context.config.name}_series.log"
        
        # Clear log file if requested
        if context.clear_log and Path(log_file).exists():
            with open(log_file, 'w') as f:
                f.write(f"# Experiment Series: {context.config.name}\n")
                f.write(f"# Description: {context.config.description}\n")
                f.write("# Log cleared and restarted\n\n")
            logger.info(f"Cleared log file: {log_file}")
        
        context.log_file = log_file
        return context


class RunExperimentsStage(PipelineStage[ExperimentSeriesContext, ExperimentSeriesContext]):
    """Run all experiments in the series."""
    
    def process(self, context: ExperimentSeriesContext) -> ExperimentSeriesContext:
        """Run experiments and update the context."""
        results = []
        
        for i, experiment_config in enumerate(context.experiments):
            logger.info(f"Running experiment {i+1}/{len(context.experiments)}: {experiment_config.name}")
            
            # Create experiment context
            exp_context = ExperimentContext(
                config=experiment_config,
                experiment_number=i+1,
                log_file=context.log_file
            )
            
            # Create and run experiment pipeline
            pipeline = create_experiment_pipeline()
            pipeline.context['experiment_context'] = exp_context
            
            try:
                result_context = pipeline.run(exp_context)
                results.append(result_context.results)
                logger.info(f"Experiment {i+1} completed successfully")
            except Exception as e:
                logger.error(f"Experiment {i+1} failed: {str(e)}", exc_info=True)
                results.append(None)
        
        context.results = results
        return context


class AnalyzeResultsStage(PipelineStage[ExperimentSeriesContext, ExperimentSeriesContext]):
    """Analyze the results of the experiment series."""
    
    def process(self, context: ExperimentSeriesContext) -> ExperimentSeriesContext:
        """Analyze results and update the context."""
        # Analyze the log file
        summary = analyze_experiment_log(context.config)
        context.summary = summary
        
        logger.info(
            f"Found {summary['completed']} completed experiments out of {summary['total_experiments']}"
        )
        
        # Check for incomplete experiments
        if summary['completed'] < summary['total_experiments']:
            logger.info("Some experiments did not complete successfully.")
            
            # List failed experiments
            if 'failed' in summary and summary['failed']:
                logger.info(f"Failed experiments: {', '.join(map(str, summary['failed']))}")
        else:
            logger.info("All experiments completed successfully!")
        
        return context


def create_series_pipeline() -> Pipeline:
    """Create a pipeline for running experiment series.
    
    Returns:
        Configured Pipeline object
    """
    pipeline = Pipeline()
    
    # Add stages
    pipeline.add_stage(SaveSeriesConfigStage())
    pipeline.add_stage(ExpandExperimentsStage())
    pipeline.add_stage(SetupSeriesLogFileStage())
    pipeline.add_stage(RunExperimentsStage())
    pipeline.add_stage(AnalyzeResultsStage())
    
    return pipeline


def create_sample_series_config() -> ExperimentSeriesConfig:
    """Create a sample experiment series configuration."""

    # Create field component factories with parameter ranges
    dipole_factory = FieldComponentFactory(
        moment_type="dipole",
        B_magnitude=[0.5, 1.0],  # Multiple field strengths
        moment_orientation_latitude_degree=90,
        moment_orientation_longitude_degree=0,
    )

    quadrupole_factory = FieldComponentFactory(
        moment_type="quadrupole",
        B_magnitude={"1": 0.5, "2": 1.0},
        moment_orientation_latitude_degree={"1": 45, "2": 90},
        moment_orientation_longitude_degree=0,
    )

    # Create a multipole field factory with both components
    field_factories = MultipoleFieldFactory(
        [dipole_factory, quadrupole_factory], [dipole_factory]
    )

    # Create the experiment series configuration
    return ExperimentSeriesConfig(
        name="pipeline_dipole_series",
        description="Testing pipeline architecture with dipole orientations",
        grid=GridConfig(grid_type="healpix", nside=4),  # Use fewer points for demo
        tracing=TracingConfig(
            R_max=10.0,
            R_min=1.0,
            dt=0.01,
            max_memory_mb=100.0,
            n_cores=2,
            max_batch_size=500,
        ),
        field_factories=field_factories,
        output=OutputConfig(
            output_dir=fits_dir,
            file_prefix="pipeline_",
            save_r_max=True,
            save_b_r_ini_angle=True,
            save_r_line=True,
        ),
    )


def run_series_with_pipeline(config: ExperimentSeriesConfig, clear_log: bool = False):
    """Run an experiment series using the pipeline architecture.
    
    Args:
        config: Experiment series configuration
        clear_log: Whether to clear the log file before starting
        
    Returns:
        Results of the experiment series
    """
    # Create context
    context = ExperimentSeriesContext(config, clear_log)
    
    # Create and run pipeline
    pipeline = create_series_pipeline()
    
    try:
        result_context = pipeline.run(context)
        return result_context.results
    except Exception as e:
        logger.error(f"Experiment series pipeline failed: {str(e)}", exc_info=True)
        raise


def resume_series_with_pipeline(config: ExperimentSeriesConfig):
    """Resume an experiment series using the pipeline architecture.
    
    Args:
        config: Experiment series configuration
        
    Returns:
        Results of the resumed experiment series
    """
    # First analyze the log to see what needs to be resumed
    summary = analyze_experiment_log(config)
    
    if summary['completed'] == summary['total_experiments']:
        logger.info("All experiments have already been completed.")
        return []
    
    # Create a new configuration with only the experiments that need to be run
    resume_config = ExperimentSeriesConfig(
        name=f"{config.name}_resume",
        description=f"Resuming {config.description}",
        grid=config.grid,
        tracing=config.tracing,
        field_factories=config.field_factories,
        output=config.output,
    )
    
    # Expand to all experiments
    all_experiments = config.expand_to_experiments()
    
    # Filter to only the ones that need to be run
    if 'failed' in summary and summary['failed']:
        experiments_to_run = [all_experiments[i-1] for i in summary['failed']]
        logger.info(f"Resuming {len(experiments_to_run)} failed experiments")
    else:
        # If no specific failures are listed, run all incomplete experiments
        completed_indices = set(summary.get('completed_indices', []))
        experiments_to_run = [
            exp for i, exp in enumerate(all_experiments) 
            if i+1 not in completed_indices
        ]
        logger.info(f"Resuming {len(experiments_to_run)} incomplete experiments")
    
    # Create a custom pipeline that only runs the specified experiments
    pipeline = Pipeline()
    pipeline.add_stage(SetupSeriesLogFileStage())
    
    class RunSelectedExperimentsStage(PipelineStage[ExperimentSeriesContext, ExperimentSeriesContext]):
        def __init__(self, experiments):
            self.experiments = experiments
            
        def process(self, context: ExperimentSeriesContext) -> ExperimentSeriesContext:
            results = []
            
            for i, experiment_config in enumerate(self.experiments):
                logger.info(f"Running experiment {i+1}/{len(self.experiments)}: {experiment_config.name}")
                
                # Create experiment context
                exp_context = ExperimentContext(
                    config=experiment_config,
                    experiment_number=i+1,
                    log_file=context.log_file
                )
                
                # Create and run experiment pipeline
                exp_pipeline = create_experiment_pipeline()
                exp_pipeline.context['experiment_context'] = exp_context
                
                try:
                    result_context = exp_pipeline.run(exp_context)
                    results.append(result_context.results)
                    logger.info(f"Experiment {i+1} completed successfully")
                except Exception as e:
                    logger.error(f"Experiment {i+1} failed: {str(e)}", exc_info=True)
                    results.append(None)
            
            context.results = results
            return context
    
    pipeline.add_stage(RunSelectedExperimentsStage(experiments_to_run))
    pipeline.add_stage(AnalyzeResultsStage())
    
    # Create context and run pipeline
    context = ExperimentSeriesContext(resume_config, clear_log=False)
    
    try:
        result_context = pipeline.run(context)
        return result_context.results
    except Exception as e:
        logger.error(f"Resume pipeline failed: {str(e)}", exc_info=True)
        raise


def main():
    # Create configuration
    config = create_sample_series_config()
    
    # Run the experiment series with the pipeline
    logger.info("Running experiment series with pipeline architecture...")
    results = run_series_with_pipeline(config, clear_log=True)
    
    # Check if we need to resume any experiments
    summary = analyze_experiment_log(config)
    if summary["completed"] < summary["total_experiments"]:
        logger.info("Resuming experiment series...")
        resume_results = resume_series_with_pipeline(config)
    else:
        logger.info("All experiments have been completed.")


if __name__ == "__main__":
    main() 