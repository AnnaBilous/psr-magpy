"""
Modular pipeline implementation for running field line experiments.

This module provides a pipeline-based approach to running field line experiments,
with distinct stages that can be composed, tested independently, and reused.
"""

import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.calculator import FieldLineCalculator
from psr_magpy.line.field_line import FieldLine

from .experiment_config import ExperimentConfig, TracingConfig
from .run_experiment import get_file_size, log_experiment_status

logger = logging.getLogger(__name__)

# Type variables for input and output types
T = TypeVar('T')
U = TypeVar('U')

# ---- Pipeline Architecture ----

class PipelineStage(Generic[T, U], ABC):
    """Abstract base class for all pipeline stages.
    
    A pipeline stage represents a single processing step that takes
    an input of type T and produces an output of type U.
    """
    
    @abstractmethod
    def process(self, input_data: T) -> U:
        """Process the input data and return the result.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed output data
        """
        pass
    
    def __str__(self) -> str:
        return self.__class__.__name__


class Pipeline:
    """A configurable pipeline for processing data through a series of stages.
    
    The pipeline manages the flow of data through multiple processing stages,
    with support for hooks, error handling, and conditional execution.
    """
    
    def __init__(self):
        """Initialize a new pipeline with no stages."""
        self.stages: List[PipelineStage] = []
        self.hooks: Dict[str, List[Callable]] = {
            "pre_execution": [],
            "post_execution": [],
            "on_error": [],
            "on_completion": []
        }
        self.context: Dict[str, Any] = {}
    
    def add_stage(self, stage: PipelineStage, position: Optional[int] = None) -> 'Pipeline':
        """Add a processing stage to the pipeline.
        
        Args:
            stage: The stage to add
            position: Optional position to insert the stage (default: append)
            
        Returns:
            Self for method chaining
        """
        if position is not None:
            self.stages.insert(position, stage)
        else:
            self.stages.append(stage)
        return self
    
    def add_hook(self, hook_type: str, hook_function: Callable) -> 'Pipeline':
        """Register a new hook function.
        
        Args:
            hook_type: Type of hook ("pre_execution", "post_execution", "on_error", "on_completion")
            hook_function: Function to call when the hook is triggered
            
        Returns:
            Self for method chaining
        """
        if hook_type not in self.hooks:
            raise ValueError(f"Unknown hook type: {hook_type}")
        self.hooks[hook_type].append(hook_function)
        return self
    
    def _execute_hooks(self, hook_type: str, stage: Optional[PipelineStage] = None, data: Any = None) -> None:
        """Execute all registered hooks of the specified type.
        
        Args:
            hook_type: Type of hook to execute
            stage: The stage that triggered the hook (if applicable)
            data: Data to pass to the hook functions
        """
        for hook in self.hooks[hook_type]:
            hook(stage, data, self.context)
    
    def run(self, initial_data: Any) -> Any:
        """Execute all stages in the pipeline on the provided data.
        
        Args:
            initial_data: The initial input data for the pipeline
            
        Returns:
            The final result after all stages have processed the data
        """
        result = initial_data
        
        try:
            # Process each stage
            for stage in self.stages:
                # Execute pre-stage hooks
                self._execute_hooks("pre_execution", stage, result)
                
                try:
                    # Process the data through this stage
                    logger.debug(f"Executing pipeline stage: {stage}")
                    result = stage.process(result)
                    
                    # Store result in context if stage provides a context key
                    if hasattr(stage, 'context_key') and stage.context_key:
                        self.context[stage.context_key] = result
                    
                except Exception as e:
                    # Execute error hooks and re-raise
                    self._execute_hooks("on_error", stage, e)
                    raise
                
                # Execute post-stage hooks
                self._execute_hooks("post_execution", stage, result)
            
            # Execute completion hooks
            self._execute_hooks("on_completion", None, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise


# ---- Experiment Pipeline Stages ----

class ExperimentContext:
    """Context object for experiment pipeline execution.
    
    This class holds all the intermediate data and state during
    pipeline execution.
    """
    
    def __init__(self, config: ExperimentConfig, experiment_number: int = 1, log_file: Optional[str] = None):
        """Initialize experiment context.
        
        Args:
            config: Experiment configuration
            experiment_number: Number of the experiment in the series
            log_file: Path to the log file
        """
        self.config = config
        self.experiment_number = experiment_number
        self.log_file = log_file
        self.start_time = time.time()
        self.output_filename = None
        self.field = None
        self.grid = None
        self.template = None
        self.calculator = None
        self.results = None
        self.file_sizes = None


class SetupLoggingStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Set up logging for the experiment."""
    
    context_key = "log_file"
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Set up logging and update the context."""
        config = context.config
        
        # Set default log file if not provided
        if context.log_file is None or not context.log_file:
            log_dir = config.output.output_dir
            context.log_file = os.path.join(log_dir, f"{config.name}_log.txt")

        # Log experiment start
        log_experiment_status(
            context.log_file, 
            context.experiment_number, 
            config.name, 
            "Started"
        )

        logger.info(f"Starting experiment: {config.name}")
        return context


class CreateOutputFilenameStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Create output filename for the experiment."""
    
    context_key = "output_filename"
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Create output filename and update the context."""
        context.output_filename = str(context.config.output.get_output_filename(context.config.name))
        logger.info(f"Output will be saved to: {context.output_filename}")
        return context


class CreateFieldStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Create magnetic field from configuration."""
    
    context_key = "field"
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Create field and update the context."""
        field_components = [
            comp_config.create_component() for comp_config in context.config.field_components
        ]
        context.field = MultipoleField(field_components)
        logger.info(f"Created magnetic field with {len(field_components)} components")
        return context


class CreateGridStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Create grid from configuration."""
    
    context_key = "grid"
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Create grid and update the context."""
        context.grid = context.config.grid.create_grid()
        logger.info(f"Created {context.config.grid.grid_type} grid with {context.grid.nline} points")
        return context


class CreateTemplateStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Create field line template from configuration."""
    
    context_key = "template"
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Create template and update the context."""
        context.template = FieldLine(**context.config.output.get_save_flags())
        logger.info(f"Successfully created template: {context.template}")
        return context


class CreateCalculatorStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Create field line calculator."""
    
    context_key = "calculator"
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Create calculator and update the context."""
        context.calculator = FieldLineCalculator(context.field, context.grid)
        logger.info(f"Successfully created calculator: {repr(context.calculator)}")
        return context


class CalculateFieldLinesStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Calculate field lines."""
    
    context_key = "results"
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Calculate field lines and update the context."""
        # Update the filename in the tracing config
        tracing_config = context.config.tracing
        tracing_config.filename = context.output_filename

        # Calculate field lines
        logger.info("Starting field line calculations...")
        context.results = context.calculator.get_field_lines(
            template=context.template, 
            config=tracing_config
        )
        logger.info(f"Completed field line calculations: {len(context.results)} lines")
        return context


class SaveConfigurationStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Save experiment configuration to file."""
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Save configuration and update the context."""
        config_path = context.output_filename + "_config.json"
        context.config.save(config_path)
        logger.info(f"Configuration saved to: {config_path}")
        return context


class CalculateFileSizesStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Calculate sizes of output files."""
    
    context_key = "file_sizes"
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Calculate file sizes and update the context."""
        context.file_sizes = {
            "Data": get_file_size(context.output_filename + "*.fits", use_wildcard=True)
        }
        logger.info(f"Output file sizes: {context.file_sizes}")
        return context


class LogCompletionStage(PipelineStage[ExperimentContext, ExperimentContext]):
    """Log experiment completion."""
    
    def process(self, context: ExperimentContext) -> ExperimentContext:
        """Log completion and update the context."""
        duration = time.time() - context.start_time
        log_experiment_status(
            context.log_file,
            context.experiment_number,
            context.config.name,
            "Completed",
            duration=duration,
            file_sizes=context.file_sizes,
        )
        logger.info(f"Experiment completed in {duration:.2f} seconds")
        return context


# ---- Error Handling ----

def log_pipeline_error(stage: Optional[PipelineStage], exception: Exception, context: Dict[str, Any]) -> None:
    """Log pipeline error.
    
    Args:
        stage: The stage that caused the error
        exception: The exception that was raised
        context: Pipeline context
    """
    if 'experiment_context' not in context:
        logger.error(f"Pipeline error: {str(exception)}", exc_info=True)
        return
        
    exp_context = context['experiment_context']
    duration = time.time() - exp_context.start_time
    error_message = f"{type(exception).__name__}: {str(exception)}\n{traceback.format_exc()}"
    
    log_experiment_status(
        exp_context.log_file,
        exp_context.experiment_number,
        exp_context.config.name,
        "Failed",
        duration=duration,
        error=error_message,
    )

    logger.error(f"Error during experiment: {str(exception)}", exc_info=True)


# ---- Pipeline Factory ----

def create_experiment_pipeline() -> Pipeline:
    """Create a standard experiment pipeline.
    
    Returns:
        Configured Pipeline object
    """
    pipeline = Pipeline()
    
    # Add stages
    pipeline.add_stage(SetupLoggingStage())
    pipeline.add_stage(CreateOutputFilenameStage())
    pipeline.add_stage(CreateFieldStage())
    pipeline.add_stage(CreateGridStage())
    pipeline.add_stage(CreateTemplateStage())
    pipeline.add_stage(CreateCalculatorStage())
    pipeline.add_stage(CalculateFieldLinesStage())
    pipeline.add_stage(SaveConfigurationStage())
    pipeline.add_stage(CalculateFileSizesStage())
    pipeline.add_stage(LogCompletionStage())
    
    # Add error hook
    pipeline.add_hook("on_error", log_pipeline_error)
    
    return pipeline


# ---- Pipeline Execution ----

def run_experiment_pipeline(
    config: ExperimentConfig, experiment_number: int = 1, log_file: Optional[str] = None
):
    """Run the experiment pipeline.

    Args:
        config: Experiment configuration
        experiment_number: Number of the experiment in the series
        log_file: Path to the log file

    Returns:
        Results of the experiment
    """
    # Create context
    context = ExperimentContext(config, experiment_number, log_file)
    
    # Create and run pipeline
    pipeline = create_experiment_pipeline()
    pipeline.context['experiment_context'] = context
    
    try:
        result_context = pipeline.run(context)
        return result_context.results
    except Exception as e:
        logger.error(f"Experiment pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Example usage
    from psr_magpy.experiment.experiment_config import sparse_meridional_field_lines_config
    
    run_experiment_pipeline(sparse_meridional_field_lines_config, experiment_number=1)
