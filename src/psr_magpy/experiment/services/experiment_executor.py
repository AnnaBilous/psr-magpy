"""Service for executing a single experiment with consistent failure policy."""

from dataclasses import dataclass
import time
from typing import Callable, cast

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_result import ExperimentResult
from psr_magpy.experiment.experiment_runner import run_experiment


@dataclass
class ExperimentExecutionOutcome:
    """Single-experiment execution result with measured wall-clock duration."""

    result: ExperimentResult
    duration: float


class ExperimentExecutor:
    """Execute one experiment and apply continue-on-failure policy."""

    def __init__(self, run_func: Callable = run_experiment) -> None:
        self._run_func = run_func

    def execute(
        self,
        experiment: ExperimentConfig,
        *,
        save_grids: bool,
        continue_on_failure: bool,
    ) -> ExperimentExecutionOutcome:
        """Run one experiment and return structured outcome."""
        start_time = time.time()
        try:
            result = cast(
                ExperimentResult, self._run_func(experiment, save_grids=save_grids)
            )
            duration = time.time() - start_time

            if not result.success and not continue_on_failure:
                error_msg = result.error_message or "Unknown error"
                raise RuntimeError(error_msg)

            return ExperimentExecutionOutcome(result=result, duration=duration)
        except Exception as exc:
            duration = time.time() - start_time
            if not continue_on_failure:
                raise

            failed_result = ExperimentResult(
                experiment_name=experiment.name,
                success=False,
                duration=duration,
                error_message=str(exc),
            )
            return ExperimentExecutionOutcome(result=failed_result, duration=duration)
