"""Result types for experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, List


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    experiment_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    results: Optional[Any] = None


@dataclass
class SeriesResult:
    """Result of an experiment series."""

    series_name: str
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    total_duration: float
    experiment_results: List[ExperimentResult]
    output_directory: Path
