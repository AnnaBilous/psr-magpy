"""Result types for experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, List
from psr_magpy.experiment.services.formatting import format_duration


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
    experiments_in_this_session: int
    successful_experiments: int
    failed_experiments: int
    total_duration: float
    experiment_results: List[ExperimentResult]
    output_directory: Path

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in appropriate units.

        Parameters
        ----------
        seconds : float
            Duration in seconds.

        Returns
        -------
        str
            Formatted string like "10.5s", "2.5min", "1.2h", or "3.5d".
        """
        return format_duration(seconds)

    def print_summary(
        self,
        include_details: bool = False,
        max_experiments: int = 5,
        format_duration: bool = True,
    ) -> None:
        """Print formatted series result summary.

        Parameters
        ----------
        include_details : bool, optional
            If True, include individual experiment results. Default is False.
        max_experiments : int, optional
            Maximum number of experiments to show in details. Default is 5.
        format_duration : bool, optional
            If True, format duration in human-readable units. Default is True.
        """
        print("✓ Experiment series completed!")
        print(f"\nSeries Results Summary:")
        print(f"  Series name: {self.series_name}")
        print(f"  Total experiments: {self.total_experiments}")
        print(f"  Successful: {self.successful_experiments}")
        print(f"  Failed: {self.failed_experiments}")

        # Calculate success rate safely
        if self.total_experiments > 0:
            success_rate = (
                self.successful_experiments / self.total_experiments * 100
            )
            print(f"  Success rate: {success_rate:.1f}%")
        else:
            print(f"  Success rate: N/A (no experiments)")

        # Format duration
        if format_duration:
            duration_str = self._format_duration(self.total_duration)
        else:
            duration_str = f"{self.total_duration:.2f}s"
        print(f"  Total duration: {duration_str}")
        print(f"  Output directory: {self.output_directory}")

        # Include details if requested
        if include_details and self.experiment_results:
            print(f"\nIndividual Experiment Results:")
            shown_count = min(len(self.experiment_results), max_experiments)
            for i, exp_result in enumerate(self.experiment_results[:shown_count], 1):
                status = "✓" if exp_result.success else "✗"
                if format_duration:
                    exp_duration = self._format_duration(exp_result.duration)
                else:
                    exp_duration = f"{exp_result.duration:.2f}s"
                print(
                    f"  {i}. {status} {exp_result.experiment_name} ({exp_duration})"
                )
                if not exp_result.success and exp_result.error_message:
                    print(f"     Error: {exp_result.error_message}")

            if len(self.experiment_results) > shown_count:
                remaining = len(self.experiment_results) - shown_count
                print(f"  ... and {remaining} more experiment(s)")

            # Duration statistics for successful experiments
            successful_durations = [
                r.duration for r in self.experiment_results if r.success
            ]
            if successful_durations:
                avg_duration = sum(successful_durations) / len(successful_durations)
                min_duration = min(successful_durations)
                max_duration = max(successful_durations)
                print(f"\n  Duration statistics (successful experiments):")
                if format_duration:
                    print(f"    Average: {self._format_duration(avg_duration)}")
                    print(f"    Min: {self._format_duration(min_duration)}")
                    print(f"    Max: {self._format_duration(max_duration)}")
                else:
                    print(f"    Average: {avg_duration:.2f}s")
                    print(f"    Min: {min_duration:.2f}s")
                    print(f"    Max: {max_duration:.2f}s")

    def __str__(self) -> str:
        """Return concise string representation."""
        if self.total_experiments > 0:
            success_rate = (
                self.successful_experiments / self.total_experiments * 100
            )
            success_str = f"{success_rate:.1f}%"
        else:
            success_str = "N/A"
        duration_str = self._format_duration(self.total_duration)
        return (
            f"SeriesResult({self.series_name}: "
            f"{self.successful_experiments}/{self.total_experiments} successful "
            f"({success_str}), duration: {duration_str})"
        )
