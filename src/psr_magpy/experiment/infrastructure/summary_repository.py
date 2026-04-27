"""Repository for reading/writing experiment series summary files."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
import glob
import time
import yaml

from psr_magpy.experiment.experiment_result import ExperimentResult
from psr_magpy.experiment.services.formatting import format_duration
from psr_magpy.experiment.infrastructure.path_policy import (
    lines_dir,
    results_summary_filename,
    series_dir,
)


class SummaryRepository:
    """Persistence adapter for series run summaries."""

    def summary_path(self, output_dir: Optional[str], series_name: str) -> Path:
        """Return canonical summary file path."""
        return series_dir(output_dir, series_name) / results_summary_filename(series_name)

    def load_summary(self, output_dir: Optional[str], series_name: str) -> Dict[str, Any]:
        """Load existing summary dictionary, or empty dict if missing."""
        path = self.summary_path(output_dir, series_name)
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def save_summary(
        self, output_dir: Optional[str], series_name: str, summary: Dict[str, Any]
    ) -> Path:
        """Write summary dictionary to YAML preserving order."""
        path = self.summary_path(output_dir, series_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        class OrderedDumper(yaml.SafeDumper):
            pass

        def _dict_representer(
            dumper: yaml.SafeDumper, data: OrderedDict
        ) -> yaml.nodes.MappingNode:
            return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

        OrderedDumper.add_representer(OrderedDict, _dict_representer)

        with open(path, "w") as f:
            yaml.dump(summary, f, Dumper=OrderedDumper, sort_keys=False, width=1000)
        return path

    def merge_summary(
        self,
        *,
        output_dir: Optional[str],
        series_name: str,
        experiment_results: List[ExperimentResult],
    ) -> Dict[str, Any]:
        """Merge new experiment results into existing summary, preserving current semantics."""
        existing_summary = self.load_summary(output_dir, series_name)
        existing_results = {
            r["name"]: r for r in existing_summary.get("experiment_results", [])
        }

        lines_directory = lines_dir(output_dir, series_name)

        for result in experiment_results:
            batch_pattern = str(
                lines_directory / f"{result.experiment_name}*batch_*_out_of_*.fits"
            )
            batch_files = glob.glob(batch_pattern)

            result_dict: Dict[str, Any] = {
                "name": result.experiment_name,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": format_duration(result.duration),
                "success": result.success,
                "batch_files": len(batch_files),
            }
            if result.error_message:
                result_dict["error"] = result.error_message

            ordered_items = [
                ("name", result_dict["name"]),
                ("start_time", result_dict["start_time"]),
                ("duration", result_dict["duration"]),
                ("success", result_dict["success"]),
            ]
            if "error" in result_dict:
                ordered_items.append(("error", result_dict["error"]))
            ordered_items.append(("batch_files", result_dict["batch_files"]))
            ordered_result = OrderedDict(ordered_items)

            if result.experiment_name in existing_results:
                existing_results[result.experiment_name].update(ordered_result)
            else:
                existing_results[result.experiment_name] = ordered_result

        sorted_results = sorted(existing_results.values(), key=lambda x: x["name"])

        total_seconds = sum(self._parse_duration_seconds(r["duration"]) for r in sorted_results)
        successful = sum(1 for r in sorted_results if r["success"])
        total = len(sorted_results)

        return {
            "series_name": series_name,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": total,
            "successful_experiments": successful,
            "failed_experiments": total - successful,
            "total_duration": format_duration(total_seconds),
            "success_rate": (successful / total * 100) if total else 0.0,
            "experiment_results": sorted_results,
        }

    @staticmethod
    def _parse_duration_seconds(duration: str) -> float:
        """Parse duration formats currently used in summary files."""
        if duration.endswith("s"):
            return float(duration.rstrip("s"))
        if duration.endswith("min"):
            return float(duration.rstrip("min")) * 60
        if duration.endswith("h"):
            return float(duration.rstrip("h")) * 3600
        if duration.endswith("d"):
            return float(duration.rstrip("d")) * 86400
        return float(duration)
