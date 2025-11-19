"""Memory monitoring system for I/O operations.

This module provides comprehensive memory monitoring for all I/O operations
with configurable thresholds and safety checks.
"""

import psutil
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, List
from dataclasses import dataclass
from enum import Enum

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class MemoryThreshold(Enum):
    """Memory threshold levels for different operations."""

    CONSERVATIVE = 0.25  # 25% of available memory
    MODERATE = 0.50  # 50% of available memory
    AGGRESSIVE = 0.75  # 75% of available memory
    MAXIMUM = 0.90  # 90% of available memory


@dataclass
class MemoryInfo:
    """Information about current memory usage."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_mb: float


@dataclass
class MemoryCheckResult:
    """Result of a memory check operation."""

    is_safe: bool
    available_mb: float
    required_mb: float
    threshold_mb: float
    safety_margin_mb: float
    warnings: list
    details: Dict[str, Any]


class MemoryMonitor:
    """Comprehensive memory monitoring for I/O operations."""

    def __init__(
        self,
        threshold: MemoryThreshold = MemoryThreshold.MODERATE,
        safety_margin_mb: float = 100.0,
        enable_warnings: bool = True,
        enable_checks: bool = True,
    ):
        """Initialize memory monitor with configuration."""
        self.threshold = threshold
        self.safety_margin_mb = safety_margin_mb
        self.enable_warnings = enable_warnings
        self.enable_checks = enable_checks

        logger.debug(
            f"Initialized MemoryMonitor: threshold={threshold.value}, "
            f"safety_margin={safety_margin_mb}MB, checks={enable_checks}"
        )

    def get_memory_info(self) -> MemoryInfo:
        """Get current memory information."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            total_mb = memory.total / (1024 * 1024)
            available_mb = memory.available / (1024 * 1024)
            used_mb = memory.used / (1024 * 1024)
            percent_used = memory.percent

            # Process memory
            process = psutil.Process()
            process_mb = process.memory_info().rss / (1024 * 1024)

            return MemoryInfo(
                total_mb=total_mb,
                available_mb=available_mb,
                used_mb=used_mb,
                percent_used=percent_used,
                process_mb=process_mb,
            )
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            # Return safe defaults
            return MemoryInfo(
                total_mb=8192.0,  # 8GB default
                available_mb=4096.0,  # 4GB default
                used_mb=4096.0,
                percent_used=50.0,
                process_mb=100.0,
            )

    def check_memory_safety(
        self, required_mb: float, operation: str = "unknown", **kwargs
    ) -> MemoryCheckResult:
        """Check if memory operation is safe."""
        if not self.enable_checks:
            return MemoryCheckResult(
                is_safe=True,
                available_mb=0.0,
                required_mb=required_mb,
                threshold_mb=0.0,
                safety_margin_mb=self.safety_margin_mb,
                warnings=[],
                details={"checks_disabled": True},
            )

        try:
            memory_info = self.get_memory_info()
            threshold_mb = memory_info.available_mb * self.threshold.value
            effective_threshold = threshold_mb - self.safety_margin_mb

            warnings = []
            details = {
                "operation": operation,
                "memory_info": memory_info,
                "threshold_level": self.threshold.value,
                "safety_margin_mb": self.safety_margin_mb,
            }

            # Check if operation is safe
            is_safe = required_mb <= effective_threshold

            # Generate warnings
            if self.enable_warnings:
                if memory_info.percent_used > 80:
                    warnings.append(
                        f"High system memory usage: {memory_info.percent_used:.1f}%"
                    )

                if memory_info.process_mb > 1000:
                    warnings.append(
                        f"High process memory usage: {memory_info.process_mb:.1f}MB"
                    )

                if required_mb > threshold_mb * 0.8:
                    warnings.append(
                        f"Operation requires {required_mb:.1f}MB, approaching threshold {threshold_mb:.1f}MB"
                    )

            logger.debug(
                f"Memory check for {operation}: required={required_mb:.1f}MB, "
                f"available={memory_info.available_mb:.1f}MB, threshold={threshold_mb:.1f}MB, safe={is_safe}"
            )

            return MemoryCheckResult(
                is_safe=is_safe,
                available_mb=memory_info.available_mb,
                required_mb=required_mb,
                threshold_mb=threshold_mb,
                safety_margin_mb=self.safety_margin_mb,
                warnings=warnings,
                details=details,
            )

        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return MemoryCheckResult(
                is_safe=False,
                available_mb=0.0,
                required_mb=required_mb,
                threshold_mb=0.0,
                safety_margin_mb=self.safety_margin_mb,
                warnings=[f"Memory check failed: {e}"],
                details={"error": str(e)},
            )

    def estimate_data_size(self, data: Any) -> float:
        """Estimate the size of data in MB."""
        try:
            if data is None:
                return 0.0

            # Use built-in size estimation if available
            if hasattr(data, "estimate_size"):
                size_bytes_est = data.estimate_size()
                return float(size_bytes_est) / (1024 * 1024)

            # Use nbytes for numpy arrays
            if hasattr(data, "nbytes"):
                return float(data.nbytes) / (1024 * 1024)

            # Use getsizeof for other objects
            import sys

            size_bytes: int = sys.getsizeof(data)

            # For containers, estimate recursively
            if hasattr(data, "__len__") and len(data) > 0:
                if isinstance(data, (list, tuple)):
                    # Estimate based on first few elements
                    sample_size = min(10, len(data))
                    sample_bytes = sum(
                        sys.getsizeof(item) for item in data[:sample_size]
                    )
                    estimated_bytes = float(sample_bytes * len(data)) / sample_size
                    return estimated_bytes / (1024 * 1024)
                elif isinstance(data, dict):
                    # Estimate based on key-value pairs
                    total_bytes = sum(
                        sys.getsizeof(k) + sys.getsizeof(v) for k, v in data.items()
                    )
                    return float(total_bytes) / (1024 * 1024)

            return float(size_bytes) / (1024 * 1024)

        except Exception as e:
            logger.warning(f"Could not estimate data size: {e}")
            return 100.0  # Conservative default

    def estimate_file_size(self, path: Union[str, Path]) -> float:
        """Estimate the size of a file in MB."""
        try:
            path_obj = Path(path)
            if path_obj.exists():
                return path_obj.stat().st_size / (1024 * 1024)
            else:
                # Estimate based on file extension and typical sizes
                suffix = path_obj.suffix.lower()
                if suffix == ".fits":
                    return 50.0  # Conservative estimate for FITS files
                elif suffix in [".yaml", ".yml"]:
                    return 0.1  # Small for config files
                else:
                    return 10.0  # Default estimate
        except Exception as e:
            logger.warning(f"Could not estimate file size: {e}")
            return 10.0  # Conservative default

    def check_load_safety(
        self, path: Union[str, Path], operation: str = "load", **kwargs
    ) -> MemoryCheckResult:
        """Check if loading a file is memory-safe."""
        file_size_mb = self.estimate_file_size(path)

        # For loading, we typically need 2-3x the file size in memory
        # (original data + processing overhead + temporary objects)
        required_mb = file_size_mb * 3.0

        return self.check_memory_safety(required_mb, operation, **kwargs)

    def check_save_safety(
        self, data: Any, path: Union[str, Path], operation: str = "save", **kwargs
    ) -> MemoryCheckResult:
        """Check if saving data is memory-safe."""
        data_size_mb = self.estimate_data_size(data)

        # For saving, we need the data size plus some overhead for processing
        required_mb = data_size_mb * 1.5

        return self.check_memory_safety(required_mb, operation, **kwargs)

    def check_batch_operation_safety(
        self, batch_size: int, item_size_mb: float, operation: str = "batch", **kwargs
    ) -> MemoryCheckResult:
        """Check if a batch operation is memory-safe."""
        total_required_mb = batch_size * item_size_mb

        return self.check_memory_safety(total_required_mb, operation, **kwargs)

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of current memory status."""
        memory_info = self.get_memory_info()

        return {
            "system_memory": {
                "total_mb": memory_info.total_mb,
                "available_mb": memory_info.available_mb,
                "used_mb": memory_info.used_mb,
                "percent_used": memory_info.percent_used,
            },
            "process_memory": {"current_mb": memory_info.process_mb},
            "thresholds": {
                "level": self.threshold.value,
                "threshold_mb": memory_info.available_mb * self.threshold.value,
                "safety_margin_mb": self.safety_margin_mb,
            },
            "monitor_config": {
                "enable_checks": self.enable_checks,
                "enable_warnings": self.enable_warnings,
            },
        }

    def log_memory_status(self, operation: str = "unknown") -> None:
        """Log current memory status."""
        memory_info = self.get_memory_info()
        threshold_mb = memory_info.available_mb * self.threshold.value

        logger.info(f"Memory status for {operation}:")
        logger.info(
            f"  System: {memory_info.used_mb:.1f}MB used / {memory_info.total_mb:.1f}MB total ({memory_info.percent_used:.1f}%)"
        )
        logger.info(f"  Available: {memory_info.available_mb:.1f}MB")
        logger.info(f"  Process: {memory_info.process_mb:.1f}MB")
        logger.info(
            f"  Threshold: {threshold_mb:.1f}MB (level: {self.threshold.value})"
        )

    def suggest_optimization(self, required_mb: float) -> List[str]:
        """Suggest optimizations for memory usage."""
        suggestions: List[str] = []
        memory_info = self.get_memory_info()

        # Only suggest optimizations for operations requiring more than 100MB
        if required_mb < 100:
            return suggestions

        if required_mb > memory_info.available_mb * 0.5:
            suggestions.append("Consider processing data in smaller batches")

        if memory_info.percent_used > 80:
            suggestions.append(
                "System memory usage is high - consider closing other applications"
            )

        if memory_info.process_mb > 1000:
            suggestions.append(
                "Process memory usage is high - consider garbage collection"
            )

        if required_mb > memory_info.available_mb:
            suggestions.append(
                "Required memory exceeds available memory - consider disk-based processing"
            )

        return suggestions
