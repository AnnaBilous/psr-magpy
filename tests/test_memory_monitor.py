"""Tests for memory monitoring system."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from psr_magpy.io.memory_monitor import (
    MemoryMonitor, MemoryInfo, MemoryCheckResult, MemoryThreshold
)
from psr_magpy.io.validator import DataIOValidator, ValidationLevel
from psr_magpy.grid.grid import SphericalGrid


class TestMemoryInfo:
    """Test MemoryInfo dataclass."""
    
    def test_memory_info_creation(self):
        """Test creating MemoryInfo with valid values."""
        info = MemoryInfo(
            total_mb=8192.0,
            available_mb=4096.0,
            used_mb=4096.0,
            percent_used=50.0,
            process_mb=100.0
        )
        
        assert info.total_mb == 8192.0
        assert info.available_mb == 4096.0
        assert info.used_mb == 4096.0
        assert info.percent_used == 50.0
        assert info.process_mb == 100.0
    
    def test_memory_info_repr(self):
        """Test MemoryInfo string representation."""
        info = MemoryInfo(
            total_mb=8192.0,
            available_mb=4096.0,
            used_mb=4096.0,
            percent_used=50.0,
            process_mb=100.0
        )
        
        repr_str = repr(info)
        assert "MemoryInfo" in repr_str
        assert "total_mb=8192.0" in repr_str
        assert "available_mb=4096.0" in repr_str


class TestMemoryCheckResult:
    """Test MemoryCheckResult dataclass."""
    
    def test_memory_check_result_creation(self):
        """Test creating MemoryCheckResult with valid values."""
        result = MemoryCheckResult(
            is_safe=True,
            available_mb=4096.0,
            required_mb=1000.0,
            threshold_mb=2048.0,
            safety_margin_mb=100.0,
            warnings=["Test warning"],
            details={"test": "value"}
        )
        
        assert result.is_safe is True
        assert result.available_mb == 4096.0
        assert result.required_mb == 1000.0
        assert result.threshold_mb == 2048.0
        assert result.safety_margin_mb == 100.0
        assert result.warnings == ["Test warning"]
        assert result.details == {"test": "value"}
    
    def test_memory_check_result_unsafe(self):
        """Test MemoryCheckResult for unsafe operation."""
        result = MemoryCheckResult(
            is_safe=False,
            available_mb=1000.0,
            required_mb=2000.0,
            threshold_mb=1500.0,
            safety_margin_mb=100.0,
            warnings=["Insufficient memory"],
            details={"reason": "required > available"}
        )
        
        assert result.is_safe is False
        assert result.required_mb > result.available_mb


class TestMemoryThreshold:
    """Test MemoryThreshold enum."""
    
    def test_threshold_values(self):
        """Test MemoryThreshold enum values."""
        assert MemoryThreshold.CONSERVATIVE.value == 0.25
        assert MemoryThreshold.MODERATE.value == 0.50
        assert MemoryThreshold.AGGRESSIVE.value == 0.75
        assert MemoryThreshold.MAXIMUM.value == 0.90
    
    def test_threshold_names(self):
        """Test MemoryThreshold enum names."""
        assert MemoryThreshold.CONSERVATIVE.name == "CONSERVATIVE"
        assert MemoryThreshold.MODERATE.name == "MODERATE"
        assert MemoryThreshold.AGGRESSIVE.name == "AGGRESSIVE"
        assert MemoryThreshold.MAXIMUM.name == "MAXIMUM"


class TestMemoryMonitor:
    """Test MemoryMonitor class."""
    
    def test_memory_monitor_initialization(self):
        """Test MemoryMonitor initialization with default parameters."""
        monitor = MemoryMonitor()
        
        assert monitor.threshold == MemoryThreshold.MODERATE
        assert monitor.safety_margin_mb == 100.0
        assert monitor.enable_warnings is True
        assert monitor.enable_checks is True
    
    def test_memory_monitor_custom_initialization(self):
        """Test MemoryMonitor initialization with custom parameters."""
        monitor = MemoryMonitor(
            threshold=MemoryThreshold.CONSERVATIVE,
            safety_margin_mb=200.0,
            enable_warnings=False,
            enable_checks=False
        )
        
        assert monitor.threshold == MemoryThreshold.CONSERVATIVE
        assert monitor.safety_margin_mb == 200.0
        assert monitor.enable_warnings is False
        assert monitor.enable_checks is False
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_get_memory_info_success(self, mock_psutil):
        """Test successful memory info retrieval."""
        # Mock psutil responses
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB in bytes
        mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        mock_memory.percent = 50.0
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB in bytes
        
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.Process.return_value = mock_process
        
        monitor = MemoryMonitor()
        info = monitor.get_memory_info()
        
        assert info.total_mb == 16384.0  # 16GB in MB
        assert info.available_mb == 8192.0  # 8GB in MB
        assert info.used_mb == 8192.0  # 8GB in MB
        assert info.percent_used == 50.0
        assert info.process_mb == 100.0
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_get_memory_info_exception(self, mock_psutil):
        """Test memory info retrieval with exception handling."""
        mock_psutil.virtual_memory.side_effect = Exception("psutil error")
        
        monitor = MemoryMonitor()
        info = monitor.get_memory_info()
        
        # Should return safe defaults
        assert info.total_mb == 8192.0
        assert info.available_mb == 4096.0
        assert info.used_mb == 4096.0
        assert info.percent_used == 50.0
        assert info.process_mb == 100.0
    
    def test_check_memory_safety_disabled(self):
        """Test memory safety check when disabled."""
        monitor = MemoryMonitor(enable_checks=False)
        result = monitor.check_memory_safety(1000.0, "test")
        
        assert result.is_safe is True
        assert result.details['checks_disabled'] is True
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_check_memory_safety_safe_operation(self, mock_psutil):
        """Test memory safety check for safe operation."""
        # Mock memory info with proper attributes
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB total
        mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB available
        mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB used
        mock_memory.percent = 50.0
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.Process.return_value = mock_process
        
        monitor = MemoryMonitor(threshold=MemoryThreshold.MODERATE)
        result = monitor.check_memory_safety(1000.0, "test")  # 1GB required
        
        assert result.is_safe is True
        assert result.available_mb == 8192.0
        assert result.required_mb == 1000.0
        assert result.threshold_mb == 4096.0  # 50% of 8GB
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_check_memory_safety_unsafe_operation(self, mock_psutil):
        """Test memory safety check for unsafe operation."""
        # Mock memory info with proper attributes
        mock_memory = Mock()
        mock_memory.total = 4 * 1024 * 1024 * 1024  # 4GB total
        mock_memory.available = 2 * 1024 * 1024 * 1024  # 2GB available
        mock_memory.used = 2 * 1024 * 1024 * 1024  # 2GB used
        mock_memory.percent = 50.0
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.Process.return_value = mock_process
        
        monitor = MemoryMonitor(threshold=MemoryThreshold.CONSERVATIVE)
        result = monitor.check_memory_safety(2000.0, "test")  # 2GB required
        
        assert result.is_safe is False
        assert result.available_mb == 2048.0
        assert result.required_mb == 2000.0
        assert result.threshold_mb == 512.0  # 25% of 2GB
    
    def test_estimate_data_size_numpy_array(self):
        """Test data size estimation for numpy arrays."""
        monitor = MemoryMonitor()
        
        # Small array
        small_array = np.zeros((100, 100), dtype=np.float64)
        size = monitor.estimate_data_size(small_array)
        expected_size = 100 * 100 * 8 / (1024 * 1024)  # 8 bytes per float64
        assert abs(size - expected_size) < 0.1
        
        # Large array
        large_array = np.zeros((1000, 1000), dtype=np.float32)
        size = monitor.estimate_data_size(large_array)
        expected_size = 1000 * 1000 * 4 / (1024 * 1024)  # 4 bytes per float32
        assert abs(size - expected_size) < 0.1
    
    def test_estimate_data_size_list(self):
        """Test data size estimation for lists."""
        monitor = MemoryMonitor()
        
        # List of arrays
        arrays = [np.zeros(100) for _ in range(5)]
        size = monitor.estimate_data_size(arrays)
        assert size > 0
        
        # Empty list
        size = monitor.estimate_data_size([])
        assert size >= 0.0  # Empty list might have small overhead
    
    def test_estimate_data_size_dict(self):
        """Test data size estimation for dictionaries."""
        monitor = MemoryMonitor()
        
        # Dictionary with data
        data_dict = {
            'array': np.zeros(100),
            'value': 42,
            'string': 'test'
        }
        size = monitor.estimate_data_size(data_dict)
        assert size > 0
    
    def test_estimate_data_size_none(self):
        """Test data size estimation for None."""
        monitor = MemoryMonitor()
        size = monitor.estimate_data_size(None)
        assert size == 0.0
    
    def test_estimate_file_size_existing(self):
        """Test file size estimation for existing file."""
        monitor = MemoryMonitor()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'x' * 1024)  # 1KB file
            tmp_path = tmp.name
        
        try:
            size = monitor.estimate_file_size(tmp_path)
            assert size == 1.0 / 1024  # 1KB in MB
        finally:
            Path(tmp_path).unlink()
    
    def test_estimate_file_size_nonexistent(self):
        """Test file size estimation for nonexistent file."""
        monitor = MemoryMonitor()
        size = monitor.estimate_file_size("nonexistent_file.fits")
        assert size == 50.0  # Default estimate
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_check_load_safety(self, mock_psutil):
        """Test load safety check."""
        # Mock memory info
        mock_memory = Mock()
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        
        monitor = MemoryMonitor()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'x' * 1024)  # 1KB file
            tmp_path = tmp.name
        
        try:
            result = monitor.check_load_safety(tmp_path, "load")
            assert result.is_safe is True
            assert result.required_mb > 0
        finally:
            Path(tmp_path).unlink()
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_check_save_safety(self, mock_psutil):
        """Test save safety check."""
        # Mock memory info
        mock_memory = Mock()
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        
        monitor = MemoryMonitor()
        
        # Create test data
        data = np.zeros((100, 100))
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            result = monitor.check_save_safety(data, tmp_path, "save")
            assert result.is_safe is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_check_batch_operation_safety(self):
        """Test batch operation safety check."""
        monitor = MemoryMonitor()
        
        # Safe batch operation
        result = monitor.check_batch_operation_safety(
            batch_size=10,
            item_size_mb=10.0,
            operation="batch"
        )
        assert result.is_safe is True
        assert result.required_mb == 100.0  # 10 * 10MB
        
        # Unsafe batch operation
        result = monitor.check_batch_operation_safety(
            batch_size=1000,
            item_size_mb=10.0,
            operation="batch"
        )
        assert result.is_safe is False
        assert result.required_mb == 10000.0  # 1000 * 10MB
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_get_memory_summary(self, mock_psutil):
        """Test memory summary generation."""
        # Mock memory info
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024
        mock_memory.available = 8 * 1024 * 1024 * 1024
        mock_memory.used = 8 * 1024 * 1024 * 1024
        mock_memory.percent = 50.0
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.Process.return_value = mock_process
        
        monitor = MemoryMonitor()
        summary = monitor.get_memory_summary()
        
        assert 'system_memory' in summary
        assert 'process_memory' in summary
        assert 'thresholds' in summary
        assert 'monitor_config' in summary
        
        system_memory = summary['system_memory']
        assert system_memory['total_mb'] == 16384.0
        assert system_memory['available_mb'] == 8192.0
        assert system_memory['used_mb'] == 8192.0
        assert system_memory['percent_used'] == 50.0
    
    def test_suggest_optimization_small_operation(self):
        """Test optimization suggestions for small operations."""
        monitor = MemoryMonitor()
        suggestions = monitor.suggest_optimization(50.0)  # 50MB
        assert len(suggestions) == 0  # No suggestions needed
    
    def test_suggest_optimization_large_operation(self):
        """Test optimization suggestions for large operations."""
        monitor = MemoryMonitor()
        suggestions = monitor.suggest_optimization(8000.0)  # 8GB
        
        assert len(suggestions) > 0
        assert any("batch" in suggestion.lower() for suggestion in suggestions)
        assert any("disk" in suggestion.lower() for suggestion in suggestions)


class TestMemoryMonitorIntegration:
    """Test memory monitor integration with I/O operations."""
    
    def test_memory_monitor_with_validator(self):
        """Test memory monitor integration with validator."""
        monitor = MemoryMonitor()
        validator = DataIOValidator(validation_level=ValidationLevel.BASIC)
        
        # Both should work together
        assert monitor is not None
        assert validator is not None
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_memory_monitor_with_grid_io(self, mock_psutil):
        """Test memory monitor with grid I/O operations."""
        # Mock memory info
        mock_memory = Mock()
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        
        monitor = MemoryMonitor()
        
        # Create a grid
        grid = SphericalGrid(nlon=10, nlat=5)
        
        # Check save safety
        result = monitor.check_save_safety(grid, "test_grid.fits", "save")
        assert result.is_safe is True
        
        # Check load safety
        result = monitor.check_load_safety("test_grid.fits", "load")
        assert result.is_safe is True
    
    def test_memory_monitor_threshold_levels(self):
        """Test different threshold levels."""
        # Conservative threshold
        conservative = MemoryMonitor(threshold=MemoryThreshold.CONSERVATIVE)
        assert conservative.threshold.value == 0.25
        
        # Moderate threshold
        moderate = MemoryMonitor(threshold=MemoryThreshold.MODERATE)
        assert moderate.threshold.value == 0.50
        
        # Aggressive threshold
        aggressive = MemoryMonitor(threshold=MemoryThreshold.AGGRESSIVE)
        assert aggressive.threshold.value == 0.75
        
        # Maximum threshold
        maximum = MemoryMonitor(threshold=MemoryThreshold.MAXIMUM)
        assert maximum.threshold.value == 0.90


class TestMemoryMonitorEdgeCases:
    """Test memory monitor edge cases and error conditions."""
    
    def test_memory_monitor_zero_safety_margin(self):
        """Test memory monitor with zero safety margin."""
        monitor = MemoryMonitor(safety_margin_mb=0.0)
        assert monitor.safety_margin_mb == 0.0
    
    def test_memory_monitor_large_safety_margin(self):
        """Test memory monitor with large safety margin."""
        monitor = MemoryMonitor(safety_margin_mb=10000.0)
        assert monitor.safety_margin_mb == 10000.0
    
    def test_estimate_data_size_complex_objects(self):
        """Test data size estimation for complex objects."""
        monitor = MemoryMonitor()
        
        # Test with object that has no obvious size
        class ComplexObject:
            def __init__(self):
                self.data = "test"
        
        obj = ComplexObject()
        size = monitor.estimate_data_size(obj)
        assert size >= 0.0  # Should not crash
    
    def test_estimate_data_size_nested_structures(self):
        """Test data size estimation for nested data structures."""
        monitor = MemoryMonitor()
        
        # Nested structure
        nested_data = {
            'level1': {
                'level2': {
                    'array': np.zeros((10, 10)),
                    'list': [np.zeros(5) for _ in range(3)]
                }
            }
        }
        
        size = monitor.estimate_data_size(nested_data)
        assert size > 0.0
    
    @patch('psr_magpy.io.memory_monitor.psutil')
    def test_memory_monitor_very_large_operation(self, mock_psutil):
        """Test memory monitor with very large operation."""
        # Mock very limited memory with proper attributes
        mock_memory = Mock()
        mock_memory.total = 2 * 1024 * 1024 * 1024  # 2GB total
        mock_memory.available = 1 * 1024 * 1024 * 1024  # 1GB available
        mock_memory.used = 1 * 1024 * 1024 * 1024  # 1GB used
        mock_memory.percent = 50.0
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.Process.return_value = mock_process
        
        monitor = MemoryMonitor(threshold=MemoryThreshold.CONSERVATIVE)
        
        # Very large operation
        result = monitor.check_memory_safety(10000.0, "very_large")  # 10GB
        
        assert result.is_safe is False
        assert result.required_mb == 10000.0
        assert result.threshold_mb == 256.0  # 25% of 1GB 