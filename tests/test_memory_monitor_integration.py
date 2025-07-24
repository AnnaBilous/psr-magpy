"""Integration tests for memory monitoring with I/O operations."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import os
import time

from psr_magpy.io.memory_monitor import MemoryMonitor, MemoryThreshold
from psr_magpy.io.validator import DataIOValidator, ValidationLevel
from psr_magpy.io.factory import IOFactory
from psr_magpy.grid.grid import SphericalGrid
from psr_magpy.experiment.experiment_config import ExperimentConfig


class TestMemoryMonitorIOIntegration:
    """Test memory monitoring integration with I/O operations."""
    
    def test_memory_monitor_with_grid_save_load(self):
        """Test memory monitoring with grid save/load operations."""
        monitor = MemoryMonitor(threshold=MemoryThreshold.MODERATE)
        
        # Create a grid
        grid = SphericalGrid(nlon=20, nlat=10)
        
        # Check save safety before saving
        save_result = monitor.check_save_safety(grid, "test_grid.fits", "save")
        assert save_result.is_safe is True
        
        # Save grid with memory monitoring
        grid_path = Path(tempfile.mktemp(suffix=".fits"))
        
        try:
            # Save with memory monitoring
            IOFactory.save_data(grid, grid_path)
            
            # Check load safety before loading
            load_result = monitor.check_load_safety(grid_path, "load")
            assert load_result.is_safe is True
            
            # Load with memory monitoring
            loaded_grid = IOFactory.load_data(type(grid), grid_path)
            
            # Verify round-trip
            assert loaded_grid.nline == grid.nline
            assert loaded_grid.grid_type == grid.grid_type
            
        finally:
            grid_path.unlink(missing_ok=True)
    
    def test_memory_monitor_with_config_save_load(self):
        """Test memory monitoring with configuration save/load operations."""
        monitor = MemoryMonitor(threshold=MemoryThreshold.MODERATE)
        
        # Create a configuration
        config = ExperimentConfig.create_dipole_example()
        
        # Check save safety before saving
        save_result = monitor.check_save_safety(config, "test_config.yaml", "save")
        assert save_result.is_safe is True  # Configs are small
        
        # Generate a unique config path (do not pre-create the file)
        config_path = Path(tempfile.gettempdir()) / f"test_config_{os.getpid()}.yaml"
        
        try:
            # Save with memory monitoring
            IOFactory.save_config(config, config_path)
            
            # Ensure file is written and non-empty
            assert config_path.exists()
            for _ in range(10):
                if config_path.stat().st_size > 0:
                    break
                time.sleep(0.05)
            assert config_path.stat().st_size > 0
            
            # Load with memory monitoring
            loaded_config = IOFactory.load_config(ExperimentConfig, config_path)
            
            # Verify round-trip
            assert loaded_config.name == config.name
            assert loaded_config.grid.model_dump() == config.grid.model_dump()
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_memory_monitor_batch_operations(self):
        """Test memory monitoring with batch operations."""
        monitor = MemoryMonitor(threshold=MemoryThreshold.CONSERVATIVE)
        
        # Test safe batch operation
        safe_result = monitor.check_batch_operation_safety(
            batch_size=10,
            item_size_mb=10.0,
            operation="batch"
        )
        assert safe_result.is_safe is True
        assert safe_result.required_mb == 100.0
        
        # Test unsafe batch operation
        unsafe_result = monitor.check_batch_operation_safety(
            batch_size=1000,
            item_size_mb=10.0,
            operation="batch"
        )
        assert unsafe_result.is_safe is False
        assert unsafe_result.required_mb == 10000.0
    
    def test_memory_monitor_threshold_levels_integration(self):
        """Test different threshold levels with real operations."""
        # Create test data
        large_array = np.zeros((1000, 1000), dtype=np.float64)
        estimated_size = large_array.nbytes / (1024 * 1024)  # Size in MB
        
        # Test conservative threshold
        conservative_monitor = MemoryMonitor(threshold=MemoryThreshold.CONSERVATIVE)
        conservative_result = conservative_monitor.check_save_safety(
            large_array, "test.fits", "save"
        )
        
        # Test moderate threshold
        moderate_monitor = MemoryMonitor(threshold=MemoryThreshold.MODERATE)
        moderate_result = moderate_monitor.check_save_safety(
            large_array, "test.fits", "save"
        )
        
        # Test aggressive threshold
        aggressive_monitor = MemoryMonitor(threshold=MemoryThreshold.AGGRESSIVE)
        aggressive_result = aggressive_monitor.check_save_safety(
            large_array, "test.fits", "save"
        )
        
        # Conservative should be most restrictive
        if conservative_result.is_safe:
            assert moderate_result.is_safe
            assert aggressive_result.is_safe
        elif moderate_result.is_safe:
            assert aggressive_result.is_safe
    
    def test_memory_monitor_with_validator_integration(self):
        """Test memory monitor integration with validator."""
        monitor = MemoryMonitor()
        validator = DataIOValidator(validation_level=ValidationLevel.STANDARD)
        
        # Create test data
        grid = SphericalGrid(nlon=10, nlat=5)
        
        # Both should work together without conflicts
        memory_info = monitor.get_memory_info()
        validation_result = validator.validate_data(grid)
        
        assert memory_info.total_mb > 0
        assert memory_info.available_mb > 0
        assert validation_result.is_valid is True
    
    def test_memory_monitor_optimization_suggestions_integration(self):
        """Test optimization suggestions with real data."""
        monitor = MemoryMonitor()
        
        # Small operation - no suggestions needed
        small_grid = SphericalGrid(nlon=5, nlat=3)
        small_suggestions = monitor.suggest_optimization(50.0)  # 50MB
        assert len(small_suggestions) == 0
        
        # Large operation - should suggest optimizations
        large_suggestions = monitor.suggest_optimization(8000.0)  # 8GB
        assert len(large_suggestions) > 0
        
        # Check for specific suggestion types
        suggestion_text = " ".join(large_suggestions).lower()
        assert "batch" in suggestion_text or "disk" in suggestion_text
    
    def test_memory_monitor_error_handling_integration(self):
        """Test memory monitor error handling with real operations."""
        monitor = MemoryMonitor()
        
        # Test with invalid file path
        invalid_path = "/nonexistent/path/file.fits"
        result = monitor.check_load_safety(invalid_path, "load")
        
        # Should handle gracefully and provide reasonable estimate
        assert result.required_mb == 150.0  # Default estimate for nonexistent files
        
        # Test with None data
        result = monitor.check_save_safety(None, "test.fits", "save")
        # Note: Memory monitor doesn't validate data content, only size
        # The actual validation happens in the I/O system
        assert result.required_mb == 0.0  # None data has 0 size
    
    def test_memory_monitor_summary_integration(self):
        """Test memory summary with real system information."""
        monitor = MemoryMonitor()
        summary = monitor.get_memory_summary()
        
        # Check summary structure
        assert 'system_memory' in summary
        assert 'process_memory' in summary
        assert 'thresholds' in summary
        assert 'monitor_config' in summary
        
        # Check system memory values are reasonable
        system_memory = summary['system_memory']
        assert system_memory['total_mb'] > 0
        assert system_memory['available_mb'] > 0
        assert system_memory['used_mb'] > 0
        assert 0 <= system_memory['percent_used'] <= 100
        
        # Check process memory
        process_memory = summary['process_memory']
        assert process_memory['current_mb'] > 0
        
        # Check thresholds
        thresholds = summary['thresholds']
        assert thresholds['level'] == 0.5  # MODERATE default
        assert thresholds['threshold_mb'] > 0
        assert thresholds['safety_margin_mb'] == 100.0
        
        # Check monitor config
        monitor_config = summary['monitor_config']
        assert monitor_config['enable_checks'] is True
        assert monitor_config['enable_warnings'] is True


class TestMemoryMonitorPerformance:
    """Test memory monitor performance characteristics."""
    
    def test_memory_monitor_initialization_performance(self):
        """Test memory monitor initialization performance."""
        import time
        
        start_time = time.time()
        monitor = MemoryMonitor()
        init_time = time.time() - start_time
        
        # Initialization should be fast (< 100ms)
        assert init_time < 0.1
    
    def test_memory_info_retrieval_performance(self):
        """Test memory info retrieval performance."""
        import time
        
        monitor = MemoryMonitor()
        
        # Test multiple retrievals
        times = []
        for _ in range(10):
            start_time = time.time()
            info = monitor.get_memory_info()
            times.append(time.time() - start_time)
        
        # Each retrieval should be fast (< 50ms)
        avg_time = sum(times) / len(times)
        assert avg_time < 0.05
        assert all(t < 0.1 for t in times)  # No single retrieval should be too slow
    
    def test_data_size_estimation_performance(self):
        """Test data size estimation performance."""
        import time
        
        monitor = MemoryMonitor()
        
        # Test with different data types
        test_data = [
            np.zeros((100, 100)),
            np.zeros((1000, 1000)),
            [np.zeros(100) for _ in range(10)],
            {"array": np.zeros(100), "value": 42}
        ]
        
        for data in test_data:
            start_time = time.time()
            size = monitor.estimate_data_size(data)
            estimation_time = time.time() - start_time
            
            # Estimation should be very fast (< 10ms)
            assert estimation_time < 0.01
            assert size >= 0.0 