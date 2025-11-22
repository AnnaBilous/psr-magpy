"""Test robustness improvements for batch file management."""

import pytest
import tempfile
import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from psr_magpy.io.batch_file_manager import BatchFileManager, BatchFileInfo


class TestBatchFileManagerRobustness:
    """Test robustness features of BatchFileManager."""

    @pytest.fixture(autouse=True)
    def setup_test_data_dir(self, test_data_dir, request):
        """Set up test fixtures."""
        import shutil
        # Use test name to create unique directory per test
        test_name = request.node.name
        self.temp_dir = test_data_dir / "batch_file_manager_robustness" / test_name
        # Clean up any existing directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.base_filename = self.temp_dir / "test_experiment.fits"
        yield
        # Clean up after test
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_sample_batch_files(self, n_batches=3):
        """Create sample batch files for testing."""
        from astropy.io import fits
        
        for i in range(n_batches):
            batch_filename = f"{self.base_filename.stem}_batch_{i:03d}.fits"
            batch_path = Path(self.temp_dir) / batch_filename
            
            # Create simple FITS file
            primary_hdu = fits.PrimaryHDU()
            data_hdu = fits.BinTableHDU.from_columns([
                fits.Column(name='test', format='D', array=[1.0, 2.0, 3.0])
            ])
            hdul = fits.HDUList([primary_hdu, data_hdu])
            hdul.writeto(batch_path, overwrite=True)
            hdul.close()

    def test_filesystem_health_check_success(self):
        """Test filesystem health check when everything is OK."""
        batch_manager = BatchFileManager(str(self.base_filename))
        
        # Should pass for a normal directory
        assert batch_manager._check_filesystem_health(Path(self.temp_dir)) is True

    def test_filesystem_health_check_nonexistent_directory(self):
        """Test filesystem health check with nonexistent directory."""
        batch_manager = BatchFileManager(str(self.base_filename))
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        assert batch_manager._check_filesystem_health(nonexistent_dir) is False

    @patch('os.access')
    def test_filesystem_health_check_not_writable(self, mock_access):
        """Test filesystem health check when directory is not writable."""
        mock_access.return_value = False
        batch_manager = BatchFileManager(str(self.base_filename))
        
        assert batch_manager._check_filesystem_health(Path(self.temp_dir)) is False

    @patch('shutil.disk_usage')
    def test_filesystem_health_check_insufficient_space(self, mock_disk_usage):
        """Test filesystem health check with insufficient disk space."""
        # Mock disk usage to return very little free space
        mock_disk_usage.return_value = MagicMock(free=50*1024*1024)  # 50MB free
        batch_manager = BatchFileManager(str(self.base_filename))
        
        assert batch_manager._check_filesystem_health(Path(self.temp_dir)) is False

    def test_organize_batch_files_with_retry_logic(self):
        """Test that organize_batch_files uses retry logic."""
        self.create_sample_batch_files(2)
        
        batch_manager = BatchFileManager(str(self.base_filename))
        
        # Should succeed normally
        result = batch_manager.organize_batch_files()
        assert result is not None

    @patch('shutil.move')
    def test_organize_batch_files_retry_on_failure(self, mock_move):
        """Test retry logic when move operations fail."""
        self.create_sample_batch_files(1)
        
        # Make move fail twice, then succeed
        mock_move.side_effect = [OSError("Permission denied"), OSError("File locked"), None]
        
        batch_manager = BatchFileManager(str(self.base_filename))
        
        # Should eventually succeed
        result = batch_manager.organize_batch_files()
        assert result is not None
        assert mock_move.call_count == 3

    @patch('shutil.move')
    def test_organize_batch_files_fallback_on_persistent_failure(self, mock_move):
        """Test fallback behavior when move operations persistently fail."""
        self.create_sample_batch_files(1)
        
        # Make move always fail
        mock_move.side_effect = OSError("Permission denied")
        
        batch_manager = BatchFileManager(str(self.base_filename))
        
        # Should raise PSRMagPyIOError on persistent failure
        from psr_magpy.utils.exceptions import PSRMagPyIOError
        with pytest.raises(PSRMagPyIOError, match="Failed to organize batch files"):
            batch_manager.organize_batch_files()

    def test_organize_batch_files_handles_missing_source(self):
        """Test handling of missing source files during organization."""
        self.create_sample_batch_files(1)
        
        batch_manager = BatchFileManager(str(self.base_filename))
        
        # Delete the source file before organization
        batch_files = batch_manager.find_batch_files()
        if batch_files:
            batch_files[0].path.unlink()
        
        # Should handle gracefully
        result = batch_manager.organize_batch_files()
        # Should return None since no files to organize
        assert result is None

    def test_organize_batch_files_handles_existing_destination(self):
        """Test handling of existing destination files."""
        self.create_sample_batch_files(1)
        
        batch_manager = BatchFileManager(str(self.base_filename))
        
        # Create a destination file that would conflict
        output_dir = Path(self.temp_dir)
        conflict_file = output_dir / f"{self.base_filename.stem}_batch_001_out_of_001.fits"
        conflict_file.write_text("conflict")
        
        # Should handle gracefully by removing the conflict and replacing with actual data
        result = batch_manager.organize_batch_files()
        assert result is not None
        
        # Check that organized files exist
        organized_files = list(output_dir.glob(f"{self.base_filename.stem}_batch_*.fits"))
        assert len(organized_files) > 0
        
        # Find the valid FITS file (not the conflict text file)
        valid_fits_files = []
        from astropy.io import fits
        for file_path in organized_files:
            try:
                with fits.open(file_path) as hdul:
                    if len(hdul) > 0:
                        valid_fits_files.append(file_path)
            except OSError:
                # Skip invalid FITS files (like our conflict file)
                continue
        
        # Should have at least one valid FITS file
        assert len(valid_fits_files) > 0


if __name__ == "__main__":
    pytest.main([__file__])
