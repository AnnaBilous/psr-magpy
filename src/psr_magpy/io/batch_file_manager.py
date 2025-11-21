"""Batch file management for organizing and cleaning up batch files.

This module provides functionality to organize batch files into a clean
directory structure and optionally clean up temporary batch files.
"""

import glob
import shutil
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError
from psr_magpy.line.progress_tracker import ProgressReporter

logger = get_logger(__name__, console_format_brief=True)


@dataclass
class BatchFileInfo:
    """Information about a batch file."""

    path: Path
    batch_index: int
    file_size_mb: float
    creation_time: float


class BatchFileManager:
    """Manages batch file organization and cleanup."""

    def __init__(
        self,
        base_filename: str,
        output_directory: Optional[Path] = None,
        organize_files: bool = True,
        cleanup_batch_files: bool = False,
        progress_reporter: Optional[ProgressReporter] = None,
    ):
        """Initialize batch file manager."""
        self.base_filename = Path(base_filename)
        self.output_directory = output_directory or self.base_filename.parent
        self.organize_files = organize_files
        self._cleanup_enabled = cleanup_batch_files
        self.progress_reporter = progress_reporter

        logger.debug(
            f"Initialized BatchFileManager: base={self.base_filename}, "
            f"organize={organize_files}, cleanup={self._cleanup_enabled}"
        )

    def find_batch_files(self) -> List[BatchFileInfo]:
        """Find all batch files for the base filename."""
        pattern = str(
            self.output_directory / f"{self.base_filename.stem}*_batch_*.fits"
        )
        batch_files = glob.glob(pattern)

        if not batch_files:
            logger.warning(f"No batch files found matching pattern: {pattern}")
            return []

        batch_infos = []
        for file_path in sorted(batch_files):
            path = Path(file_path)
            try:
                # Extract batch index from filename
                filename = path.name
                batch_part = filename.split("_batch_")[1].split(".")[0]
                # Handle both formats: "001" and "001_out_of_004"
                batch_index = int(batch_part.split("_")[0])

                # Get file info
                stat = path.stat()
                file_size_mb = stat.st_size / (1024 * 1024)
                creation_time = stat.st_ctime

                batch_info = BatchFileInfo(
                    path=path,
                    batch_index=batch_index,
                    file_size_mb=file_size_mb,
                    creation_time=creation_time,
                )
                batch_infos.append(batch_info)

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse batch index from {path}: {e}")
                continue

        logger.info(f"Found {len(batch_infos)} batch files")
        return batch_infos

    def _check_filesystem_health(self, directory: Path) -> bool:
        """Check if the filesystem is healthy for file operations."""
        try:
            # Check if directory exists and is writable
            if not directory.exists():
                logger.error(f"Directory does not exist: {directory}")
                return False

            if not os.access(directory, os.W_OK):
                logger.error(f"Directory is not writable: {directory}")
                return False

            # Check available disk space (require at least 100MB free)
            disk_usage = shutil.disk_usage(directory)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 0.1:  # Less than 100MB
                logger.error(f"Insufficient disk space: {free_gb:.2f} GB free")
                return False

            # Try to create a test file
            test_file = directory / f".test_write_{int(time.time())}"
            try:
                test_file.write_text("test")
                test_file.unlink()  # Clean up
            except Exception as e:
                logger.error(f"Cannot write to directory: {e}")
                return False

            logger.debug(f"Filesystem health check passed for {directory}")
            return True

        except Exception as e:
            logger.error(f"Filesystem health check failed: {e}")
            return False

    def organize_batch_files(self) -> Optional[Path]:
        """Organize batch files into a subdirectory."""
        batch_files = self.find_batch_files()

        if not batch_files:
            logger.info("No batch files to organize")
            return None

        # Use the parent directory (lines directory)
        output_dir = self.output_directory

        # Check filesystem health before proceeding
        if not self._check_filesystem_health(output_dir):
            raise PSRMagPyIOError(f"Filesystem health check failed for {output_dir}")

        logger.info(f"Organizing {len(batch_files)} batch files in {output_dir}")

        if self.progress_reporter:
            self.progress_reporter.report_io_start_simple("organize", output_dir, len(batch_files))

        try:
            # Move and rename batch files with retry logic
            for i, batch_info in enumerate(batch_files):
                if self.progress_reporter:
                    self.progress_reporter.update_io_progress(
                        i, len(batch_files), f"Moving {batch_info.path.name}"
                    )

                # Keep experiment name from original filename
                experiment_name = self.base_filename.stem
                # Rename to organized format: experiment_name_batch_001_out_of_004.fits
                new_filename = f"{experiment_name}_batch_{i+1:03d}_out_of_{len(batch_files):03d}.fits"
                new_path = output_dir / new_filename

                # Retry logic for file operations
                max_retries = 3
                retry_delay = 0.5  # seconds

                for attempt in range(max_retries):
                    try:
                        # Check if source file still exists
                        if not batch_info.path.exists():
                            logger.warning(
                                f"Source file {batch_info.path} no longer exists, skipping"
                            )
                            break

                        # Check if destination already exists
                        if new_path.exists():
                            logger.warning(
                                f"Destination {new_path} already exists, removing old file"
                            )
                            new_path.unlink()

                        # Perform the move operation
                        shutil.move(str(batch_info.path), str(new_path))
                        logger.debug(
                            f"Moved and renamed {batch_info.path.name} to {new_filename}"
                        )
                        break  # Success, exit retry loop

                    except (OSError, IOError) as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Move attempt {attempt + 1} failed for {batch_info.path}: {e}"
                            )
                            logger.warning(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            # Final attempt failed
                            raise PSRMagPyIOError(
                                f"Failed to move {batch_info.path} after {max_retries} attempts: {e}"
                            ) from e

            if self.progress_reporter:
                elapsed = time.time() - (self.progress_reporter.start_time or time.time())
                self.progress_reporter.report_io_completion(
                    "organize", output_dir, elapsed, len(batch_files)
                )

            logger.info(
                f"Successfully organized and renamed {len(batch_files)} batch files"
            )
            return output_dir

        except Exception as e:
            # Log additional diagnostic information
            logger.error(f"Output directory: {output_dir}")
            logger.error(f"Directory exists: {output_dir.exists()}")
            logger.error(
                f"Directory writable: {os.access(output_dir, os.W_OK) if output_dir.exists() else 'N/A'}"
            )
            logger.error(
                f"Available disk space: {shutil.disk_usage(output_dir).free / (1024**3):.2f} GB"
            )
            raise PSRMagPyIOError(f"Failed to organize batch files: {e}") from e

    def _create_summary_file(
        self, batch_files: List[BatchFileInfo], summary_path: Path
    ) -> None:
        """Create a summary file with batch file information."""
        with open(summary_path, "w") as f:
            f.write("Batch File Summary\n")
            f.write("==================\n\n")
            f.write(f"Base filename: {self.base_filename.name}\n")
            f.write(f"Total batch files: {len(batch_files)}\n")
            f.write(
                f"Total size: {sum(batch.file_size_mb for batch in batch_files):.2f} MB\n\n"
            )

            f.write("Batch Files:\n")
            f.write("------------\n")
            for batch in sorted(batch_files, key=lambda x: x.batch_index):
                f.write(
                    f"  {batch.path.name}: {batch.file_size_mb:.2f} MB (batch {batch.batch_index})\n"
                )

            f.write(f"\nBatch files are kept in the lines directory.\n")
            f.write(
                f"Each file contains a subset of field lines from the calculation.\n"
            )

        logger.info(f"Created batch summary: {summary_path}")

    def cleanup_batch_files(self) -> None:
        """Clean up batch files after organization."""
        if not self._cleanup_enabled:
            logger.info("Batch file cleanup disabled")
            return

        batch_files = self.find_batch_files()

        if not batch_files:
            logger.info("No batch files to clean up")
            return

        logger.info(f"Cleaning up {len(batch_files)} batch files")

        if self.progress_reporter:
            self.progress_reporter.report_io_start_simple(
                "cleanup", Path("batch_files"), len(batch_files)
            )

        try:
            for i, batch_info in enumerate(batch_files):
                if self.progress_reporter:
                    self.progress_reporter.update_io_progress(
                        i, len(batch_files), f"Cleaning up {batch_info.path.name}"
                    )

                try:
                    batch_info.path.unlink()
                    logger.debug(f"Deleted batch file: {batch_info.path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete batch file {batch_info.path}: {e}"
                    )

            if self.progress_reporter:
                elapsed = time.time() - (self.progress_reporter.start_time or time.time())
                self.progress_reporter.report_io_completion(
                    "cleanup", Path("batch_files"), elapsed, len(batch_files)
                )

            logger.info("Batch file cleanup completed")

        except Exception as e:
            raise PSRMagPyIOError(f"Failed to cleanup batch files: {e}") from e

    def get_batch_summary(self) -> Dict[str, Any]:
        """Get a summary of batch files."""
        batch_files = self.find_batch_files()

        # If no batch files found in original location, check if they've been organized
        if not batch_files and self.organize_files:
            batch_dir = self.output_directory / f"{self.base_filename.stem}_batches"
            if batch_dir.exists():
                # Look for renamed batch files in the organized subdirectory
                pattern = str(batch_dir / "batch_*.fits")
                organized_files = glob.glob(pattern)

                if organized_files:
                    # Create batch info for organized files
                    batch_infos = []
                    for file_path in sorted(organized_files):
                        path = Path(file_path)
                        try:
                            filename = path.name
                            # Extract batch index from renamed filename: batch_001_out_of_004.fits -> 1
                            parts = filename.split("_")
                            if len(parts) >= 4 and parts[0] == "batch":
                                batch_index = int(parts[1])  # batch_001_out_of_004 -> 1
                            else:
                                logger.warning(
                                    f"Unexpected filename format: {filename}"
                                )
                                continue
                            stat = path.stat()
                            file_size_mb = stat.st_size / (1024 * 1024)
                            creation_time = stat.st_ctime

                            batch_info = BatchFileInfo(
                                path=path,
                                batch_index=batch_index,
                                file_size_mb=file_size_mb,
                                creation_time=creation_time,
                            )
                            batch_infos.append(batch_info)
                        except (ValueError, IndexError) as e:
                            logger.warning(
                                f"Could not parse batch index from {path}: {e}"
                            )
                            continue

                    batch_files = batch_infos

        if not batch_files:
            return {"status": "no_batch_files", "message": "No batch files found"}

        total_size_mb = sum(batch.file_size_mb for batch in batch_files)

        return {
            "status": "batch_files_found",
            "batch_files": len(batch_files),
            "total_size_mb": total_size_mb,
            "organize_enabled": self.organize_files,
            "cleanup_enabled": self._cleanup_enabled,
            "files": [
                {
                    "name": batch.path.name,
                    "batch_index": batch.batch_index,
                    "size_mb": batch.file_size_mb,
                }
                for batch in sorted(batch_files, key=lambda x: x.batch_index)
            ],
        }
