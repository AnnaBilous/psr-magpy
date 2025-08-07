"""Batch file management for organizing and cleaning up batch files.

This module provides functionality to organize batch files into a clean
directory structure and optionally clean up temporary batch files.
"""

import glob
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from psr_magpy.utils.logging_config import get_logger

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
        progress_tracker=None,
    ):
        """Initialize batch file manager."""
        self.base_filename = Path(base_filename)
        self.output_directory = output_directory or self.base_filename.parent
        self.organize_files = organize_files
        self._cleanup_enabled = cleanup_batch_files
        self.progress_tracker = progress_tracker

        logger.debug(
            f"Initialized BatchFileManager: base={self.base_filename}, "
            f"organize={organize_files}, cleanup={self._cleanup_enabled}"
        )

    def find_batch_files(self) -> List[BatchFileInfo]:
        """Find all batch files for the base filename."""
        pattern = str(self.output_directory / f"{self.base_filename.stem}_batch_*.fits")
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
                batch_index = int(filename.split("_batch_")[1].split(".")[0])

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

    def organize_batch_files(self) -> Optional[Path]:
        """Organize batch files into a subdirectory."""
        batch_files = self.find_batch_files()

        if not batch_files:
            logger.info("No batch files to organize")
            return None

        # Create batch directory
        batch_dir = self.output_directory / f"{self.base_filename.stem}_batches"

        # Clean up existing files in batch directory
        if batch_dir.exists():
            logger.info(f"Cleaning up existing files in {batch_dir}")
            for file_path in batch_dir.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        logger.debug(f"Removed existing file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
        else:
            batch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Organizing {len(batch_files)} batch files into {batch_dir}")

        if self.progress_tracker:
            self.progress_tracker.log_io_start("organize", batch_dir, len(batch_files))

        try:
            # Move and rename batch files to subdirectory
            for i, batch_info in enumerate(batch_files):
                if self.progress_tracker:
                    self.progress_tracker.update_io_progress(
                        i, len(batch_files), f"Moving {batch_info.path.name}"
                    )

                # Rename to organized format: batch_001_out_of_004.fits, batch_002_out_of_004.fits, etc.
                new_filename = f"batch_{i+1:03d}_out_of_{len(batch_files):03d}.fits"
                new_path = batch_dir / new_filename
                shutil.move(str(batch_info.path), str(new_path))
                logger.debug(
                    f"Moved and renamed {batch_info.path.name} to {new_filename}"
                )

            if self.progress_tracker:
                self.progress_tracker.log_io_completion(
                    "organize", batch_dir, 0.0, len(batch_files)
                )

            logger.info(
                f"Successfully organized and renamed {len(batch_files)} batch files into {batch_dir}"
            )
            return batch_dir

        except Exception as e:
            error_msg = f"Failed to organize batch files: {e}"
            logger.error(error_msg)
            raise

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

            f.write(
                f"\nBatch files are intentionally kept separate due to large size.\n"
            )
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

        if self.progress_tracker:
            self.progress_tracker.log_io_start(
                "cleanup", Path("batch_files"), len(batch_files)
            )

        try:
            for i, batch_info in enumerate(batch_files):
                if self.progress_tracker:
                    self.progress_tracker.update_io_progress(
                        i, len(batch_files), f"Cleaning up {batch_info.path.name}"
                    )

                try:
                    batch_info.path.unlink()
                    logger.debug(f"Deleted batch file: {batch_info.path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete batch file {batch_info.path}: {e}"
                    )

            if self.progress_tracker:
                self.progress_tracker.log_io_completion(
                    "cleanup", Path("batch_files"), 0.0, len(batch_files)
                )

            logger.info("Batch file cleanup completed")

        except Exception as e:
            error_msg = f"Failed to cleanup batch files: {e}"
            logger.error(error_msg)
            raise

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
