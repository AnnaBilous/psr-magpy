"""Batch data reading helpers for experiment line outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, Optional
import glob

import numpy as np

from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.utils.exceptions import PSRMagPyValidationError
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


def _field_lines_row_count(batch_file: Path) -> int:
    """Return the number of rows in the FIELD_LINES HDU, or 0 if absent/empty."""
    from astropy.io import fits

    with fits.open(batch_file) as hdul:
        if "FIELD_LINES" not in [h.name for h in hdul]:
            return 0
        data = hdul["FIELD_LINES"].data
        if data is None:
            return 0
        return len(data)


class BatchDataReader:
    """Read and process batch FITS outputs for one experiment."""

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def load_combined_batch_data(
        self,
        template: FieldLine,
        lines_dir: Optional[str] = None,
        *,
        batch_filename_prefix: Optional[str] = None,
    ) -> Optional[FieldLineSet]:
        """Load and combine all batch files for the experiment."""
        from psr_magpy.io import load_data

        if lines_dir is None:
            raise PSRMagPyValidationError("lines_dir must be specified")

        prefix = (
            batch_filename_prefix
            if batch_filename_prefix is not None
            else self.experiment_name
        )
        batch_pattern = f"{lines_dir}/{prefix}_batch_*.fits"
        batch_files = sorted(glob.glob(batch_pattern))
        if not batch_files:
            single = Path(lines_dir) / f"{prefix}.fits"
            if single.is_file():
                batch_files = [str(single)]
        if not batch_files:
            logger.warning(f"No batch files found for {prefix}")
            return None

        logger.info(f"Found {len(batch_files)} batch files for {prefix}")
        combined = FieldLineSet(template, copy_data=False)
        for batch_file in batch_files:
            batch_path = Path(batch_file)
            if _field_lines_row_count(batch_path) == 0:
                logger.debug(f"Skipping empty batch: {batch_path.name}")
                continue
            logger.debug(f"Loading {batch_file}")
            field_line_set = load_data(FieldLineSet, batch_file, template=template)
            combined.append(field_line_set)

        logger.info(f"Combined field line set: {combined.n_field_lines} field lines")

        if hasattr(combined, "footpoint_id") and combined.footpoint_id is not None:
            footpoint_ids = np.asarray(combined.footpoint_id)
            unique_ids, counts = np.unique(footpoint_ids, return_counts=True)
            if unique_ids.size != footpoint_ids.size:
                duplicates = unique_ids[counts > 1]
                raise PSRMagPyValidationError(
                    f"Duplicate footpoint indices found: {duplicates[:10].tolist()}{'...' if duplicates.size > 10 else ''}"
                )
            logger.info(f"Verified {footpoint_ids.size} unique footpoint indices")

        return combined

    def iter_batch_files(
        self,
        lines_dir: str,
        *,
        batch_filename_prefix: Optional[str] = None,
    ) -> Iterator[Path]:
        """Iterate batch file paths in sorted order.

        Parameters
        ----------
        lines_dir
            Directory containing FITS files.
        batch_filename_prefix
            If set, match ``{prefix}_batch_*.fits`` (and optionally ``{prefix}.fits``).
            Default: ``experiment_name`` from the reader constructor.
        """
        if lines_dir is None:
            raise PSRMagPyValidationError("lines_dir must be specified")

        prefix = (
            batch_filename_prefix
            if batch_filename_prefix is not None
            else self.experiment_name
        )
        batch_pattern = f"{lines_dir}/{prefix}_batch_*.fits"
        found = sorted(glob.glob(batch_pattern))
        if not found:
            single = Path(lines_dir) / f"{prefix}.fits"
            if single.is_file():
                found = [str(single)]
        for batch_file in found:
            yield Path(batch_file)

    def iter_batch_data(
        self,
        template: FieldLine,
        lines_dir: str,
        *,
        batch_filename_prefix: Optional[str] = None,
    ) -> Iterator[FieldLineSet]:
        """Iterate loaded non-empty batch datasets."""
        from psr_magpy.io import load_data

        for batch_file in self.iter_batch_files(
            lines_dir, batch_filename_prefix=batch_filename_prefix
        ):
            if _field_lines_row_count(batch_file) == 0:
                logger.debug(f"Skipping empty batch: {batch_file.name}")
                continue
            logger.debug(f"Loading batch: {batch_file.name}")
            batch_data = load_data(FieldLineSet, batch_file, template=template)
            if batch_data is not None and batch_data.n_field_lines > 0:
                yield batch_data

    def process_batches_incremental(
        self,
        template: FieldLine,
        lines_dir: str,
        processor_fn: Callable[[FieldLineSet], Any],
        accumulator_fn: Callable[[Any, Any], Any],
        initial_value: Any = None,
        *,
        batch_filename_prefix: Optional[str] = None,
    ) -> Any:
        """Process batches incrementally and accumulate result."""
        result = initial_value
        batch_count = 0

        for batch_data in self.iter_batch_data(
            template, lines_dir, batch_filename_prefix=batch_filename_prefix
        ):
            batch_count += 1
            logger.debug(
                f"Processing batch {batch_count}: {batch_data.n_field_lines} field lines"
            )
            batch_result = processor_fn(batch_data)

            if result is None and initial_value is None:
                result = batch_result
            else:
                result = accumulator_fn(result, batch_result)

        logger.info(f"Processed {batch_count} batches")
        return result
