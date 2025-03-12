from typing import List, Optional, Dict
import logging
from pathlib import Path
import time
import numpy as np

from .config import ExperimentConfig, FieldConfig, GridConfig
from src.field.multipole import MultipoleField
from src.field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from src.footpoint_grid.grid import HealpixGrid, SphericalGrid, RandomGrid
from src.line.calculator import FieldLineCalculator
from src.line.field_line import FieldLine
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs field line calculation experiments."""
    
    def __init__(self, configs: List[ExperimentConfig]):
        self.configs = configs
        self._progress: Dict[str, bool] = {
            config.experiment_name: False for config in configs
        }
        
    def run_all(self, resume_from: Optional[str] = None):
        """Run all experiments in sequence."""
        start_idx = self._find_resume_index(resume_from) if resume_from else 0
        total = len(self.configs)
        
        for i, config in enumerate(self.configs[start_idx:], start=start_idx):
            try:
                logger.info(f"Running experiment {i+1}/{total}: {config.experiment_name}")
                self.run_single(config)
                self._progress[config.experiment_name] = True
            except Exception as e:
                logger.error(f"Experiment {config.experiment_name} failed: {e}")
                raise

    def _find_resume_index(self, experiment_name: str) -> int:
        """Find index to resume from."""
        for i, config in enumerate(self.configs):
            if config.experiment_name == experiment_name:
                return i
        raise ValueError(f"Experiment {experiment_name} not found")

    def _setup_field(self, config: FieldConfig) -> MultipoleField:
        """Create magnetic field from configuration."""
        components = []
        
        # Add dipole component
        dipole = DipoleFieldComponent(
            B_magnitude=config.dipole_strength,
            moment_orientation_latitude_degree=config.dipole_lat,
            moment_orientation_longitude_degree=config.dipole_lon,
            offset_r_ns=config.dipole_offset or np.zeros(3)
        )
        components.append(dipole)
        
        # Add quadrupole if specified
        if config.quadrupole_strength is not None:
            quadrupole = QuadrupoleFieldComponent(
                B_magnitude=config.quadrupole_strength,
                moment_orientation_latitude_degree=config.quadrupole_lat or 0.0,
                moment_orientation_longitude_degree=config.quadrupole_lon or 0.0
            )
            components.append(quadrupole)
            
        return MultipoleField(components)

    def _setup_grid(self, config: GridConfig):
        """Create grid from configuration."""
        if config.grid_type == 'healpix':
            if not config.nside:
                raise ValueError("nside required for healpix grid")
            return HealpixGrid(nside=config.nside)
        elif config.grid_type == 'spherical':
            if not (config.nlon and config.nlat):
                raise ValueError("nlon and nlat required for spherical grid")
            return SphericalGrid(nlon=config.nlon, nlat=config.nlat)
        elif config.grid_type == 'random':
            if not config.npoints:
                raise ValueError("npoints required for random grid")
            return RandomGrid(nline_ori=config.npoints)
        else:
            raise ValueError(f"Unknown grid type: {config.grid_type}")

    def _create_template(self, save_vars: List[str]) -> FieldLine:
        """Create field line template with specified variables to save."""
        template = FieldLine()
        for var in save_vars:
            setattr(template.save_flags, f"save_{var}", True)
        return template

    def _get_output_path(self, config: ExperimentConfig) -> Path:
        """Get output path for experiment results."""
        return config.output_dir / f"{config.experiment_name}.fits"

    def run_single(self, config: ExperimentConfig):
        """Run single experiment with given configuration."""
        logger.info(f"Starting experiment: {config.experiment_name}")
        
        # Setup field
        field = self._setup_field(config.field_config)
        
        # Setup grid
        grid = self._setup_grid(config.grid_config)
        
        # Run calculation
        calculator = FieldLineCalculator(field, grid)
        results = calculator.get_field_lines(
            template=self._create_template(config.save_vars),
            config=config.tracer_config
        )
        
        # Save results
        output_path = self._get_output_path(config)
        results.save_to_fits(output_path)
        logger.info(f"Results saved to {output_path}") 