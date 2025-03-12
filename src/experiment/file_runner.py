import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .config import ExperimentConfig, ExperimentSetup
from .runner import ExperimentRunner

logger = logging.getLogger(__name__)

class ExperimentFileRunner:
    """Runs experiments from a configuration file with status tracking."""
    
    def __init__(self, config_file: Path):
        """Initialize with path to configuration file.
        
        Parameters
        ----------
        config_file : Path
            Path to JSON configuration file
        """
        self.config_file = config_file
        self.experiments = []
        self.status = {}
        
        # Load or create config file
        if config_file.exists():
            self._load_config()
        else:
            logger.info(f"Config file {config_file} does not exist, will be created when experiments are added")
    
    def add_experiments(self, setup: ExperimentSetup):
        """Add experiments from setup to the configuration file.
        
        Parameters
        ----------
        setup : ExperimentSetup
            Experiment setup with parameter sets/ranges
        """
        # Generate experiment configs
        configs = setup.generate_configs()
        logger.info(f"Adding {len(configs)} experiments to configuration")
        
        # Convert to serializable format
        for config in configs:
            exp_dict = self._config_to_dict(config)
            exp_id = exp_dict['experiment_name']
            
            # Add to experiments list if not already present
            if exp_id not in self.status:
                self.experiments.append(exp_dict)
                self.status[exp_id] = {
                    'completed': False,
                    'start_time': None,
                    'end_time': None,
                    'duration_sec': None,
                    'error': None
                }
        
        # Save updated configuration
        self._save_config()
        logger.info(f"Configuration saved to {self.config_file}")
    
    def run_all(self, resume: bool = True):
        """Run all experiments in the configuration file.
        
        Parameters
        ----------
        resume : bool, default=True
            If True, skip completed experiments
        """
        if not self.experiments:
            logger.warning("No experiments to run")
            return
        
        logger.info(f"Starting execution of {len(self.experiments)} experiments")
        
        for i, exp_dict in enumerate(self.experiments):
            exp_id = exp_dict['experiment_name']
            
            # Skip completed experiments if resuming
            if resume and self.status[exp_id]['completed']:
                logger.info(f"Skipping completed experiment {i+1}/{len(self.experiments)}: {exp_id}")
                continue
            
            logger.info(f"Running experiment {i+1}/{len(self.experiments)}: {exp_id}")
            
            # Update status
            self.status[exp_id]['start_time'] = datetime.now().isoformat()
            self.status[exp_id]['completed'] = False
            self.status[exp_id]['error'] = None
            self._save_config()
            
            try:
                # Convert dict back to config and run
                config = self._dict_to_config(exp_dict)
                runner = ExperimentRunner([config])
                start_time = time.time()
                runner.run_all()
                end_time = time.time()
                
                # Update status on success
                self.status[exp_id]['completed'] = True
                self.status[exp_id]['end_time'] = datetime.now().isoformat()
                self.status[exp_id]['duration_sec'] = end_time - start_time
                
            except Exception as e:
                # Update status on error
                logger.error(f"Error in experiment {exp_id}: {str(e)}")
                self.status[exp_id]['error'] = str(e)
                self.status[exp_id]['end_time'] = datetime.now().isoformat()
                
            # Save after each experiment
            self._save_config()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of experiment status."""
        total = len(self.experiments)
        completed = sum(1 for exp_id in self.status if self.status[exp_id]['completed'])
        failed = sum(1 for exp_id in self.status if self.status[exp_id]['error'])
        
        return {
            'total': total,
            'completed': completed,
            'pending': total - completed - failed,
            'failed': failed,
            'progress_percent': (completed / total * 100) if total > 0 else 0
        }
    
    def _load_config(self):
        """Load configuration from file."""
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        
        self.experiments = data['experiments']
        self.status = data['status']
    
    def _save_config(self):
        """Save configuration to file."""
        data = {
            'experiments': self.experiments,
            'status': self.status,
            'last_updated': datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to serializable dictionary."""
        # Implementation depends on your serialization needs
        # This is a simplified example
        return {
            'experiment_name': config.experiment_name,
            'field_config': self._serialize_field_config(config.field_config),
            'grid_config': self._serialize_grid_config(config.grid_config),
            'tracer_config': self._serialize_tracer_config(config.tracer_config),
            'output_dir': str(config.output_dir),
            'save_vars': config.save_vars
        }
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary back to ExperimentConfig."""
        # Implementation depends on your deserialization needs
        return ExperimentConfig(
            experiment_name=data['experiment_name'],
            field_config=self._deserialize_field_config(data['field_config']),
            grid_config=self._deserialize_grid_config(data['grid_config']),
            tracer_config=self._deserialize_tracer_config(data['tracer_config']),
            output_dir=Path(data['output_dir']),
            save_vars=data['save_vars']
        )
    
    # Serialization/deserialization methods for each config type
    # These would need to be implemented based on your specific needs 