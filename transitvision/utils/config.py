"""Configuration utilities for the TransitVision package."""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for TransitVision.
    
    This class handles loading, saving, and accessing configuration settings.
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None) -> None:
        """Initialize the configuration manager.
        
        Args:
            config_file: Optional path to a configuration file.
        """
        # Default configuration
        self._config: Dict[str, Any] = {
            "data_processing": {
                "raw_data_dir": "data/raw",
                "processed_data_dir": "data/processed",
                "external_data_dir": "data/external",
                "time_columns": ["departure_time", "arrival_time"],
                "categorical_columns": ["route_id", "service_id", "trip_id", "stop_id"],
                "numerical_columns": ["ridership", "capacity", "delay"],
                "date_columns": ["service_date", "schedule_date"],
            },
            "analysis": {
                "plot_style": "whitegrid",
                "plot_palette": "viridis",
                "plot_figsize": [12, 8],
                "output_dir": "output/analysis",
            },
            "prediction": {
                "model_dir": "models",
                "train_test_split": 0.2,
                "random_state": 42,
                "cv_folds": 5,
            },
            "logging": {
                "level": "INFO",
                "log_file": "logs/transitvision.log",
                "console": True,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
        }
        
        # Load configuration from file if provided
        if config_file:
            self.load_config(config_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key, can use dot notation for nested keys.
            default: Default value to return if key is not found.
            
        Returns:
            Configuration value or default.
        """
        # Handle nested keys with dot notation
        keys = key.split('.')
        
        # Start with the entire config
        value = self._config
        
        # Navigate through nested keys
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key, can use dot notation for nested keys.
            value: Value to set.
        """
        # Handle nested keys with dot notation
        keys = key.split('.')
        
        # Navigate to the correct location
        config = self._config
        
        # Create nested structure if needed
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                # Convert to dict if not already
                config[k] = {}
            
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with a dictionary.
        
        Args:
            config_dict: Dictionary with configuration values to update.
        """
        self._update_nested(self._config, config_dict)
    
    def _update_nested(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries.
        
        Args:
            target: Target dictionary to update.
            source: Source dictionary with new values.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_nested(target[key], value)
            else:
                # Update or add value
                target[key] = value
    
    def load_config(self, config_file: Union[str, Path]) -> None:
        """Load configuration from a file.
        
        Args:
            config_file: Path to configuration file (JSON or YAML).
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            # Determine file type from extension
            if config_path.suffix in ['.json']:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return
            
            # Update configuration
            self.update(config_data)
            
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    def save_config(self, config_file: Union[str, Path], format: str = 'json') -> None:
        """Save configuration to a file.
        
        Args:
            config_file: Path to save configuration to.
            format: File format ('json' or 'yaml').
        """
        config_path = Path(config_file)
        
        # Create directory if it doesn't exist
        if not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save based on specified format
            if format.lower() == 'json':
                with open(config_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported configuration format: {format}")
                return
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {str(e)}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration.
        
        Returns:
            Dictionary with all configuration values.
        """
        return self._config.copy()
    
    def reset(self) -> None:
        """Reset configuration to default values."""
        self.__init__()
    
    def from_env(self, prefix: str = "TRANSITVISION_") -> None:
        """Load configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables.
        """
        # Find all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Replace double underscore with dot for nested keys
                config_key = config_key.replace('__', '.')
                
                # Try to parse value as JSON, fallback to string
                try:
                    parsed_value = json.loads(value)
                    self.set(config_key, parsed_value)
                except json.JSONDecodeError:
                    self.set(config_key, value)
                
                logger.debug(f"Loaded configuration from environment: {config_key} = {value}")


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Global Config instance.
    """
    return config