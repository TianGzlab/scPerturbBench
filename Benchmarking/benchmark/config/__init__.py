"""
Configuration management module
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager for the benchmark framework"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)

        self.models = self._load_yaml('models.yaml')
        self.tasks = self._load_yaml('tasks.yaml')
        self.datasets = self._load_yaml('datasets.yaml')

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        filepath = self.config_dir / filename
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        if model_name not in self.models['models']:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return self.models['models'][model_name]

    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for a specific task"""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found in configuration")
        return self.tasks[task_name]

    def get_dataset_config(self, task_name: str) -> Dict[str, Any]:
        """Get dataset configuration for a specific task"""
        if task_name not in self.datasets:
            raise ValueError(f"No dataset configuration found for task '{task_name}'")
        return self.datasets[task_name]

    def get_models_for_task(self, task_name: str) -> list:
        """Get list of models applicable for a specific task"""
        task_config = self.get_task_config(task_name)
        return task_config.get('models', [])

    def get_tasks_for_model(self, model_name: str) -> list:
        """Get list of tasks a model can be applied to"""
        model_config = self.get_model_config(model_name)
        return model_config.get('tasks', [])

    def get_all_models(self) -> list:
        """Get list of all available models"""
        return list(self.models['models'].keys())

    def get_all_tasks(self) -> list:
        """Get list of all available tasks"""
        return [k for k in self.tasks.keys() if k.startswith('task')]
