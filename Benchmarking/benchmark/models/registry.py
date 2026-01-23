"""
Model registry for managing all benchmark models
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from ..config import Config

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing all models in the benchmark.

    Provides a unified interface to run any model by delegating to the original
    model prediction scripts.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the model registry.

        Args:
            config: Configuration object (if None, will create default)
        """
        self.config = config or Config()
        self.models_dir = Path(self.config.datasets['model_scripts_dir'])
        self.metrics_dir = Path(self.config.datasets['metrics_scripts_dir'])

        # Verify directories exist
        if not self.models_dir.exists():
            raise ValueError(f"Model scripts directory not found: {self.models_dir}")
        if not self.metrics_dir.exists():
            raise ValueError(f"Metrics scripts directory not found: {self.metrics_dir}")

    def get_model_script_path(self, model_name: str) -> Path:
        """
        Get the path to a model's prediction script.

        Args:
            model_name: Name of the model

        Returns:
            Path to the model script
        """
        model_config = self.config.get_model_config(model_name)
        script_name = model_config['script']
        script_path = self.models_dir / script_name

        if not script_path.exists():
            raise FileNotFoundError(f"Model script not found: {script_path}")

        return script_path

    def get_metrics_script_path(self, model_name: str) -> Path:
        """
        Get the path to a model's metrics script.

        Args:
            model_name: Name of the model

        Returns:
            Path to the metrics script
        """
        model_config = self.config.get_model_config(model_name)
        script_name = model_config['metrics_script']
        script_path = self.metrics_dir / script_name

        if not script_path.exists():
            raise FileNotFoundError(f"Metrics script not found: {script_path}")

        return script_path

    def run_model_prediction(self, model_name: str, task: str,
                            env: Optional[Dict[str, str]] = None,
                            cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a model's prediction script.

        Args:
            model_name: Name of the model to run
            task: Task name (task1, task2, task3)
            env: Optional environment variables
            cwd: Optional working directory

        Returns:
            Dictionary with execution results
        """
        # Validate model and task compatibility
        model_tasks = self.config.get_tasks_for_model(model_name)
        if task not in model_tasks:
            raise ValueError(
                f"Model '{model_name}' is not configured for '{task}'. "
                f"Available tasks: {model_tasks}"
            )

        script_path = self.get_model_script_path(model_name)
        model_config = self.config.get_model_config(model_name)

        logger.info(f"Running {model_config['name']} for {task}")
        logger.info(f"Script: {script_path}")

        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Run the script
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                env=run_env,
                cwd=cwd,
                capture_output=True,
                text=True
            )

            return {
                'model': model_name,
                'task': task,
                'script': str(script_path),
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }

        except Exception as e:
            logger.error(f"Error running {model_name}: {e}")
            return {
                'model': model_name,
                'task': task,
                'script': str(script_path),
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'error': str(e)
            }

    def run_model_metrics(self, model_name: str, task: str,
                         env: Optional[Dict[str, str]] = None,
                         cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a model's metrics calculation script.

        Args:
            model_name: Name of the model
            task: Task name (task1, task2, task3)
            env: Optional environment variables
            cwd: Optional working directory

        Returns:
            Dictionary with execution results
        """
        script_path = self.get_metrics_script_path(model_name)
        model_config = self.config.get_model_config(model_name)

        logger.info(f"Computing metrics for {model_config['name']} on {task}")
        logger.info(f"Script: {script_path}")

        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Run the script
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                env=run_env,
                cwd=cwd,
                capture_output=True,
                text=True
            )

            return {
                'model': model_name,
                'task': task,
                'script': str(script_path),
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }

        except Exception as e:
            logger.error(f"Error computing metrics for {model_name}: {e}")
            return {
                'model': model_name,
                'task': task,
                'script': str(script_path),
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'error': str(e)
            }

    def list_models(self, task: Optional[str] = None) -> list:
        """
        List available models.

        Args:
            task: If specified, only list models for this task

        Returns:
            List of model names
        """
        if task:
            return self.config.get_models_for_task(task)
        else:
            return self.config.get_all_models()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration dictionary
        """
        return self.config.get_model_config(model_name)
