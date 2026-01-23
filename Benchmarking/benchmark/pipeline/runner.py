"""
Main benchmark runner for executing models and computing metrics
"""

import os
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..config import Config
from ..models.registry import ModelRegistry
from ..data.loader import DataLoader

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Main benchmark runner that orchestrates model execution and evaluation.

    This class provides a unified interface to:
    - Run individual models or full benchmark
    - Execute predictions and compute metrics
    - Save and organize results
    """

    def __init__(self, config: Optional[Config] = None,
                 output_dir: str = "./results"):
        """
        Initialize the benchmark runner.

        Args:
            config: Configuration object (if None, uses default)
            output_dir: Directory to save results
        """
        self.config = config or Config()
        self.model_registry = ModelRegistry(self.config)
        self.data_loader = DataLoader(self.config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        logger.info(f"BenchmarkRunner initialized with output_dir: {self.output_dir}")

    def run_single_model(self, model_name: str, task: str,
                        predict: bool = True,
                        compute_metrics: bool = True,
                        env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run a single model on a specific task.

        Args:
            model_name: Name of the model to run
            task: Task name (task1, task2, task3)
            predict: Whether to run prediction
            compute_metrics: Whether to compute metrics
            env: Optional environment variables

        Returns:
            Dictionary with execution results
        """
        logger.info(f"Running {model_name} on {task}")

        results = {
            'model': model_name,
            'task': task,
            'timestamp': datetime.now().isoformat(),
            'prediction': None,
            'metrics': None
        }

        # Run prediction
        if predict:
            logger.info(f"Running prediction for {model_name}")
            pred_result = self.model_registry.run_model_prediction(
                model_name, task, env=env
            )
            results['prediction'] = pred_result

            if not pred_result['success']:
                logger.error(f"Prediction failed for {model_name}")
                logger.error(f"Error: {pred_result.get('stderr', 'Unknown error')}")
                return results

        # Compute metrics
        if compute_metrics:
            logger.info(f"Computing metrics for {model_name}")
            metrics_result = self.model_registry.run_model_metrics(
                model_name, task, env=env
            )
            results['metrics'] = metrics_result

            if not metrics_result['success']:
                logger.error(f"Metrics computation failed for {model_name}")
                logger.error(f"Error: {metrics_result.get('stderr', 'Unknown error')}")

        # Save results summary
        self._save_run_summary(results)

        return results

    def run_task(self, task: str,
                models: Optional[List[str]] = None,
                predict: bool = True,
                compute_metrics: bool = True,
                env: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Run all applicable models for a specific task.

        Args:
            task: Task name (task1, task2, task3)
            models: List of specific models to run (if None, run all)
            predict: Whether to run predictions
            compute_metrics: Whether to compute metrics
            env: Optional environment variables

        Returns:
            List of results for each model
        """
        logger.info(f"Running benchmark for {task}")

        # Get models to run
        if models is None:
            models = self.config.get_models_for_task(task)
        else:
            # Validate models
            available = self.config.get_models_for_task(task)
            invalid = set(models) - set(available)
            if invalid:
                raise ValueError(
                    f"Models {invalid} are not available for {task}. "
                    f"Available models: {available}"
                )

        logger.info(f"Running {len(models)} models: {models}")

        # Run each model
        all_results = []
        for model_name in models:
            try:
                result = self.run_single_model(
                    model_name, task,
                    predict=predict,
                    compute_metrics=compute_metrics,
                    env=env
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error running {model_name}: {e}")
                all_results.append({
                    'model': model_name,
                    'task': task,
                    'error': str(e),
                    'success': False
                })

        # Save task summary
        self._save_task_summary(task, all_results)

        return all_results

    def run_full_benchmark(self, tasks: Optional[List[str]] = None,
                          predict: bool = True,
                          compute_metrics: bool = True,
                          env: Optional[Dict[str, str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run the full benchmark across all tasks.

        Args:
            tasks: List of tasks to run (if None, run all)
            predict: Whether to run predictions
            compute_metrics: Whether to compute metrics
            env: Optional environment variables

        Returns:
            Dictionary mapping task names to their results
        """
        logger.info("Running full benchmark")

        # Get tasks to run
        if tasks is None:
            tasks = self.config.get_all_tasks()

        logger.info(f"Running {len(tasks)} tasks: {tasks}")

        # Run each task
        all_results = {}
        for task in tasks:
            try:
                task_results = self.run_task(
                    task,
                    predict=predict,
                    compute_metrics=compute_metrics,
                    env=env
                )
                all_results[task] = task_results
            except Exception as e:
                logger.error(f"Error running {task}: {e}")
                all_results[task] = [{
                    'task': task,
                    'error': str(e),
                    'success': False
                }]

        # Save benchmark summary
        self._save_benchmark_summary(all_results)

        logger.info("Full benchmark completed")
        return all_results

    def _save_run_summary(self, results: Dict[str, Any]) -> None:
        """Save summary of a single model run"""
        model = results['model']
        task = results['task']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = self.output_dir / "logs" / f"{task}_{model}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Run summary saved to {output_file}")

    def _save_task_summary(self, task: str, results: List[Dict[str, Any]]) -> None:
        """Save summary of all models for a task"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / "logs" / f"{task}_summary_{timestamp}.json"

        summary = {
            'task': task,
            'timestamp': timestamp,
            'total_models': len(results),
            'successful': sum(1 for r in results if r.get('prediction', {}).get('success', False)),
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Task summary saved to {output_file}")

    def _save_benchmark_summary(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save summary of the full benchmark"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_summary_{timestamp}.json"

        summary = {
            'timestamp': timestamp,
            'tasks': list(results.keys()),
            'total_runs': sum(len(task_results) for task_results in results.values()),
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Benchmark summary saved to {output_file}")

    def list_available_models(self, task: Optional[str] = None) -> List[str]:
        """
        List available models.

        Args:
            task: If specified, only list models for this task

        Returns:
            List of model names
        """
        return self.model_registry.list_models(task)

    def list_available_tasks(self) -> List[str]:
        """
        List available tasks.

        Returns:
            List of task names
        """
        return self.config.get_all_tasks()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model_name: Model name

        Returns:
            Model information dictionary
        """
        return self.model_registry.get_model_info(model_name)
