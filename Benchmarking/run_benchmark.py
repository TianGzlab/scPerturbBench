#!/usr/bin/env python
"""
Benchmark Pipeline - Unified Model Execution Script

This script provides a simplified interface to run perturbation prediction models.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add benchmark module to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark.config import Config
from benchmark.models.registry import ModelRegistry


def setup_logging():
    """Setup basic logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run perturbation prediction benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Main options
    parser.add_argument('--list-tasks', action='store_true',
                       help='List available tasks')
    parser.add_argument('--list-models', type=str,
                       help='List models for a task (task1/task2/task3)')
    parser.add_argument('--task', type=str, choices=['task1', 'task2', 'task3'],
                       help='Task to run')
    parser.add_argument('--model', type=str,
                       help='Model to run')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory')

    args = parser.parse_args()
    logger = setup_logging()

    # Initialize
    config = Config()
    registry = ModelRegistry(config)

    # List tasks
    if args.list_tasks:
        logger.info("\nAvailable tasks:")
        for task in config.get_all_tasks():
            task_config = config.get_task_config(task)
            logger.info(f"  {task}: {task_config['name']}")
            models = config.get_models_for_task(task)
            logger.info(f"    Models: {len(models)}")
        return 0

    # List models
    if args.list_models:
        task = args.list_models
        models = config.get_models_for_task(task)
        logger.info(f"\nModels for {task}:")
        for model in models:
            info = registry.get_model_info(model)
            logger.info(f"  - {model}: {info['name']}")
        return 0

    # Run model
    if args.task and args.model:
        logger.info(f"Running {args.model} on {args.task}")

        # Get model script path
        script_path = registry.get_model_script_path(args.model)
        logger.info(f"Model script: {script_path}")

        # Run prediction
        logger.info("Executing model prediction...")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("✓ Prediction completed successfully")
            print(result.stdout)
        else:
            logger.error("✗ Prediction failed")
            print(result.stderr)
            return 1

        # Run metrics
        metrics_script = registry.get_metrics_script_path(args.model)
        logger.info(f"Computing metrics: {metrics_script}")

        result = subprocess.run(
            [sys.executable, str(metrics_script)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("✓ Metrics computed successfully")
            print(result.stdout)
        else:
            logger.error("✗ Metrics computation failed")
            print(result.stderr)
            return 1

        return 0

    # No action specified
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
