"""
Evaluation pipeline for computing metrics
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from ..metrics.calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluation pipeline for computing and aggregating metrics.

    This class provides utilities to:
    - Compute metrics on predictions
    - Aggregate results across models
    - Generate comparison tables
    """

    def __init__(self, output_dir: str = "./results"):
        """
        Initialize the evaluator.

        Args:
            output_dir: Directory for saving results
        """
        self.metrics_calculator = MetricsCalculator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(self, pred_data: Any, true_data: Any,
                       ctrl_data: Optional[Any] = None,
                       metrics: Optional[list] = None) -> Dict[str, float]:
        """
        Compute metrics for predictions.

        Args:
            pred_data: Predicted data
            true_data: Ground truth data
            ctrl_data: Optional control data (for delta metrics)
            metrics: List of metrics to compute (if None, compute all)

        Returns:
            Dictionary of computed metrics
        """
        if ctrl_data is not None:
            # Compute comprehensive metrics including delta metrics
            return self.metrics_calculator.compute_all_with_control(
                pred_data, true_data, ctrl_data, metrics
            )
        else:
            # Compute standard metrics only
            return self.metrics_calculator.compute_all(
                true_data, pred_data, metrics
            )

    def aggregate_metrics(self, metrics_dir: str,
                         pattern: str = "*_metrics.csv") -> pd.DataFrame:
        """
        Aggregate metrics from multiple files.

        Args:
            metrics_dir: Directory containing metrics files
            pattern: File pattern to match

        Returns:
            DataFrame with aggregated metrics
        """
        metrics_path = Path(metrics_dir)
        all_metrics = []

        for metrics_file in metrics_path.glob(pattern):
            try:
                df = pd.read_csv(metrics_file)
                df['source_file'] = metrics_file.name
                all_metrics.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {metrics_file}: {e}")

        if not all_metrics:
            logger.warning(f"No metrics files found in {metrics_dir}")
            return pd.DataFrame()

        aggregated = pd.concat(all_metrics, ignore_index=True)
        return aggregated

    def create_comparison_table(self, task: str,
                               metrics_to_compare: Optional[list] = None) -> pd.DataFrame:
        """
        Create a comparison table of all models for a task.

        Args:
            task: Task name
            metrics_to_compare: List of metrics to include

        Returns:
            DataFrame with model comparison
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['r2_score', 'pearson', 'mse',
                                 'delta_pearson', 'delta_r2']

        # This is a placeholder - actual implementation would aggregate
        # results from the metrics files
        logger.info(f"Creating comparison table for {task}")
        return pd.DataFrame()

    def save_metrics(self, metrics: Dict[str, float],
                    output_path: str) -> None:
        """
        Save metrics to file.

        Args:
            metrics: Dictionary of metrics
            output_path: Path to save
        """
        self.metrics_calculator.save_metrics(metrics, output_path)
