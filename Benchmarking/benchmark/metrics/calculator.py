"""
Comprehensive metrics calculator for perturbation prediction evaluation
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from .base import BaseMetrics
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator(BaseMetrics):
    """
    Extended metrics calculator with distribution and biological metrics.

    Includes all common metrics plus:
    - MMD (Maximum Mean Discrepancy)
    - Wasserstein distance
    - Delta metrics (perturbation effect comparison)
    """

    def __init__(self):
        """Initialize the metrics calculator"""
        super().__init__()
        # Add extended metrics
        self.available_metrics.extend([
            'mmd', 'wasserstein',
            'delta_pearson', 'delta_spearman', 'delta_r2'
        ])

    def compute_mmd(self, y_true: Any, y_pred: Any,
                   kernel: str = 'rbf', gamma: float = 1.0) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD) between predicted and true distributions.

        MMD measures the distance between two probability distributions by comparing
        their mean embeddings in a reproducing kernel Hilbert space.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            kernel: Kernel type (currently only 'rbf' supported)
            gamma: RBF kernel parameter

        Returns:
            MMD value
        """
        y_true = self._to_array(y_true)
        y_pred = self._to_array(y_pred)

        if kernel == 'rbf':
            # Compute pairwise squared Euclidean distances
            dist_pred = cdist(y_pred, y_pred, metric='sqeuclidean')
            dist_truth = cdist(y_true, y_true, metric='sqeuclidean')
            dist_cross = cdist(y_pred, y_true, metric='sqeuclidean')

            # Apply RBF kernel transformation
            Kxx = np.exp(-gamma * dist_pred)
            Kyy = np.exp(-gamma * dist_truth)
            Kxy = np.exp(-gamma * dist_cross)

            # MMD formula: E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
            mmd = np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
            return float(mmd)
        else:
            raise ValueError("Unsupported kernel type. Use 'rbf'.")

    def compute_wasserstein(self, y_true: Any, y_pred: Any) -> float:
        """
        Compute Wasserstein distance between predicted and true distributions.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Wasserstein distance
        """
        y_true = self._to_array(y_true)
        y_pred = self._to_array(y_pred)
        return float(wasserstein_distance(y_true.flatten(), y_pred.flatten()))

    def compute_delta_metrics(self, pred_data: Any, true_data: Any,
                             ctrl_data: Any) -> Dict[str, float]:
        """
        Compute metrics on perturbation effects (delta = perturbed - control).

        This is particularly important for perturbation prediction as it measures
        how well the model captures the perturbation effect rather than absolute values.

        Args:
            pred_data: Predicted perturbed expression
            true_data: True perturbed expression
            ctrl_data: Control (unperturbed) expression

        Returns:
            Dictionary of delta-based metrics
        """
        pred_data = self._to_array(pred_data)
        true_data = self._to_array(true_data)
        ctrl_data = self._to_array(ctrl_data)

        # Compute mean expressions
        mean_pred = np.mean(pred_data, axis=0)
        mean_true = np.mean(true_data, axis=0)
        mean_ctrl = np.mean(ctrl_data, axis=0)

        # Compute deltas (perturbation effects)
        delta_pred = mean_pred - mean_ctrl
        delta_true = mean_true - mean_ctrl

        metrics = {}

        # Delta Pearson correlation
        try:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(delta_true, delta_pred)
            metrics['delta_pearson'] = float(corr) if not np.isnan(corr) else 0.0
        except:
            metrics['delta_pearson'] = 0.0

        # Delta Spearman correlation
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(delta_true, delta_pred)
            metrics['delta_spearman'] = float(corr) if not np.isnan(corr) else 0.0
        except:
            metrics['delta_spearman'] = 0.0

        # Delta R²
        try:
            from sklearn.metrics import r2_score
            metrics['delta_r2'] = float(r2_score(delta_true, delta_pred))
        except:
            metrics['delta_r2'] = 0.0

        return metrics

    def compute_all_with_control(self, pred_data: Any, true_data: Any,
                                 ctrl_data: Any,
                                 metrics: Optional[list] = None) -> Dict[str, float]:
        """
        Compute comprehensive metrics including both absolute and delta metrics.

        Args:
            pred_data: Predicted perturbed expression
            true_data: True perturbed expression
            ctrl_data: Control expression
            metrics: List of metric names to compute (if None, compute all)

        Returns:
            Dictionary of all computed metrics
        """
        results = {}

        # Compute standard metrics (absolute comparison)
        standard_metrics = self.compute_all(true_data, pred_data, metrics)
        results.update(standard_metrics)

        # Compute delta metrics (perturbation effect comparison)
        if metrics is None or any(m.startswith('delta_') for m in metrics):
            delta_metrics = self.compute_delta_metrics(pred_data, true_data, ctrl_data)
            results.update(delta_metrics)

        # Compute distribution metrics if requested
        if metrics is None or 'mmd' in metrics:
            try:
                results['mmd'] = self.compute_mmd(true_data, pred_data)
            except Exception as e:
                logger.warning(f"Failed to compute MMD: {e}")
                results['mmd'] = np.nan

        if metrics is None or 'wasserstein' in metrics:
            try:
                results['wasserstein'] = self.compute_wasserstein(true_data, pred_data)
            except Exception as e:
                logger.warning(f"Failed to compute Wasserstein: {e}")
                results['wasserstein'] = np.nan

        return results

    def save_metrics(self, metrics: Dict[str, float], output_path: str) -> None:
        """
        Save metrics to a file.

        Args:
            metrics: Dictionary of metrics
            output_path: Path to save the metrics (CSV format)
        """
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False)
        logger.info(f"Metrics saved to {output_path}")

    def load_metrics(self, input_path: str) -> Dict[str, float]:
        """
        Load metrics from a file.

        Args:
            input_path: Path to the metrics file

        Returns:
            Dictionary of metrics
        """
        df = pd.read_csv(input_path)
        return df.iloc[0].to_dict()
