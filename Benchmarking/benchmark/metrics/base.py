"""
Base metrics class for the benchmark framework
"""

from abc import ABC
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)


class BaseMetrics:
    """
    Base class for computing evaluation metrics.

    Provides common metric computation methods that can be used across different tasks.
    """

    def __init__(self):
        """Initialize the metrics calculator"""
        self.available_metrics = [
            'r2_score', 'mse', 'mae', 'rmse',
            'pearson', 'spearman',
            'mean_pearson', 'mean_spearman'
        ]

    @staticmethod
    def _to_array(data: Any) -> np.ndarray:
        """
        Convert various data formats to numpy array.

        Args:
            data: Data to convert (DataFrame, sparse matrix, or array)

        Returns:
            Numpy array
        """
        if isinstance(data, pd.DataFrame):
            return data.values
        elif hasattr(data, 'toarray'):  # Sparse matrix
            return data.toarray()
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def compute_r2(self, y_true: Any, y_pred: Any, per_gene: bool = False) -> float:
        """
        Compute R² score.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            per_gene: If True, compute per-gene R² and return mean

        Returns:
            R² score
        """
        y_true = self._to_array(y_true)
        y_pred = self._to_array(y_pred)

        if per_gene:
            # Compute R² per gene (column-wise)
            r2_scores = []
            for i in range(y_true.shape[1]):
                try:
                    r2 = r2_score(y_true[:, i], y_pred[:, i])
                    r2_scores.append(r2)
                except:
                    continue
            return np.mean(r2_scores) if r2_scores else 0.0
        else:
            return r2_score(y_true.flatten(), y_pred.flatten())

    def compute_mse(self, y_true: Any, y_pred: Any) -> float:
        """Compute Mean Squared Error"""
        y_true = self._to_array(y_true)
        y_pred = self._to_array(y_pred)
        return mean_squared_error(y_true.flatten(), y_pred.flatten())

    def compute_mae(self, y_true: Any, y_pred: Any) -> float:
        """Compute Mean Absolute Error"""
        y_true = self._to_array(y_true)
        y_pred = self._to_array(y_pred)
        return mean_absolute_error(y_true.flatten(), y_pred.flatten())

    def compute_rmse(self, y_true: Any, y_pred: Any) -> float:
        """Compute Root Mean Squared Error"""
        return np.sqrt(self.compute_mse(y_true, y_pred))

    def compute_pearson(self, y_true: Any, y_pred: Any, per_sample: bool = False) -> float:
        """
        Compute Pearson correlation.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            per_sample: If True, compute per-sample correlation and return mean

        Returns:
            Pearson correlation coefficient
        """
        y_true = self._to_array(y_true)
        y_pred = self._to_array(y_pred)

        if per_sample:
            # Compute correlation per sample (row-wise)
            correlations = []
            for i in range(y_true.shape[0]):
                try:
                    corr, _ = pearsonr(y_true[i, :], y_pred[i, :])
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
            return np.mean(correlations) if correlations else 0.0
        else:
            try:
                corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
                return corr if not np.isnan(corr) else 0.0
            except:
                return 0.0

    def compute_spearman(self, y_true: Any, y_pred: Any, per_sample: bool = False) -> float:
        """
        Compute Spearman correlation.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            per_sample: If True, compute per-sample correlation and return mean

        Returns:
            Spearman correlation coefficient
        """
        y_true = self._to_array(y_true)
        y_pred = self._to_array(y_pred)

        if per_sample:
            # Compute correlation per sample (row-wise)
            correlations = []
            for i in range(y_true.shape[0]):
                try:
                    corr, _ = spearmanr(y_true[i, :], y_pred[i, :])
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
            return np.mean(correlations) if correlations else 0.0
        else:
            try:
                corr, _ = spearmanr(y_true.flatten(), y_pred.flatten())
                return corr if not np.isnan(corr) else 0.0
            except:
                return 0.0

    def compute_all(self, y_true: Any, y_pred: Any,
                   metrics: Optional[list] = None) -> Dict[str, float]:
        """
        Compute multiple metrics at once.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            metrics: List of metric names to compute (if None, compute all)

        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = self.available_metrics

        results = {}

        if 'r2_score' in metrics:
            results['r2_score'] = self.compute_r2(y_true, y_pred)
        if 'mse' in metrics:
            results['mse'] = self.compute_mse(y_true, y_pred)
        if 'mae' in metrics:
            results['mae'] = self.compute_mae(y_true, y_pred)
        if 'rmse' in metrics:
            results['rmse'] = self.compute_rmse(y_true, y_pred)
        if 'pearson' in metrics:
            results['pearson'] = self.compute_pearson(y_true, y_pred)
        if 'spearman' in metrics:
            results['spearman'] = self.compute_spearman(y_true, y_pred)
        if 'mean_pearson' in metrics:
            results['mean_pearson'] = self.compute_pearson(y_true, y_pred, per_sample=True)
        if 'mean_spearman' in metrics:
            results['mean_spearman'] = self.compute_spearman(y_true, y_pred, per_sample=True)

        return results

    def get_available_metrics(self) -> list:
        """Get list of available metrics"""
        return self.available_metrics
