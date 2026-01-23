"""
Data loading and preprocessing utilities
"""

import scanpy as sc
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for benchmark datasets"""

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize data loader.

        Args:
            config: Configuration object
        """
        from ..config import Config
        self.config = config or Config()

    def load_dataset(self, task: str, dataset_name: str):
        """
        Load a dataset for a specific task.

        Args:
            task: Task name (task1, task2, task3)
            dataset_name: Name of the dataset

        Returns:
            AnnData object
        """
        dataset_config = self.config.get_dataset_config(task)
        data_dir = Path(dataset_config['data_dir'])

        # Find the dataset
        dataset_info = None
        for ds in dataset_config['datasets']:
            if ds['name'] == dataset_name:
                dataset_info = ds
                break

        if dataset_info is None:
            raise ValueError(f"Dataset '{dataset_name}' not found for {task}")

        # Load the h5ad file
        file_path = data_dir / dataset_info['file']
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading dataset: {file_path}")
        adata = sc.read_h5ad(file_path)
        logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

        return adata

    def get_dataset_info(self, task: str, dataset_name: str) -> Dict[str, Any]:
        """
        Get metadata about a dataset.

        Args:
            task: Task name
            dataset_name: Dataset name

        Returns:
            Dataset information dictionary
        """
        dataset_config = self.config.get_dataset_config(task)

        for ds in dataset_config['datasets']:
            if ds['name'] == dataset_name:
                return ds

        raise ValueError(f"Dataset '{dataset_name}' not found for {task}")

    def list_datasets(self, task: str) -> list:
        """
        List all datasets for a task.

        Args:
            task: Task name

        Returns:
            List of dataset names
        """
        dataset_config = self.config.get_dataset_config(task)
        return [ds['name'] for ds in dataset_config['datasets']]
