"""
Base model class for the benchmark framework
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all perturbation prediction models.

    All models in the benchmark should inherit from this class and implement
    the required methods.
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.

        Args:
            model_name: Name of the model
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        logger.info(f"Initialized {model_name}")

    @abstractmethod
    def train(self, train_data: Any, **kwargs) -> None:
        """
        Train the model on the provided data.

        Args:
            train_data: Training data (format depends on model)
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, test_data: Any, **kwargs) -> Any:
        """
        Generate predictions for the test data.

        Args:
            test_data: Test data (format depends on model)
            **kwargs: Additional prediction parameters

        Returns:
            Predictions (format depends on model)
        """
        pass

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        logger.info(f"Saving {self.model_name} to {path}")
        # Default implementation - subclasses can override
        raise NotImplementedError(f"Save not implemented for {self.model_name}")

    def load(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        logger.info(f"Loading {self.model_name} from {path}")
        # Default implementation - subclasses can override
        raise NotImplementedError(f"Load not implemented for {self.model_name}")

    def get_name(self) -> str:
        """Get the model name"""
        return self.model_name

    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration"""
        return self.config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
