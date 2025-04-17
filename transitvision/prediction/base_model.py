"""Base model module for transit prediction."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import joblib
from abc import ABC, abstractmethod
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all prediction models.
    
    This class defines the interface for all prediction models in the TransitVision
    package. Concrete implementations should inherit from this class and
    implement the abstract methods.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the base model.
        
        Args:
            model_params: Parameters for the model.
            model_dir: Directory to save/load model files.
        """
        self.model_params = model_params or {}
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model = None
        self.feature_names = None
        self.target_name = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    @abstractmethod
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """Fit the model to the training data.
        
        Args:
            X: Training features.
            y: Target values.
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with the trained model.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of predictions.
        """
        pass
    
    def save_model(self, filename: str) -> str:
        """Save the trained model to a file.
        
        Args:
            filename: Name of the file to save the model to.
            
        Returns:
            Path to the saved model file.
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        file_path = self.model_dir / filename
        
        # Create directory if it doesn't exist
        os.makedirs(file_path.parent, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, file_path)
        logger.info(f"Model saved to {file_path}")
        
        return str(file_path)
    
    def load_model(self, filename: str) -> None:
        """Load a trained model from a file.
        
        Args:
            filename: Name of the file to load the model from.
        """
        file_path = self.model_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        self.model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
    
    def evaluate(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        plot: bool = False,
        figure_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X: Test features.
            y: True target values.
            plot: Whether to generate evaluation plots.
            figure_path: Path to save the evaluation plot.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Convert to 1D arrays
        y_true = np.array(y).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Calculate metrics
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Add small constant to avoid division by zero
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        
        # Generate evaluation plot if requested
        if plot:
            self._plot_evaluation(y_true, y_pred, metrics, figure_path)
        
        return metrics
    
    def _plot_evaluation(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        metrics: Dict[str, float],
        figure_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Generate evaluation plots.
        
        Args:
            y_true: True target values.
            y_pred: Predicted values.
            metrics: Evaluation metrics.
            figure_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot actual vs predicted
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'k--', lw=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Actual vs. Predicted')
        
        # Add metrics as text
        metrics_text = "\n".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Plot residuals
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.axvline(x=0, color='k', linestyle='--')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            figure_path = Path(figure_path)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plot saved to {figure_path}")
        
        return fig
    
    def plot_feature_importance(
        self, 
        importance: np.ndarray,
        feature_names: List[str],
        figure_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot feature importance.
        
        Args:
            importance: Feature importance values.
            feature_names: Names of features.
            figure_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Create sorted feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Limit to top 20 features for readability
        if len(importance_df) > 20:
            importance_df = importance_df.head(20)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        sns.barplot(
            data=importance_df,
            y='feature',
            x='importance',
            ax=ax
        )
        
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            figure_path = Path(figure_path)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {figure_path}")
        
        return fig