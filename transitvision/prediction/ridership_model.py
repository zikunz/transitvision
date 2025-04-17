"""Transit ridership prediction model."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .base_model import BaseModel

# Check for optional dependencies
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn package not available")
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logging.warning("XGBoost package not available")
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class RidershipModel(BaseModel):
    """Model for predicting transit ridership.
    
    This class implements various machine learning models for predicting
    ridership patterns on transit routes.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[Union[str, Path]] = None,
        scale_features: bool = True,
    ) -> None:
        """Initialize the ridership prediction model.
        
        Args:
            model_type: Type of model to use ("random_forest", "gradient_boosting", 
                       "linear", "ridge", "lasso", "xgboost").
            model_params: Parameters for the model.
            model_dir: Directory to save/load model files.
            scale_features: Whether to standardize features.
        """
        super().__init__(model_params, model_dir)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RidershipModel")
        
        self.model_type = model_type
        self.scale_features = scale_features
        
        # Set default parameters based on model type
        self._set_default_params()
        
        # Initialize model
        self._initialize_model()
    
    def _set_default_params(self) -> None:
        """Set default parameters based on model type."""
        default_params = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.8,
                "random_state": 42
            },
            "linear": {},
            "ridge": {
                "alpha": 1.0,
                "random_state": 42
            },
            "lasso": {
                "alpha": 0.1,
                "random_state": 42
            },
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
        }
        
        # Set default parameters if not provided
        if not self.model_params and self.model_type in default_params:
            self.model_params = default_params[self.model_type]
        
        # Update default parameters with user-provided parameters
        elif self.model_type in default_params:
            for key, value in default_params[self.model_type].items():
                if key not in self.model_params:
                    self.model_params[key] = value
    
    def _initialize_model(self) -> None:
        """Initialize the model based on model_type."""
        # Create base model based on model type
        if self.model_type == "random_forest":
            base_model = RandomForestRegressor(**self.model_params)
        elif self.model_type == "gradient_boosting":
            base_model = GradientBoostingRegressor(**self.model_params)
        elif self.model_type == "linear":
            base_model = LinearRegression(**self.model_params)
        elif self.model_type == "ridge":
            base_model = Ridge(**self.model_params)
        elif self.model_type == "lasso":
            base_model = Lasso(**self.model_params)
        elif self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is required for model_type='xgboost'")
            base_model = xgb.XGBRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline with optional scaling
        if self.scale_features:
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', base_model)
            ])
        else:
            self.model = base_model
        
        logger.info(f"Initialized {self.model_type} model")
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """Fit the ridership prediction model.
        
        Args:
            X: Training features.
            y: Target ridership values.
        """
        # Convert to numpy arrays if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Store target name if available
        if isinstance(y, pd.Series):
            self.target_name = y.name
        
        # Fit the model
        self.model.fit(X_array, y_array)
        
        logger.info(f"Model trained on {X_array.shape[0]} samples with {X_array.shape[1]} features")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make ridership predictions.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of predicted ridership values.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Make predictions
        predictions = self.model.predict(X_array)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Check if feature names are available
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        # Extract feature importance
        if self.scale_features:
            # Get regressor from pipeline
            regressor = self.model.named_steps['regressor']
        else:
            regressor = self.model
        
        # Extract based on model type
        if hasattr(regressor, 'feature_importances_'):
            importance = regressor.feature_importances_
        elif hasattr(regressor, 'coef_'):
            importance = np.abs(regressor.coef_)
        else:
            raise ValueError(f"Feature importance not available for {self.model_type} model")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_predictions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        time_column: Optional[Union[str, np.ndarray]] = None,
        figure_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot actual vs. predicted ridership over time.
        
        Args:
            X: Features to predict on.
            y: Actual ridership values.
            time_column: Time values for x-axis (column name or array).
            figure_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        if self.model is None:
            raise ValueError("Model must be trained before plotting predictions")
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Convert to arrays
        y_true = np.array(y).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Get time values if provided as column name
        if isinstance(time_column, str) and isinstance(X, pd.DataFrame):
            if time_column in X.columns:
                time_values = X[time_column].values
            else:
                time_values = np.arange(len(y_true))
                logger.warning(f"Time column '{time_column}' not found in data")
        elif time_column is not None:
            time_values = np.array(time_column)
        else:
            time_values = np.arange(len(y_true))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual and predicted values
        ax.plot(time_values, y_true, label='Actual', marker='o', alpha=0.7)
        ax.plot(time_values, y_pred, label='Predicted', marker='s', alpha=0.7)
        
        # Calculate and add metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        metrics_text = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}"
        ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, 
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Set labels and title
        target_name = self.target_name or "Ridership"
        ax.set_xlabel('Time')
        ax.set_ylabel(target_name)
        ax.set_title(f'{target_name} Prediction')
        ax.legend()
        
        # Set x-axis ticks
        if len(time_values) > 20:
            # Limit x-ticks for readability
            tick_indices = np.linspace(0, len(time_values) - 1, 10, dtype=int)
            ax.set_xticks(time_values[tick_indices])
            ax.set_xticklabels([str(time_values[i]) for i in tick_indices], rotation=45)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            figure_path = Path(figure_path)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to {figure_path}")
        
        return fig