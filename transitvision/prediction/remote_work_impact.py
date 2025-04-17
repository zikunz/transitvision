"""Model for predicting impact of remote work on transit ridership."""

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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn package not available")
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow package not available")
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class RemoteWorkImpactModel(BaseModel):
    """Model for predicting impact of remote work on transit ridership.
    
    This class implements models for analyzing and predicting how changes in
    remote work patterns affect transit ridership.
    """
    
    def __init__(
        self,
        model_type: str = "elastic_net",
        model_params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[Union[str, Path]] = None,
        scale_features: bool = True,
        remote_work_column: str = "remote_work_percent",
        time_features: Optional[List[str]] = None,
    ) -> None:
        """Initialize the remote work impact model.
        
        Args:
            model_type: Type of model to use ("elastic_net", "random_forest", "neural_network").
            model_params: Parameters for the model.
            model_dir: Directory to save/load model files.
            scale_features: Whether to standardize features.
            remote_work_column: Name of the column representing remote work percentage.
            time_features: List of time-related feature columns.
        """
        super().__init__(model_params, model_dir)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RemoteWorkImpactModel")
        
        self.model_type = model_type
        self.scale_features = scale_features
        self.remote_work_column = remote_work_column
        self.time_features = time_features or []
        
        # Set default parameters based on model type
        self._set_default_params()
        
        # Initialize model
        self._initialize_model()
    
    def _set_default_params(self) -> None:
        """Set default parameters based on model type."""
        default_params = {
            "elastic_net": {
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "max_iter": 1000,
                "random_state": 42
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "neural_network": {
                "hidden_layers": [64, 32],
                "dropout_rate": 0.2,
                "activation": "relu",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "patience": 10
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
        if self.model_type == "elastic_net":
            # Create elastic net model with polynomial features
            model_steps = []
            
            if self.scale_features:
                model_steps.append(('scaler', StandardScaler()))
            
            # Add polynomial features for interactions
            model_steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))
            
            # Add regressor
            model_steps.append((
                'regressor', 
                ElasticNet(
                    alpha=self.model_params["alpha"],
                    l1_ratio=self.model_params["l1_ratio"],
                    max_iter=self.model_params["max_iter"],
                    random_state=self.model_params["random_state"]
                )
            ))
            
            self.model = Pipeline(model_steps)
            
        elif self.model_type == "random_forest":
            # Create random forest model
            model_steps = []
            
            if self.scale_features:
                model_steps.append(('scaler', StandardScaler()))
            
            # Add regressor
            model_steps.append((
                'regressor', 
                RandomForestRegressor(
                    n_estimators=self.model_params["n_estimators"],
                    max_depth=self.model_params["max_depth"],
                    min_samples_split=self.model_params["min_samples_split"],
                    min_samples_leaf=self.model_params["min_samples_leaf"],
                    random_state=self.model_params["random_state"]
                )
            ))
            
            self.model = Pipeline(model_steps)
            
        elif self.model_type == "neural_network":
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is required for model_type='neural_network'")
            
            # For neural network, we'll create a placeholder
            # The actual model will be built in fit() when we know the input shape
            self.model = None
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def _build_neural_network(self, input_dim: int) -> 'Sequential':
        """Build neural network model architecture.
        
        Args:
            input_dim: Number of input features.
            
        Returns:
            TensorFlow Sequential model.
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.model_params["hidden_layers"][0],
            activation=self.model_params["activation"],
            input_dim=input_dim
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.model_params["dropout_rate"]))
        
        # Hidden layers
        for units in self.model_params["hidden_layers"][1:]:
            model.add(Dense(units, activation=self.model_params["activation"]))
            model.add(BatchNormalization())
            model.add(Dropout(self.model_params["dropout_rate"]))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_params["learning_rate"]),
            loss='mse'
        )
        
        return model
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """Fit the remote work impact model.
        
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
        
        # Handle neural network case separately
        if self.model_type == "neural_network":
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is required for model_type='neural_network'")
            
            # Standardize features if requested
            if self.scale_features:
                scaler = StandardScaler()
                X_array = scaler.fit_transform(X_array)
                self.scaler = scaler
            
            # Build neural network
            input_dim = X_array.shape[1]
            self.model = self._build_neural_network(input_dim)
            
            # Set up early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.model_params["patience"],
                restore_best_weights=True
            )
            
            # Fit model
            self.model.fit(
                X_array, y_array,
                epochs=self.model_params["epochs"],
                batch_size=self.model_params["batch_size"],
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
        else:
            # Fit model using scikit-learn pipeline
            self.model.fit(X_array, y_array)
        
        logger.info(f"Model trained on {X_array.shape[0]} samples with {X_array.shape[1]} features")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make ridership predictions with remote work model.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of predicted ridership values.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Handle neural network case separately
        if self.model_type == "neural_network":
            # Apply scaling if it was used during training
            if self.scale_features and hasattr(self, 'scaler'):
                X_array = self.scaler.transform(X_array)
                
            # Make predictions
            predictions = self.model.predict(X_array, verbose=0)
            
            # Flatten predictions
            predictions = predictions.flatten()
            
        else:
            # Make predictions using scikit-learn pipeline
            predictions = self.model.predict(X_array)
        
        return predictions
    
    def sensitivity_analysis(
        self, 
        X: pd.DataFrame,
        remote_work_values: Optional[List[float]] = None,
        feature_values: Optional[Dict[str, List[float]]] = None,
    ) -> pd.DataFrame:
        """Perform sensitivity analysis of ridership to remote work percentage.
        
        Args:
            X: Baseline features.
            remote_work_values: List of remote work percentages to test.
            feature_values: Dictionary of other features to vary.
            
        Returns:
            DataFrame with sensitivity analysis results.
        """
        if self.model is None:
            raise ValueError("Model must be trained before sensitivity analysis")
        
        if self.remote_work_column not in X.columns:
            raise ValueError(f"Remote work column '{self.remote_work_column}' not found in data")
        
        # Set default remote work values if not provided
        if remote_work_values is None:
            remote_work_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        # Create all combinations of values to test
        scenarios = []
        
        for remote_pct in remote_work_values:
            # Start with baseline features
            baseline = X.iloc[0].copy()
            
            # Set remote work percentage
            baseline[self.remote_work_column] = remote_pct
            
            # Add row for base scenario
            scenarios.append(baseline.to_dict())
            
            # Add additional scenarios for varying other features
            if feature_values:
                for feature, values in feature_values.items():
                    if feature in X.columns:
                        for value in values:
                            scenario = baseline.copy()
                            scenario[feature] = value
                            scenarios.append(scenario)
        
        # Create DataFrame from scenarios
        scenarios_df = pd.DataFrame(scenarios)
        
        # Make predictions
        predictions = self.predict(scenarios_df)
        
        # Add predictions to scenarios
        scenarios_df['prediction'] = predictions
        
        return scenarios_df
    
    def plot_remote_work_impact(
        self,
        sensitivity_results: pd.DataFrame,
        group_by: Optional[str] = None,
        figure_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot impact of remote work on ridership.
        
        Args:
            sensitivity_results: Output from sensitivity_analysis.
            group_by: Optional column to group by.
            figure_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot based on whether to group
        if group_by and group_by in sensitivity_results.columns:
            # Group data
            groups = sensitivity_results[group_by].unique()
            
            # Create plot for each group
            for group in groups:
                group_data = sensitivity_results[sensitivity_results[group_by] == group]
                
                # Plot line for this group
                sns.lineplot(
                    data=group_data,
                    x=self.remote_work_column,
                    y='prediction',
                    label=f"{group_by}={group}",
                    marker='o',
                    ax=ax
                )
        else:
            # Simple line plot
            sns.lineplot(
                data=sensitivity_results,
                x=self.remote_work_column,
                y='prediction',
                marker='o',
                ax=ax
            )
        
        # Set labels and title
        target_name = self.target_name or "Ridership"
        ax.set_xlabel('Remote Work Percentage')
        ax.set_ylabel(target_name)
        ax.set_title(f'Impact of Remote Work on {target_name}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            figure_path = Path(figure_path)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            logger.info(f"Remote work impact plot saved to {figure_path}")
        
        return fig
    
    def scenario_analysis(
        self,
        X: pd.DataFrame,
        scenario_configs: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Analyze different remote work scenarios.
        
        Args:
            X: Baseline features.
            scenario_configs: List of scenario configurations.
            
        Returns:
            DataFrame with scenario analysis results.
        """
        if self.model is None:
            raise ValueError("Model must be trained before scenario analysis")
        
        scenarios = []
        
        # Use the first row as a baseline
        baseline = X.iloc[0].copy()
        
        # Define each scenario
        for i, config in enumerate(scenario_configs):
            scenario = baseline.copy()
            
            # Apply configuration
            for feature, value in config.items():
                if feature in scenario.index:
                    scenario[feature] = value
                else:
                    logger.warning(f"Feature '{feature}' not found in data")
            
            # Add to scenarios
            scenario_dict = scenario.to_dict()
            scenario_dict['scenario_id'] = i
            scenario_dict['scenario_name'] = config.get('name', f"Scenario {i}")
            scenarios.append(scenario_dict)
        
        # Create DataFrame from scenarios
        scenarios_df = pd.DataFrame(scenarios)
        
        # Extract features for prediction
        X_scenario = scenarios_df[[col for col in scenarios_df.columns if col in X.columns]]
        
        # Make predictions
        predictions = self.predict(X_scenario)
        
        # Add predictions to scenarios
        scenarios_df['prediction'] = predictions
        
        return scenarios_df
    
    def plot_scenario_comparison(
        self,
        scenario_results: pd.DataFrame,
        figure_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot comparison of different remote work scenarios.
        
        Args:
            scenario_results: Output from scenario_analysis.
            figure_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by prediction
        plot_data = scenario_results.sort_values('prediction', ascending=False)
        
        # Create bar plot
        sns.barplot(
            data=plot_data,
            x='scenario_name',
            y='prediction',
            ax=ax
        )
        
        # Add values on top of bars
        for i, v in enumerate(plot_data['prediction']):
            ax.text(i, v + 0.1, f"{v:.1f}", ha='center')
        
        # Set labels and title
        target_name = self.target_name or "Ridership"
        ax.set_xlabel('Scenario')
        ax.set_ylabel(target_name)
        ax.set_title(f'Comparison of Remote Work Scenarios - Impact on {target_name}')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        # Add remote work percentage as text on the bars
        if self.remote_work_column in plot_data.columns:
            for i, (_, row) in enumerate(plot_data.iterrows()):
                remote_pct = row[self.remote_work_column]
                ax.text(i, row['prediction'] / 2, f"{remote_pct}%", 
                        ha='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            figure_path = Path(figure_path)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scenario comparison plot saved to {figure_path}")
        
        return fig
    
    def time_series_forecast(
        self,
        X: pd.DataFrame,
        steps: int = 12,
        remote_work_trend: Optional[List[float]] = None,
        time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Forecast ridership over time with changing remote work patterns.
        
        Args:
            X: Baseline features.
            steps: Number of time steps to forecast.
            remote_work_trend: Trend of remote work percentages.
            time_col: Name of the time column for plotting.
            
        Returns:
            DataFrame with forecast results.
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        if not self.time_features:
            logger.warning("No time features specified. Forecast may not capture temporal patterns.")
        
        # Set default remote work trend if not provided
        if remote_work_trend is None:
            # Default: gradual increase over time
            remote_work_trend = list(np.linspace(
                X[self.remote_work_column].iloc[-1],
                min(X[self.remote_work_column].iloc[-1] + 20, 100),
                steps
            ))
        
        # Use the last row as a starting point
        last_row = X.iloc[-1].copy()
        
        # Get time features if any
        time_features = [f for f in self.time_features if f in X.columns]
        
        # Create forecast rows
        forecast_rows = []
        
        for i in range(steps):
            # Create new row
            new_row = last_row.copy()
            
            # Update remote work percentage
            if i < len(remote_work_trend):
                new_row[self.remote_work_column] = remote_work_trend[i]
            else:
                # Use last value if trend is shorter than steps
                new_row[self.remote_work_column] = remote_work_trend[-1]
            
            # Update time features if provided
            if time_features and time_col and time_col in X.columns:
                # Get the last time value
                last_time = X[time_col].iloc[-1]
                
                # Simple increment for the example (in practice, would be more sophisticated)
                new_time = last_time + i + 1
                
                # Add to the row
                new_row[time_col] = new_time
            
            # Add to forecast rows
            forecast_rows.append(new_row)
        
        # Create DataFrame from forecast rows
        forecast_df = pd.DataFrame(forecast_rows)
        
        # Make predictions
        predictions = self.predict(forecast_df)
        
        # Add predictions to forecast
        forecast_df['prediction'] = predictions
        
        return forecast_df
    
    def plot_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        time_col: str,
        value_col: str,
        figure_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot historical data and forecast.
        
        Args:
            historical_data: Historical ridership data.
            forecast_data: Forecast from time_series_forecast.
            time_col: Name of the time column.
            value_col: Name of the ridership value column.
            figure_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        sns.lineplot(
            data=historical_data,
            x=time_col,
            y=value_col,
            marker='o',
            label='Historical',
            ax=ax
        )
        
        # Plot forecast
        sns.lineplot(
            data=forecast_data,
            x=time_col,
            y='prediction',
            marker='s',
            linestyle='--',
            label='Forecast',
            ax=ax
        )
        
        # Add remote work percentage as secondary y-axis
        ax2 = ax.twinx()
        
        # Historical remote work
        if self.remote_work_column in historical_data.columns:
            sns.lineplot(
                data=historical_data,
                x=time_col,
                y=self.remote_work_column,
                color='red',
                alpha=0.6,
                ax=ax2
            )
        
        # Forecast remote work
        sns.lineplot(
            data=forecast_data,
            x=time_col,
            y=self.remote_work_column,
            color='red',
            alpha=0.6,
            linestyle=':',
            ax=ax2
        )
        
        # Set labels and title
        target_name = self.target_name or "Ridership"
        ax.set_xlabel('Time')
        ax.set_ylabel(target_name)
        ax2.set_ylabel('Remote Work Percentage', color='red')
        ax2.tick_params(axis='y', colors='red')
        
        ax.set_title(f'{target_name} Forecast with Remote Work Trends')
        
        # Add vertical line at forecast start
        last_historical_time = historical_data[time_col].iloc[-1]
        ax.axvline(x=last_historical_time, color='gray', linestyle='--', alpha=0.7)
        ax.text(last_historical_time, ax.get_ylim()[1] * 0.95, 'Forecast Start', 
                ha='right', rotation=90, alpha=0.7)
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + ['Remote Work %'], loc='best')
        ax2.get_legend().remove()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            figure_path = Path(figure_path)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {figure_path}")
        
        return fig