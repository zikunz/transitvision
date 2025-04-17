"""Transit data processor implementation."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import re
from datetime import datetime

from .base_processor import BaseDataProcessor

logger = logging.getLogger(__name__)


class TransitDataProcessor(BaseDataProcessor):
    """Processor for transit ridership and schedule data.
    
    This class implements specific data processing methods for transit
    ridership and schedule data, including cleaning, normalization,
    and feature engineering.
    """
    
    def __init__(
        self,
        raw_data_dir: Optional[Union[str, Path]] = None,
        processed_data_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the TransitDataProcessor.
        
        Args:
            raw_data_dir: Directory containing raw transit data files.
            processed_data_dir: Directory where processed data will be saved.
            config: Configuration parameters for the processor.
        """
        super().__init__(raw_data_dir, processed_data_dir, config)
        
        # Default configuration for transit data processing
        default_config = {
            "time_columns": ["departure_time", "arrival_time"],
            "categorical_columns": ["route_id", "service_id", "trip_id", "stop_id"],
            "numerical_columns": ["ridership", "capacity", "delay"],
            "date_columns": ["service_date", "schedule_date"],
            "drop_na_columns": ["route_id", "stop_id"],
            "outlier_threshold": 3.0,  # Standard deviations for outlier detection
        }
        
        # Update default config with user-provided config
        if config:
            for key, value in config.items():
                default_config[key] = value
        
        self.config = default_config
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load transit data from specified file path.
        
        Supports CSV, Parquet, and Excel formats based on file extension.
        
        Args:
            file_path: Path to the transit data file.
            
        Returns:
            DataFrame containing the loaded transit data.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Load based on file extension
            if file_path.suffix == ".csv":
                data = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                data = pd.read_parquet(file_path)
            elif file_path.suffix in [".xlsx", ".xls"]:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def _convert_time_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert time columns to datetime or timedelta.
        
        Args:
            data: Transit data DataFrame.
            
        Returns:
            DataFrame with converted time columns.
        """
        df = data.copy()
        
        for col in self.config["time_columns"]:
            if col in df.columns:
                # Check if the column contains HH:MM:SS format
                if df[col].dtype == object and df[col].iloc[0] and ":" in str(df[col].iloc[0]):
                    try:
                        # Handle GTFS time format that can exceed 24 hours
                        def convert_gtfs_time(time_str):
                            if pd.isna(time_str):
                                return np.nan
                            
                            # Extract hours, minutes, seconds
                            match = re.match(r"(\d+):(\d+):(\d+)", str(time_str))
                            if not match:
                                return np.nan
                            
                            hours, minutes, seconds = map(int, match.groups())
                            return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
                        
                        df[col] = df[col].apply(convert_gtfs_time)
                        logger.info(f"Converted column {col} to timedelta")
                    except Exception as e:
                        logger.warning(f"Failed to convert time column {col}: {str(e)}")
        
        # Convert date columns to datetime
        for col in self.config["date_columns"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.info(f"Converted column {col} to datetime")
                except Exception as e:
                    logger.warning(f"Failed to convert date column {col}: {str(e)}")
        
        return df
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            data: Transit data DataFrame.
            
        Returns:
            DataFrame with handled missing values.
        """
        df = data.copy()
        
        # Drop rows with missing values in critical columns
        if self.config["drop_na_columns"]:
            initial_rows = len(df)
            df = df.dropna(subset=self.config["drop_na_columns"])
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing values in {self.config['drop_na_columns']}")
        
        # Fill numerical missing values with median
        for col in self.config["numerical_columns"]:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {df[col].isna().sum()} missing values in {col} with median {median_val}")
        
        # Fill categorical missing values with most frequent value
        for col in self.config["categorical_columns"]:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Filled {df[col].isna().sum()} missing values in {col} with mode {mode_val}")
        
        return df
    
    def _detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in numerical columns.
        
        Uses Z-score method to identify outliers beyond the configured threshold.
        
        Args:
            data: Transit data DataFrame.
            
        Returns:
            DataFrame with handled outliers.
        """
        df = data.copy()
        threshold = self.config["outlier_threshold"]
        
        for col in self.config["numerical_columns"]:
            if col in df.columns:
                # Calculate z-scores
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                
                # Identify outliers
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"Found {outlier_count} outliers in {col}")
                    
                    # Replace outliers with the column median
                    median_val = df[col].median()
                    df.loc[outliers, col] = median_val
                    logger.info(f"Replaced outliers in {col} with median {median_val}")
        
        return df
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            data: Transit data DataFrame.
            
        Returns:
            DataFrame with encoded categorical features.
        """
        df = data.copy()
        
        for col in self.config["categorical_columns"]:
            if col in df.columns and df[col].dtype == object:
                # For ID columns, we don't want to one-hot encode
                if col.endswith("_id"):
                    # Just convert to category for efficiency
                    df[col] = df[col].astype("category")
                else:
                    # One-hot encode other categorical columns
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    
                    # Drop the original column
                    df = df.drop(columns=[col])
                    logger.info(f"One-hot encoded {col}, created {len(dummies.columns)} new features")
        
        return df
    
    def _extract_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from datetime columns.
        
        Args:
            data: Transit data DataFrame.
            
        Returns:
            DataFrame with additional temporal features.
        """
        df = data.copy()
        
        # Process date columns to extract temporal features
        for col in self.config["date_columns"]:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                # Extract basic date components
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                
                # Weekend indicator
                df[f"{col}_is_weekend"] = df[col].dt.dayofweek >= 5
                
                # Quarter and season
                df[f"{col}_quarter"] = df[col].dt.quarter
                
                logger.info(f"Extracted temporal features from {col}")
        
        return df
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the transit data.
        
        Applies a series of transformations to clean and prepare the transit data
        for analysis and modeling.
        
        Args:
            data: Raw transit data as a DataFrame.
            
        Returns:
            Processed DataFrame ready for analysis or modeling.
        """
        logger.info("Starting data processing pipeline")
        
        # Apply processing steps
        df = data.copy()
        df = self._convert_time_columns(df)
        df = self._handle_missing_values(df)
        df = self._detect_outliers(df)
        df = self._extract_temporal_features(df)
        
        # Apply encoding if not specified otherwise in config
        if self.config.get("encode_categorical", True):
            df = self._encode_categorical_features(df)
        
        # Log processing summary
        logger.info(f"Data processing complete. Output shape: {df.shape}")
        
        return df