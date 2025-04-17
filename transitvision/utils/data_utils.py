"""Data utilities for the TransitVision package."""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)


def load_data(
    file_path: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """Load data from various file formats.
    
    Args:
        file_path: Path to the data file.
        **kwargs: Additional arguments passed to the pandas read function.
        
    Returns:
        DataFrame with loaded data.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Determine file type from extension
    ext = file_path.suffix.lower()
    
    try:
        if ext == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif ext == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif ext == '.json':
            return pd.read_json(file_path, **kwargs)
        elif ext in ['.pkl', '.pickle']:
            return pd.read_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def save_data(
    data: pd.DataFrame,
    file_path: Union[str, Path],
    create_dir: bool = True,
    **kwargs
) -> None:
    """Save DataFrame to various file formats.
    
    Args:
        data: DataFrame to save.
        file_path: Path to save the data to.
        create_dir: Whether to create the directory if it doesn't exist.
        **kwargs: Additional arguments passed to the pandas write function.
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist and create_dir is True
    if create_dir and not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine file type from extension
    ext = file_path.suffix.lower()
    
    try:
        if ext == '.csv':
            data.to_csv(file_path, **kwargs)
        elif ext == '.parquet':
            data.to_parquet(file_path, **kwargs)
        elif ext in ['.xlsx', '.xls']:
            data.to_excel(file_path, **kwargs)
        elif ext == '.json':
            data.to_json(file_path, **kwargs)
        elif ext in ['.pkl', '.pickle']:
            data.to_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        logger.info(f"Data saved to {file_path}")
    
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise


def get_file_list(
    directory: Union[str, Path],
    pattern: Optional[str] = None,
    recursive: bool = False
) -> List[Path]:
    """Get list of files in a directory.
    
    Args:
        directory: Directory to search.
        pattern: Optional glob pattern to filter files.
        recursive: Whether to search subdirectories.
        
    Returns:
        List of file paths.
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    if recursive:
        if pattern:
            return list(directory.glob(f"**/{pattern}"))
        else:
            return [p for p in directory.glob("**/*") if p.is_file()]
    else:
        if pattern:
            return list(directory.glob(pattern))
        else:
            return [p for p in directory.glob("*") if p.is_file()]


def parse_gtfs_time(time_str: str) -> pd.Timedelta:
    """Parse GTFS time format to pandas Timedelta.
    
    Args:
        time_str: Time string in GTFS format (HH:MM:SS).
        
    Returns:
        Timedelta object.
    """
    if pd.isna(time_str):
        return pd.NaT
    
    # Extract hours, minutes, seconds
    match = re.match(r"(\d+):(\d+):(\d+)", str(time_str))
    if not match:
        return pd.NaT
    
    hours, minutes, seconds = map(int, match.groups())
    return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)


def convert_datetime_columns(
    data: pd.DataFrame,
    date_columns: List[str],
    format: Optional[str] = None
) -> pd.DataFrame:
    """Convert columns to datetime.
    
    Args:
        data: DataFrame to process.
        date_columns: List of column names to convert.
        format: Optional datetime format to use for parsing.
        
    Returns:
        DataFrame with converted columns.
    """
    df = data.copy()
    
    for column in date_columns:
        if column in df.columns:
            try:
                df[column] = pd.to_datetime(df[column], format=format)
                logger.debug(f"Converted column {column} to datetime")
            except Exception as e:
                logger.warning(f"Failed to convert column {column} to datetime: {str(e)}")
    
    return df


def time_features_from_datetime(
    data: pd.DataFrame,
    datetime_column: str,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """Extract time features from a datetime column.
    
    Args:
        data: DataFrame to process.
        datetime_column: Name of the datetime column.
        features: List of features to extract (year, month, day, dayofweek, hour, etc.).
        
    Returns:
        DataFrame with additional time features.
    """
    df = data.copy()
    
    if datetime_column not in df.columns:
        logger.warning(f"Datetime column {datetime_column} not found in data")
        return df
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        try:
            df[datetime_column] = pd.to_datetime(df[datetime_column])
        except:
            logger.error(f"Failed to convert {datetime_column} to datetime")
            return df
    
    # Default features to extract
    if features is None:
        features = ['year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend']
    
    # Extract features
    dt = df[datetime_column].dt
    
    if 'year' in features:
        df[f"{datetime_column}_year"] = dt.year
    
    if 'month' in features:
        df[f"{datetime_column}_month"] = dt.month
    
    if 'day' in features:
        df[f"{datetime_column}_day"] = dt.day
    
    if 'dayofweek' in features:
        df[f"{datetime_column}_dayofweek"] = dt.dayofweek
    
    if 'hour' in features:
        df[f"{datetime_column}_hour"] = dt.hour
    
    if 'minute' in features:
        df[f"{datetime_column}_minute"] = dt.minute
    
    if 'quarter' in features:
        df[f"{datetime_column}_quarter"] = dt.quarter
    
    if 'is_weekend' in features:
        df[f"{datetime_column}_is_weekend"] = dt.dayofweek >= 5
    
    if 'is_month_start' in features:
        df[f"{datetime_column}_is_month_start"] = dt.is_month_start
    
    if 'is_month_end' in features:
        df[f"{datetime_column}_is_month_end"] = dt.is_month_end
    
    if 'week' in features:
        df[f"{datetime_column}_week"] = dt.isocalendar().week
    
    logger.debug(f"Extracted time features from {datetime_column}")
    
    return df


def split_train_test(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    time_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets.
    
    Args:
        data: DataFrame to split.
        test_size: Proportion of data to use for test set.
        random_state: Random seed for reproducibility.
        time_column: If provided, split based on time (latest values are test).
        
    Returns:
        Tuple of (train_data, test_data).
    """
    if time_column and time_column in data.columns:
        # Sort by time
        sorted_data = data.sort_values(time_column)
        
        # Calculate split point
        split_idx = int(len(sorted_data) * (1 - test_size))
        
        # Split data
        train_data = sorted_data.iloc[:split_idx].copy()
        test_data = sorted_data.iloc[split_idx:].copy()
        
        logger.info(f"Split data by time column {time_column}: {len(train_data)} train, {len(test_data)} test")
    else:
        # Random split
        indices = np.arange(len(data))
        
        if random_state is not None:
            np.random.seed(random_state)
        
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_data = data.iloc[train_indices].copy()
        test_data = data.iloc[test_indices].copy()
        
        logger.info(f"Split data randomly: {len(train_data)} train, {len(test_data)} test")
    
    return train_data, test_data


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Clean column names for consistency.
    
    Args:
        data: DataFrame to process.
        
    Returns:
        DataFrame with cleaned column names.
    """
    df = data.copy()
    
    # Clean column names
    df.columns = [
        name.lower()
            .replace(' ', '_')
            .replace('-', '_')
            .replace('.', '_')
            .replace('/', '_')
            .replace('\\', '_')
            .replace('(', '')
            .replace(')', '')
        for name in df.columns
    ]
    
    return df


def detect_outliers(
    data: pd.DataFrame,
    columns: List[str],
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.Series:
    """Detect outliers in specified columns.
    
    Args:
        data: DataFrame to process.
        columns: List of column names to check for outliers.
        method: Method to use ('zscore', 'iqr').
        threshold: Threshold for outlier detection.
        
    Returns:
        Boolean series indicating outlier rows.
    """
    # Initialize outlier mask
    outlier_mask = pd.Series(False, index=data.index)
    
    for column in columns:
        if column not in data.columns:
            logger.warning(f"Column {column} not found in data")
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(data[column]):
            logger.warning(f"Column {column} is not numeric, skipping outlier detection")
            continue
        
        # Handle NaN values
        values = data[column].dropna()
        
        if method == 'zscore':
            # Z-score method
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                logger.warning(f"Standard deviation is zero for column {column}, skipping")
                continue
            
            z_scores = (data[column] - mean) / std
            column_outliers = (abs(z_scores) > threshold)
            
        elif method == 'iqr':
            # IQR method
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                logger.warning(f"IQR is zero for column {column}, skipping")
                continue
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            column_outliers = (
                (data[column] < lower_bound) | 
                (data[column] > upper_bound)
            )
            
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        # Update overall outlier mask
        outlier_mask = outlier_mask | column_outliers
    
    n_outliers = outlier_mask.sum()
    logger.info(f"Detected {n_outliers} outliers ({n_outliers/len(data):.1%} of data)")
    
    return outlier_mask


def encode_categorical(
    data: pd.DataFrame,
    columns: List[str],
    method: str = 'onehot',
    drop_first: bool = False
) -> pd.DataFrame:
    """Encode categorical variables.
    
    Args:
        data: DataFrame to process.
        columns: List of categorical column names.
        method: Encoding method ('onehot', 'label', 'ordinal').
        drop_first: Whether to drop the first category for one-hot encoding.
        
    Returns:
        DataFrame with encoded variables.
    """
    df = data.copy()
    
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column {column} not found in data")
            continue
        
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(
                df[column], 
                prefix=column, 
                drop_first=drop_first,
                dummy_na=False
            )
            df = pd.concat([df, dummies], axis=1)
            
            # Drop original column
            df = df.drop(columns=[column])
            
        elif method == 'label':
            # Label encoding
            categories = df[column].dropna().unique()
            mapping = {category: i for i, category in enumerate(categories)}
            
            # Apply mapping
            df[f"{column}_encoded"] = df[column].map(mapping)
            
        elif method == 'ordinal':
            # Ordinal encoding (requires predefined order)
            # This is a placeholder - actual implementation would require order information
            logger.warning(f"Ordinal encoding requires predefined order, using label encoding instead")
            
            # Fallback to label encoding
            categories = df[column].dropna().unique()
            mapping = {category: i for i, category in enumerate(categories)}
            
            # Apply mapping
            df[f"{column}_encoded"] = df[column].map(mapping)
            
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
    
    return df


def replace_missing_values(
    data: pd.DataFrame,
    strategy: Dict[str, str]
) -> pd.DataFrame:
    """Replace missing values in the data.
    
    Args:
        data: DataFrame to process.
        strategy: Dictionary mapping column names to replacement strategies
                 ('mean', 'median', 'mode', or a specific value).
        
    Returns:
        DataFrame with replaced missing values.
    """
    df = data.copy()
    
    for column, method in strategy.items():
        if column not in df.columns:
            logger.warning(f"Column {column} not found in data")
            continue
        
        if not df[column].isna().any():
            logger.debug(f"Column {column} has no missing values")
            continue
        
        if method == 'mean':
            if pd.api.types.is_numeric_dtype(df[column]):
                value = df[column].mean()
                df[column] = df[column].fillna(value)
                logger.debug(f"Filled {column} missing values with mean: {value}")
            else:
                logger.warning(f"Cannot use mean for non-numeric column {column}")
                
        elif method == 'median':
            if pd.api.types.is_numeric_dtype(df[column]):
                value = df[column].median()
                df[column] = df[column].fillna(value)
                logger.debug(f"Filled {column} missing values with median: {value}")
            else:
                logger.warning(f"Cannot use median for non-numeric column {column}")
                
        elif method == 'mode':
            value = df[column].mode()[0]
            df[column] = df[column].fillna(value)
            logger.debug(f"Filled {column} missing values with mode: {value}")
            
        else:
            # Use the provided value
            df[column] = df[column].fillna(method)
            logger.debug(f"Filled {column} missing values with specified value: {method}")
    
    return df