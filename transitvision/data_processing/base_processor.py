"""Base data processor module for transit data."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import os
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDataProcessor(ABC):
    """Abstract base class for all data processors.
    
    This class defines the interface for all data processors in the TransitVision
    package. Concrete implementations should inherit from this class and
    implement the abstract methods.
    """
    
    def __init__(
        self,
        raw_data_dir: Optional[Union[str, Path]] = None,
        processed_data_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the BaseDataProcessor.
        
        Args:
            raw_data_dir: Directory containing raw data files.
            processed_data_dir: Directory where processed data will be saved.
            config: Configuration parameters for the processor.
        """
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else Path("data/raw")
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else Path("data/processed")
        self.config = config or {}
        
        # Ensure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    @abstractmethod
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from specified file path.
        
        Args:
            file_path: Path to the raw data file.
            
        Returns:
            DataFrame containing the loaded data.
        """
        pass
    
    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the raw data.
        
        Args:
            data: Raw data as a DataFrame.
            
        Returns:
            Processed DataFrame.
        """
        pass
    
    def save_processed_data(self, data: pd.DataFrame, file_name: str) -> None:
        """Save processed data to file.
        
        Args:
            data: Processed DataFrame to save.
            file_name: Name of the output file.
        """
        file_path = self.processed_data_dir / file_name
        
        # Create directory if it doesn't exist
        os.makedirs(file_path.parent, exist_ok=True)
        
        # Save based on file extension
        if file_path.suffix == ".csv":
            data.to_csv(file_path, index=False)
        elif file_path.suffix == ".parquet":
            data.to_parquet(file_path, index=False)
        elif file_path.suffix in [".pkl", ".pickle"]:
            data.to_pickle(file_path)
        else:
            # Default to parquet if extension not recognized
            logger.warning(f"Unrecognized file extension: {file_path.suffix}. Using parquet format.")
            data.to_parquet(f"{file_path}.parquet", index=False)
        
        logger.info(f"Processed data saved to {file_path}")
    
    def run_pipeline(self, input_file: Union[str, Path], output_file: str) -> pd.DataFrame:
        """Run the full data processing pipeline.
        
        Args:
            input_file: Path to the input file.
            output_file: Name of the output file.
            
        Returns:
            Processed DataFrame.
        """
        logger.info(f"Processing {input_file}")
        
        # Load data
        data = self.load_data(input_file)
        
        # Process data
        processed_data = self.process_data(data)
        
        # Save processed data
        self.save_processed_data(processed_data, output_file)
        
        return processed_data