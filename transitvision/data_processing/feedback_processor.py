"""Transit rider feedback processor module."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import re
from collections import Counter

from .base_processor import BaseDataProcessor

logger = logging.getLogger(__name__)


class FeedbackProcessor(BaseDataProcessor):
    """Processor for transit rider feedback data.
    
    This class implements methods for cleaning and preprocessing textual
    feedback from transit riders, preparing it for sentiment analysis and
    topic extraction.
    """
    
    def __init__(
        self,
        raw_data_dir: Optional[Union[str, Path]] = None,
        processed_data_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the FeedbackProcessor.
        
        Args:
            raw_data_dir: Directory containing raw feedback data files.
            processed_data_dir: Directory where processed data will be saved.
            config: Configuration parameters for the processor.
        """
        super().__init__(raw_data_dir, processed_data_dir, config)
        
        # Default configuration for feedback processing
        default_config = {
            "text_column": "feedback_text",
            "date_column": "feedback_date",
            "rating_column": "rating",
            "route_column": "route_id",
            "min_feedback_length": 5,  # Minimum number of words
            "max_text_length": 1000,   # Maximum number of characters
            "remove_urls": True,
            "remove_emails": True,
            "remove_special_chars": True,
            "lowercase": True
        }
        
        # Update default config with user-provided config
        if config:
            for key, value in config.items():
                default_config[key] = value
        
        self.config = default_config
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load feedback data from specified file path.
        
        Args:
            file_path: Path to the feedback data file.
            
        Returns:
            DataFrame containing the loaded feedback data.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Feedback data file not found: {file_path}")
        
        try:
            # Load based on file extension
            if file_path.suffix == ".csv":
                data = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                data = pd.read_parquet(file_path)
            elif file_path.suffix in [".xlsx", ".xls"]:
                data = pd.read_excel(file_path)
            elif file_path.suffix == ".json":
                data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Validate that the text column exists
            if self.config["text_column"] not in data.columns:
                raise ValueError(f"Text column '{self.config['text_column']}' not found in data")
            
            logger.info(f"Loaded {len(data)} feedback records from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading feedback data from {file_path}: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text data.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip()
        
        # Convert to lowercase if specified
        if self.config["lowercase"]:
            text = text.lower()
        
        # Remove URLs
        if self.config["remove_urls"]:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        if self.config["remove_emails"]:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters
        if self.config["remove_special_chars"]:
            text = re.sub(r'[^\w\s]', '', text)
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _filter_valid_feedback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid or too short feedback.
        
        Args:
            data: Feedback data DataFrame.
            
        Returns:
            DataFrame with only valid feedback entries.
        """
        df = data.copy()
        initial_count = len(df)
        
        # Filter out empty feedback
        df = df[df[self.config["text_column"]].notna()]
        na_filtered_count = len(df)
        
        # Filter by minimum length (word count)
        min_length = self.config["min_feedback_length"]
        df = df[df[self.config["text_column"]].apply(lambda x: len(str(x).split()) >= min_length)]
        length_filtered_count = len(df)
        
        # Truncate too long feedback
        max_length = self.config["max_text_length"]
        df[self.config["text_column"]] = df[self.config["text_column"]].apply(
            lambda x: str(x)[:max_length] if len(str(x)) > max_length else x
        )
        
        # Log filtering results
        na_removed = initial_count - na_filtered_count
        short_removed = na_filtered_count - length_filtered_count
        
        if na_removed > 0:
            logger.info(f"Removed {na_removed} rows with missing feedback text")
        
        if short_removed > 0:
            logger.info(f"Removed {short_removed} rows with feedback shorter than {min_length} words")
        
        return df
    
    def _extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features from feedback text.
        
        Args:
            data: Feedback data DataFrame.
            
        Returns:
            DataFrame with additional text features.
        """
        df = data.copy()
        
        # Calculate text length features
        text_col = self.config["text_column"]
        df["text_char_count"] = df[text_col].apply(lambda x: len(str(x)))
        df["text_word_count"] = df[text_col].apply(lambda x: len(str(x).split()))
        df["text_sentence_count"] = df[text_col].apply(lambda x: len(re.split(r'[.!?]+', str(x))))
        
        # Calculate average word length
        df["avg_word_length"] = df.apply(
            lambda row: np.mean([len(word) for word in str(row[text_col]).split()]) 
            if row["text_word_count"] > 0 else 0, 
            axis=1
        )
        
        # Extract exclamation and question mark counts (potential sentiment indicators)
        df["exclamation_count"] = df[text_col].apply(lambda x: str(x).count('!'))
        df["question_count"] = df[text_col].apply(lambda x: str(x).count('?'))
        
        logger.info("Extracted basic text features")
        
        return df
    
    def _extract_common_keywords(self, data: pd.DataFrame, top_n: int = 20) -> Tuple[List[str], pd.DataFrame]:
        """Extract common keywords from feedback.
        
        Args:
            data: Feedback data DataFrame.
            top_n: Number of top keywords to extract.
            
        Returns:
            Tuple of (list of top keywords, DataFrame with keyword columns).
        """
        df = data.copy()
        text_col = self.config["text_column"]
        
        # Combine all words
        all_words = []
        for text in df[text_col]:
            words = str(text).lower().split()
            all_words.extend(words)
        
        # Count words and get top N
        word_counts = Counter(all_words)
        # Filter out very short words (likely stopwords)
        word_counts = {word: count for word, count in word_counts.items() if len(word) > 2}
        top_keywords = [word for word, _ in word_counts.most_common(top_n)]
        
        # Create binary features for each keyword
        for keyword in top_keywords:
            df[f"has_keyword_{keyword}"] = df[text_col].apply(
                lambda x: 1 if keyword in str(x).lower().split() else 0
            )
        
        logger.info(f"Extracted {len(top_keywords)} common keywords as features")
        
        return top_keywords, df
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the feedback data.
        
        Applies text cleaning, filtering, and feature extraction to prepare
        the feedback data for sentiment analysis and topic modeling.
        
        Args:
            data: Raw feedback data as a DataFrame.
            
        Returns:
            Processed DataFrame ready for analysis.
        """
        logger.info("Starting feedback data processing pipeline")
        
        # Apply processing steps
        df = data.copy()
        
        # Clean text data
        text_col = self.config["text_column"]
        df[text_col] = df[text_col].apply(self._clean_text)
        
        # Filter valid feedback
        df = self._filter_valid_feedback(df)
        
        # Extract basic features
        df = self._extract_basic_features(df)
        
        # Extract common keywords
        _, df = self._extract_common_keywords(df)
        
        # Convert date column if it exists
        date_col = self.config["date_column"]
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                logger.info(f"Converted {date_col} to datetime")
            except Exception as e:
                logger.warning(f"Failed to convert date column {date_col}: {str(e)}")
        
        # Log processing summary
        logger.info(f"Feedback data processing complete. Output shape: {df.shape}")
        
        return df