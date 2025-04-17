"""Tests for the data_processing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from transitvision.data_processing.base_processor import BaseDataProcessor
from transitvision.data_processing.transit_data_processor import TransitDataProcessor
from transitvision.data_processing.feedback_processor import FeedbackProcessor


class TestBaseDataProcessor:
    """Test the BaseDataProcessor class."""
    
    class ConcreteProcessor(BaseDataProcessor):
        """Concrete implementation of BaseDataProcessor for testing."""
        
        def load_data(self, file_path):
            """Implement abstract method."""
            return pd.DataFrame({'test': [1, 2, 3]})
        
        def process_data(self, data):
            """Implement abstract method."""
            return data
    
    def test_initialization(self):
        """Test initialization of BaseDataProcessor."""
        processor = self.ConcreteProcessor()
        
        # Test default attributes
        assert processor.raw_data_dir == Path("data/raw")
        assert processor.processed_data_dir == Path("data/processed")
        assert isinstance(processor.config, dict)
        
        # Test custom initialization
        custom_processor = self.ConcreteProcessor(
            raw_data_dir="custom/raw",
            processed_data_dir="custom/processed",
            config={"test_key": "test_value"}
        )
        
        assert custom_processor.raw_data_dir == Path("custom/raw")
        assert custom_processor.processed_data_dir == Path("custom/processed")
        assert custom_processor.config["test_key"] == "test_value"
    
    def test_save_processed_data(self):
        """Test saving processed data."""
        processor = self.ConcreteProcessor()
        
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            processor.processed_data_dir = Path(temp_dir)
            
            # Test CSV output
            processor.save_processed_data(test_data, "test_output.csv")
            assert Path(temp_dir, "test_output.csv").exists()
            
            # Test parquet output
            processor.save_processed_data(test_data, "test_output.parquet")
            assert Path(temp_dir, "test_output.parquet").exists()
            
            # Test pickle output
            processor.save_processed_data(test_data, "test_output.pkl")
            assert Path(temp_dir, "test_output.pkl").exists()
    
    def test_run_pipeline(self):
        """Test the full data processing pipeline."""
        processor = self.ConcreteProcessor()
        
        # Mock methods
        processor.load_data = lambda file_path: pd.DataFrame({'test': [1, 2, 3]})
        processor.process_data = lambda data: data.assign(processed=True)
        processor.save_processed_data = lambda data, file_name: None
        
        # Run pipeline
        result = processor.run_pipeline("dummy_input.csv", "dummy_output.csv")
        
        # Check result
        assert isinstance(result, pd.DataFrame)
        assert 'processed' in result.columns
        assert result['processed'].all()


class TestTransitDataProcessor:
    """Test the TransitDataProcessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transit data for testing."""
        return pd.DataFrame({
            'route_id': ['R1', 'R2', 'R1', 'R3', 'R2'],
            'service_date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'],
            'departure_time': ['08:00:00', '08:15:00', '08:30:00', '08:45:00', '09:00:00'],
            'arrival_time': ['08:30:00', '08:45:00', '09:00:00', '09:15:00', '09:30:00'],
            'stop_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
            'ridership': [100, 150, 120, 80, 200],
            'capacity': [200, 200, 200, 150, 250],
            'delay': [2, 5, 0, 10, 3]
        })
    
    def test_initialization(self):
        """Test initialization of TransitDataProcessor."""
        processor = TransitDataProcessor()
        
        # Test default configuration
        assert 'time_columns' in processor.config
        assert 'drop_na_columns' in processor.config
        
        # Test custom configuration
        custom_processor = TransitDataProcessor(
            config={
                "custom_key": "custom_value",
                "time_columns": ["custom_time"]
            }
        )
        
        assert custom_processor.config["custom_key"] == "custom_value"
        assert custom_processor.config["time_columns"] == ["custom_time"]
        # Other defaults should still be present
        assert 'drop_na_columns' in custom_processor.config
    
    def test_convert_time_columns(self, sample_data):
        """Test conversion of time columns."""
        processor = TransitDataProcessor()
        
        result = processor._convert_time_columns(sample_data)
        
        # Check time columns
        for col in processor.config["time_columns"]:
            if col in result.columns:
                assert pd.api.types.is_timedelta64_dtype(result[col])
        
        # Check date columns
        for col in processor.config["date_columns"]:
            if col in result.columns:
                assert pd.api.types.is_datetime64_any_dtype(result[col])
    
    def test_handle_missing_values(self, sample_data):
        """Test handling of missing values."""
        processor = TransitDataProcessor()
        
        # Add some missing values
        data_with_na = sample_data.copy()
        data_with_na.loc[0, 'ridership'] = np.nan
        data_with_na.loc[2, 'capacity'] = np.nan
        
        result = processor._handle_missing_values(data_with_na)
        
        # Check if NaNs are filled
        assert not result['ridership'].isna().any()
        assert not result['capacity'].isna().any()
    
    def test_detect_outliers(self, sample_data):
        """Test outlier detection."""
        processor = TransitDataProcessor()
        
        # Add an outlier
        data_with_outlier = sample_data.copy()
        data_with_outlier.loc[4, 'ridership'] = 1000  # Much higher than others
        
        result = processor._detect_outliers(data_with_outlier)
        
        # The outlier should be replaced with median
        assert result.loc[4, 'ridership'] == data_with_outlier['ridership'].median()
    
    def test_process_data(self, sample_data):
        """Test the full data processing pipeline."""
        processor = TransitDataProcessor()
        
        result = processor.process_data(sample_data)
        
        # Check result shape
        assert len(result) == len(sample_data)
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result['service_date'])
        
        # Check for temporal features
        assert 'service_date_dayofweek' in result.columns


class TestFeedbackProcessor:
    """Test the FeedbackProcessor class."""
    
    @pytest.fixture
    def sample_feedback(self):
        """Create sample feedback data for testing."""
        return pd.DataFrame({
            'feedback_text': [
                "The bus was clean and on time. Great service!",
                "Driver was rude and the bus was late.",
                "Average experience, nothing special.",
                "Very crowded bus, couldn't find a seat.",
                "Love the new schedule, makes my commute easier."
            ],
            'feedback_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'rating': [5, 1, 3, 2, 5],
            'route_id': ['R1', 'R2', 'R1', 'R3', 'R2']
        })
    
    def test_initialization(self):
        """Test initialization of FeedbackProcessor."""
        processor = FeedbackProcessor()
        
        # Test default configuration
        assert 'text_column' in processor.config
        assert processor.config['text_column'] == 'feedback_text'
        
        # Test custom configuration
        custom_processor = FeedbackProcessor(
            config={
                "text_column": "custom_text",
                "min_feedback_length": 10
            }
        )
        
        assert custom_processor.config["text_column"] == "custom_text"
        assert custom_processor.config["min_feedback_length"] == 10
    
    def test_clean_text(self, sample_feedback):
        """Test text cleaning functionality."""
        processor = FeedbackProcessor()
        
        # Add text with URLs and special characters
        text = "Check out https://example.com! My email is user@example.com."
        
        cleaned = processor._clean_text(text)
        
        # Should remove URLs and emails
        assert "https://example.com" not in cleaned
        assert "user@example.com" not in cleaned
        
        # Should convert to lowercase
        assert cleaned == cleaned.lower()
    
    def test_filter_valid_feedback(self, sample_feedback):
        """Test feedback filtering."""
        processor = FeedbackProcessor()
        
        # Add some short and empty feedback
        data_with_short = sample_feedback.copy()
        data_with_short.loc[5] = ["OK", '2023-01-06', 3, 'R1']
        data_with_short.loc[6] = [np.nan, '2023-01-07', 4, 'R2']
        
        result = processor._filter_valid_feedback(data_with_short)
        
        # Short and empty feedback should be removed
        assert len(result) < len(data_with_short)
        assert "OK" not in result['feedback_text'].values
        assert not result['feedback_text'].isna().any()
    
    def test_extract_basic_features(self, sample_feedback):
        """Test extraction of basic text features."""
        processor = FeedbackProcessor()
        
        result = processor._extract_basic_features(sample_feedback)
        
        # Check for basic text features
        assert 'text_char_count' in result.columns
        assert 'text_word_count' in result.columns
        assert 'text_sentence_count' in result.columns
        
        # Check calculations
        assert result.loc[0, 'text_word_count'] == len(sample_feedback.loc[0, 'feedback_text'].split())
    
    def test_process_data(self, sample_feedback):
        """Test the full feedback processing pipeline."""
        processor = FeedbackProcessor()
        
        result = processor.process_data(sample_feedback)
        
        # Check result
        assert len(result) == len(sample_feedback)
        
        # Check for feature columns
        assert 'text_word_count' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['feedback_date'])
        
        # Check for extracted keywords
        keyword_columns = [col for col in result.columns if col.startswith('has_keyword_')]
        assert len(keyword_columns) > 0