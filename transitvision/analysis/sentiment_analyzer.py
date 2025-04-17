"""Transit feedback sentiment analysis module."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Check for optional dependencies
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    logging.warning("NLTK package not available. Basic sentiment analysis will be used.")
    NLTK_AVAILABLE = False

try:
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("Transformers package not available. Basic sentiment analysis will be used.")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment in transit rider feedback.
    
    This class provides methods for analyzing sentiment and extracting topics
    from transit rider feedback data.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the sentiment analyzer.
        
        Args:
            config: Configuration parameters for the analyzer.
        """
        # Default configuration
        default_config = {
            "text_column": "feedback_text",
            "date_column": "feedback_date",
            "rating_column": "rating",
            "route_column": "route_id",
            "stop_column": "stop_id",
            "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
            "topic_extraction_method": "keyword",
            "num_topics": 5,
            "plot_style": "whitegrid",
            "plot_palette": "RdYlGn",
            "plot_figsize": (12, 8),
        }
        
        # Update default config with user-provided config
        self.config = default_config
        if config:
            for key, value in config.items():
                self.config[key] = value
        
        # Set plot style
        sns.set_style(self.config["plot_style"])
        
        # Initialize NLP components if available
        self._initialize_nlp()
    
    def _initialize_nlp(self) -> None:
        """Initialize NLP components based on available packages."""
        self.nlp_ready = False
        
        if NLTK_AVAILABLE:
            try:
                # Download necessary NLTK resources
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                
                # Initialize lemmatizer and stopwords
                self.lemmatizer = WordNetLemmatizer()
                self.stopwords = set(stopwords.words('english'))
                
                logger.info("NLTK components initialized")
                self.nlp_ready = True
            except Exception as e:
                logger.error(f"Error initializing NLTK components: {str(e)}")
        
        self.transformer_ready = False
        if TRANSFORMERS_AVAILABLE:
            try:
                # Initialize sentiment model if specified
                model_name = self.config["sentiment_model"]
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                logger.info(f"Transformer model '{model_name}' initialized")
                self.transformer_ready = True
            except Exception as e:
                logger.error(f"Error initializing transformer model: {str(e)}")
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load processed feedback data for analysis.
        
        Args:
            file_path: Path to the processed feedback data file.
            
        Returns:
            DataFrame containing the feedback data.
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
            elif file_path.suffix in [".pkl", ".pickle"]:
                data = pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded {len(data)} feedback records from {file_path}")
            
            # Validate that the text column exists
            if self.config["text_column"] not in data.columns:
                raise ValueError(f"Text column '{self.config['text_column']}' not found in data")
            
            # Convert date column to datetime if needed
            date_col = self.config["date_column"]
            if date_col in data.columns and not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                data[date_col] = pd.to_datetime(data[date_col])
                logger.info(f"Converted column {date_col} to datetime")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis.
        
        Args:
            text: Raw text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip().lower()
        
        if self.nlp_ready:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(word) 
                for word in tokens 
                if word.isalpha() and word not in self.stopwords
            ]
            
            return " ".join(tokens)
        else:
            # Basic preprocessing if NLTK is not available
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            return " ".join([word for word in text.split() if len(word) > 2])
    
    def _analyze_sentiment_basic(self, text: str) -> Dict[str, float]:
        """Perform basic sentiment analysis using lexicon-based approach.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Dictionary with sentiment scores.
        """
        if pd.isna(text) or not text:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
        
        # Simple sentiment lexicon
        positive_words = {
            "good", "great", "excellent", "awesome", "amazing", "love", "nice",
            "helpful", "comfortable", "clean", "efficient", "convenient", "friendly",
            "perfect", "recommend", "best", "easy", "fast", "reliable", "punctual"
        }
        
        negative_words = {
            "bad", "poor", "terrible", "awful", "horrible", "hate", "dirty",
            "uncomfortable", "slow", "late", "delay", "broken", "rude", "worst",
            "expensive", "difficult", "inconvenient", "unreliable", "disappointing", "crowded"
        }
        
        # Tokenize and count sentiment words
        tokens = text.lower().split()
        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count = sum(1 for word in tokens if word in negative_words)
        total_count = len(tokens)
        
        if total_count == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
        
        # Calculate scores
        pos_score = pos_count / total_count
        neg_score = neg_count / total_count
        neutral_score = 1.0 - pos_score - neg_score
        
        # Calculate compound score (-1 to 1)
        compound = pos_score - neg_score
        
        return {
            "positive": pos_score,
            "negative": neg_score,
            "neutral": neutral_score,
            "compound": compound
        }
    
    def _analyze_sentiment_transformer(self, text: str) -> Dict[str, float]:
        """Perform sentiment analysis using transformer model.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Dictionary with sentiment scores.
        """
        if pd.isna(text) or not text:
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "compound": 0.0}
        
        try:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # DistilBERT SST-2 has 2 classes: negative (0) and positive (1)
            neg_score = scores[0, 0].item()
            pos_score = scores[0, 1].item()
            
            # Calculate compound score (-1 to 1)
            compound = pos_score - neg_score
            
            return {
                "positive": pos_score,
                "negative": neg_score,
                "neutral": 0.0,  # DistilBERT SST-2 doesn't have neutral class
                "compound": compound
            }
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {str(e)}")
            return self._analyze_sentiment_basic(text)
    
    def analyze_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment in feedback text.
        
        Args:
            data: Feedback data DataFrame.
            
        Returns:
            DataFrame with added sentiment analysis results.
        """
        df = data.copy()
        text_col = self.config["text_column"]
        
        # Preprocess text
        df["processed_text"] = df[text_col].apply(self._preprocess_text)
        
        # Select sentiment analysis method
        if self.transformer_ready:
            logger.info("Using transformer model for sentiment analysis")
            sentiment_func = self._analyze_sentiment_transformer
        else:
            logger.info("Using basic lexicon-based sentiment analysis")
            sentiment_func = self._analyze_sentiment_basic
        
        # Apply sentiment analysis
        sentiments = df["processed_text"].apply(sentiment_func)
        
        # Extract sentiment scores
        df["sentiment_positive"] = sentiments.apply(lambda x: x["positive"])
        df["sentiment_negative"] = sentiments.apply(lambda x: x["negative"])
        df["sentiment_neutral"] = sentiments.apply(lambda x: x["neutral"])
        df["sentiment_compound"] = sentiments.apply(lambda x: x["compound"])
        
        # Add sentiment label
        df["sentiment"] = df["sentiment_compound"].apply(
            lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
        )
        
        logger.info(f"Sentiment analysis complete. Distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df
    
    def _extract_topics_keywords(self, data: pd.DataFrame, num_topics: int = 5) -> Dict[str, List[str]]:
        """Extract topics using keyword frequency.
        
        Args:
            data: Feedback data DataFrame with processed text.
            num_topics: Number of topic clusters to extract.
            
        Returns:
            Dictionary of sentiment categories with top keywords.
        """
        topics = {}
        
        # Process for each sentiment category
        for sentiment in ["positive", "negative", "neutral"]:
            # Filter data by sentiment
            sentiment_data = data[data["sentiment"] == sentiment]
            
            if len(sentiment_data) == 0:
                topics[sentiment] = []
                continue
            
            # Combine all processed text
            all_text = " ".join(sentiment_data["processed_text"].tolist())
            
            # Count word frequencies
            word_counts = Counter(all_text.split())
            
            # Get top words as topics
            topics[sentiment] = [word for word, _ in word_counts.most_common(num_topics)]
        
        return topics
    
    def extract_topics(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Extract key topics from feedback.
        
        Args:
            data: Sentiment-analyzed feedback data.
            
        Returns:
            Tuple of (DataFrame with topic features, topic dictionary).
        """
        df = data.copy()
        
        # Check if sentiment analysis has been performed
        if "sentiment" not in df.columns:
            logger.warning("Sentiment analysis not found in data. Running sentiment analysis first.")
            df = self.analyze_sentiment(df)
        
        # Extract topics based on configured method
        method = self.config["topic_extraction_method"]
        num_topics = self.config["num_topics"]
        
        if method == "keyword":
            topics = self._extract_topics_keywords(df, num_topics)
        else:
            logger.warning(f"Unsupported topic extraction method: {method}. Using keyword method.")
            topics = self._extract_topics_keywords(df, num_topics)
        
        # Add topic features to DataFrame
        for sentiment, keywords in topics.items():
            for keyword in keywords:
                feature_name = f"topic_{sentiment}_{keyword}"
                df[feature_name] = df["processed_text"].apply(
                    lambda x: 1 if keyword in x.split() else 0
                )
        
        logger.info(f"Topic extraction complete. Extracted topics: {topics}")
        
        return df, topics
    
    def plot_sentiment_distribution(
        self,
        data: pd.DataFrame,
        groupby: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot sentiment distribution.
        
        Args:
            data: Sentiment-analyzed feedback data.
            groupby: Optional column to group by (e.g., route_id).
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        df = data.copy()
        
        # Check if sentiment analysis has been performed
        if "sentiment" not in df.columns:
            logger.warning("Sentiment analysis not found in data. Running sentiment analysis first.")
            df = self.analyze_sentiment(df)
        
        # Create figure
        figsize = self.config["plot_figsize"]
        fig, ax = plt.subplots(figsize=figsize)
        
        if groupby and groupby in df.columns:
            # Limit to top 10 groups for readability
            if df[groupby].nunique() > 10:
                # Get groups with most feedback
                top_groups = df[groupby].value_counts().head(10).index.tolist()
                plot_data = df[df[groupby].isin(top_groups)]
            else:
                plot_data = df
            
            # Calculate sentiment counts by group
            plot_data = pd.crosstab(plot_data[groupby], plot_data["sentiment"], normalize="index")
            
            # Plot stacked bar chart
            plot_data.plot(
                kind="bar", 
                stacked=True, 
                ax=ax, 
                color=["red", "gray", "green"],
                width=0.8
            )
            
            ax.set_xlabel(groupby.replace("_", " ").title())
            ax.set_ylabel("Proportion")
            ax.set_title(f"Sentiment Distribution by {groupby.replace('_', ' ').title()}")
            
            # Add percentage labels
            for c in ax.containers:
                labels = [f'{v.get_height():.0%}' if v.get_height() > 0.05 else '' for v in c]
                ax.bar_label(c, labels=labels, label_type='center')
            
        else:
            # Plot overall sentiment distribution
            sentiment_counts = df["sentiment"].value_counts(normalize=True)
            
            # Set colors for sentiments
            colors = {"positive": "green", "neutral": "gray", "negative": "red"}
            bar_colors = [colors.get(s, "blue") for s in sentiment_counts.index]
            
            # Plot bar chart
            sentiment_counts.plot(
                kind="bar",
                ax=ax,
                color=bar_colors,
                width=0.6
            )
            
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Proportion")
            ax.set_title("Overall Sentiment Distribution")
            
            # Add percentage labels
            for i, v in enumerate(sentiment_counts):
                ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_sentiment_over_time(
        self,
        data: pd.DataFrame,
        time_grouping: str = "monthly",
        route_filter: Optional[Union[str, List[str]]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot sentiment trends over time.
        
        Args:
            data: Sentiment-analyzed feedback data.
            time_grouping: Time grouping level ("daily", "weekly", "monthly").
            route_filter: Optional route ID(s) to filter by.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        df = data.copy()
        
        # Check if sentiment analysis has been performed
        if "sentiment" not in df.columns:
            logger.warning("Sentiment analysis not found in data. Running sentiment analysis first.")
            df = self.analyze_sentiment(df)
        
        # Check if date column exists
        date_col = self.config["date_column"]
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        
        # Apply route filter if specified
        if route_filter:
            route_col = self.config["route_column"]
            if route_col not in df.columns:
                logger.warning(f"Route column '{route_col}' not found in data. Ignoring route filter.")
            else:
                if isinstance(route_filter, list):
                    df = df[df[route_col].isin(route_filter)]
                else:
                    df = df[df[route_col] == route_filter]
        
        # Create time grouping
        if time_grouping == "daily":
            df["time_group"] = df[date_col].dt.date
        elif time_grouping == "weekly":
            df["time_group"] = df[date_col].dt.to_period("W").dt.start_time
        elif time_grouping == "monthly":
            df["time_group"] = df[date_col].dt.to_period("M").dt.start_time
        else:
            raise ValueError(f"Invalid time grouping: {time_grouping}")
        
        # Group by time and calculate sentiment metrics
        sentiment_over_time = df.groupby("time_group").agg({
            "sentiment_positive": "mean",
            "sentiment_negative": "mean",
            "sentiment_compound": "mean",
            "sentiment": lambda x: (x == "positive").mean()
        }).reset_index()
        
        sentiment_over_time.rename(
            columns={"sentiment": "positive_ratio"}, 
            inplace=True
        )
        
        # Create figure
        figsize = self.config["plot_figsize"]
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot compound sentiment
        ax1.plot(
            sentiment_over_time["time_group"],
            sentiment_over_time["sentiment_compound"],
            marker='o',
            color='blue',
            label="Compound Sentiment"
        )
        
        # Add second y-axis for positive ratio
        ax2 = ax1.twinx()
        ax2.plot(
            sentiment_over_time["time_group"],
            sentiment_over_time["positive_ratio"],
            marker='s',
            color='green',
            linestyle='--',
            label="Positive Ratio"
        )
        
        # Set labels and title
        time_label = time_grouping.capitalize()
        ax1.set_xlabel(f"Time ({time_label})")
        ax1.set_ylabel("Compound Sentiment Score")
        ax2.set_ylabel("Positive Ratio")
        
        title = f"Sentiment Trends Over Time ({time_label})"
        if route_filter:
            route_label = route_filter if isinstance(route_filter, str) else "Selected Routes"
            title = f"{title} - {route_label}"
        
        plt.title(title)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_topic_distribution(
        self,
        data: pd.DataFrame,
        topics: Dict[str, List[str]],
        sentiment: str = "all",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot distribution of extracted topics.
        
        Args:
            data: Sentiment and topic analyzed feedback data.
            topics: Dictionary of topics by sentiment.
            sentiment: Sentiment category to plot topics for ("positive", "negative", "neutral", "all").
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        df = data.copy()
        
        # Create figure
        figsize = self.config["plot_figsize"]
        
        if sentiment == "all":
            # Create subplots for each sentiment
            sentiments = [s for s in ["positive", "negative", "neutral"] if s in topics and topics[s]]
            n_plots = len(sentiments)
            
            if n_plots == 0:
                logger.warning("No topics found for any sentiment category")
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, "No topics available", ha="center", va="center")
                return fig
            
            # Adjust figure size based on number of subplots
            fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], n_plots * 4))
            if n_plots == 1:
                axes = [axes]
            
            for i, sent in enumerate(sentiments):
                if not topics[sent]:
                    continue
                
                # Filter data by sentiment
                sent_data = df[df["sentiment"] == sent]
                
                # Calculate topic frequencies
                topic_freqs = {}
                for keyword in topics[sent]:
                    feature_name = f"topic_{sent}_{keyword}"
                    if feature_name in df.columns:
                        topic_freqs[keyword] = sent_data[feature_name].mean()
                
                if not topic_freqs:
                    axes[i].text(0.5, 0.5, f"No topic features for {sent}", ha="center", va="center")
                    continue
                
                # Sort topics by frequency
                topic_freqs = {k: v for k, v in sorted(topic_freqs.items(), key=lambda item: item[1], reverse=True)}
                
                # Set colors based on sentiment
                colors = {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "gray"
                }
                
                # Plot horizontal bar chart
                axes[i].barh(
                    list(topic_freqs.keys()),
                    list(topic_freqs.values()),
                    color=colors.get(sent, "blue")
                )
                
                # Add percentage labels
                for j, v in enumerate(topic_freqs.values()):
                    axes[i].text(v + 0.01, j, f"{v:.1%}", va="center")
                
                axes[i].set_title(f"{sent.capitalize()} Sentiment Topics")
                axes[i].set_xlabel("Frequency")
                axes[i].set_xlim(0, 1)
        
        else:
            # Plot topics for a single sentiment
            if sentiment not in topics or not topics[sentiment]:
                logger.warning(f"No topics found for sentiment: {sentiment}")
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f"No topics available for {sentiment}", ha="center", va="center")
                return fig
            
            # Filter data by sentiment
            sent_data = df[df["sentiment"] == sentiment]
            
            # Calculate topic frequencies
            topic_freqs = {}
            for keyword in topics[sentiment]:
                feature_name = f"topic_{sentiment}_{keyword}"
                if feature_name in df.columns:
                    topic_freqs[keyword] = sent_data[feature_name].mean()
            
            # Sort topics by frequency
            topic_freqs = {k: v for k, v in sorted(topic_freqs.items(), key=lambda item: item[1], reverse=True)}
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set colors based on sentiment
            colors = {
                "positive": "green",
                "negative": "red",
                "neutral": "gray"
            }
            
            # Plot horizontal bar chart
            ax.barh(
                list(topic_freqs.keys()),
                list(topic_freqs.values()),
                color=colors.get(sentiment, "blue")
            )
            
            # Add percentage labels
            for i, v in enumerate(topic_freqs.values()):
                ax.text(v + 0.01, i, f"{v:.1%}", va="center")
            
            ax.set_title(f"{sentiment.capitalize()} Sentiment Topics")
            ax.set_xlabel("Frequency")
            ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig