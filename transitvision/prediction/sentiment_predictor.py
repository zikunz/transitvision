"""Sentiment prediction model for transit feedback."""

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
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn package not available")
    SKLEARN_AVAILABLE = False

try:
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from torch.utils.data import DataLoader, Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("Transformers package not available")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentimentPredictor(BaseModel):
    """Model for predicting sentiment in transit rider feedback.
    
    This class implements methods for predicting sentiment in textual feedback
    from transit riders, ranging from traditional ML to transformer-based approaches.
    """
    
    def __init__(
        self,
        model_type: str = "logistic",
        model_params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[Union[str, Path]] = None,
        vectorizer_type: str = "tfidf",
        vectorizer_params: Optional[Dict[str, Any]] = None,
        pretrained_model: Optional[str] = None,
    ) -> None:
        """Initialize the sentiment prediction model.
        
        Args:
            model_type: Type of model to use ("logistic", "random_forest", "transformer").
            model_params: Parameters for the model.
            model_dir: Directory to save/load model files.
            vectorizer_type: Type of text vectorizer ("tfidf", "count", None for transformer).
            vectorizer_params: Parameters for the vectorizer.
            pretrained_model: Pretrained transformer model name.
        """
        super().__init__(model_params, model_dir)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SentimentPredictor")
        
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.vectorizer_params = vectorizer_params or {}
        self.pretrained_model = pretrained_model
        
        # Set default parameters based on model type
        self._set_default_params()
        
        # Initialize model
        self._initialize_model()
    
    def _set_default_params(self) -> None:
        """Set default parameters based on model type."""
        # Default model parameters
        default_model_params = {
            "logistic": {
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_split": 5,
                "class_weight": "balanced",
                "random_state": 42
            },
            "transformer": {
                "batch_size": 16,
                "learning_rate": 2e-5,
                "epochs": 4,
                "max_length": 128
            }
        }
        
        # Default vectorizer parameters
        default_vectorizer_params = {
            "tfidf": {
                "max_features": 5000,
                "min_df": 5,
                "max_df": 0.8,
                "ngram_range": (1, 2)
            },
            "count": {
                "max_features": 5000,
                "min_df": 5,
                "max_df": 0.8,
                "ngram_range": (1, 2)
            }
        }
        
        # Set default model parameters if not provided
        if not self.model_params and self.model_type in default_model_params:
            self.model_params = default_model_params[self.model_type]
        
        # Update default model parameters with user-provided parameters
        elif self.model_type in default_model_params:
            for key, value in default_model_params[self.model_type].items():
                if key not in self.model_params:
                    self.model_params[key] = value
        
        # Set default vectorizer parameters if not provided
        if (not self.vectorizer_params and 
            self.vectorizer_type and 
            self.vectorizer_type in default_vectorizer_params):
            
            self.vectorizer_params = default_vectorizer_params[self.vectorizer_type]
        
        # Update default vectorizer parameters with user-provided parameters
        elif (self.vectorizer_type and 
              self.vectorizer_type in default_vectorizer_params):
            
            for key, value in default_vectorizer_params[self.vectorizer_type].items():
                if key not in self.vectorizer_params:
                    self.vectorizer_params[key] = value
    
    def _initialize_model(self) -> None:
        """Initialize the model based on model_type."""
        # For transformer models, we'll create the model during fit
        if self.model_type == "transformer":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers is required for model_type='transformer'")
            
            # Set default pretrained model if not provided
            if not self.pretrained_model:
                self.pretrained_model = "distilbert-base-uncased-finetuned-sst-2-english"
            
            # We'll initialize the transformer model during fit
            self.model = None
            self.tokenizer = None
            
            logger.info(f"Will use transformer model: {self.pretrained_model}")
            return
        
        # Create pipeline for traditional ML models
        model_steps = []
        
        # Add vectorizer if specified
        if self.vectorizer_type == "tfidf":
            model_steps.append(('vectorizer', TfidfVectorizer(**self.vectorizer_params)))
        elif self.vectorizer_type == "count":
            model_steps.append(('vectorizer', CountVectorizer(**self.vectorizer_params)))
        
        # Add classifier
        if self.model_type == "logistic":
            model_steps.append((
                'classifier',
                LogisticRegression(
                    C=self.model_params["C"],
                    class_weight=self.model_params["class_weight"],
                    max_iter=self.model_params["max_iter"],
                    random_state=self.model_params["random_state"]
                )
            ))
        elif self.model_type == "random_forest":
            model_steps.append((
                'classifier',
                RandomForestClassifier(
                    n_estimators=self.model_params["n_estimators"],
                    max_depth=self.model_params["max_depth"],
                    min_samples_split=self.model_params["min_samples_split"],
                    class_weight=self.model_params["class_weight"],
                    random_state=self.model_params["random_state"]
                )
            ))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline
        self.model = Pipeline(model_steps)
        
        logger.info(f"Initialized {self.model_type} model with {self.vectorizer_type} vectorizer")
    
    def fit(
        self, 
        X: Union[pd.Series, List[str], np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """Fit the sentiment prediction model.
        
        Args:
            X: Training text data.
            y: Target sentiment labels.
        """
        # Convert inputs to appropriate format
        if isinstance(X, pd.Series):
            X_data = X.values
            self.feature_names = [X.name]
        elif isinstance(X, pd.DataFrame):
            # Assume single text column
            if len(X.columns) != 1:
                raise ValueError("For text input, DataFrame must have exactly one column")
            X_data = X.iloc[:, 0].values
            self.feature_names = [X.columns[0]]
        else:
            X_data = X
            self.feature_names = ["text"]
        
        # Convert labels to appropriate format
        if isinstance(y, pd.Series):
            y_data = y.values
            self.target_name = y.name
        else:
            y_data = y
            self.target_name = "sentiment"
        
        # Store label mapping
        self.classes_ = np.unique(y_data)
        
        # Handle transformer model separately
        if self.model_type == "transformer":
            return self._fit_transformer(X_data, y_data)
        
        # Fit traditional ML model
        self.model.fit(X_data, y_data)
        
        logger.info(f"Model trained on {len(X_data)} samples")
    
    def _fit_transformer(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit transformer-based sentiment model.
        
        Args:
            X: Training text data.
            y: Target sentiment labels.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required for model_type='transformer'")
        
        # Create label mapping
        label_dict = {label: i for i, label in enumerate(self.classes_)}
        id2label = {i: label for label, i in label_dict.items()}
        
        # Convert labels to IDs
        y_ids = np.array([label_dict[label] for label in y])
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model,
            num_labels=len(self.classes_),
            id2label=id2label,
            label2id=label_dict
        )
        
        # Create PyTorch dataset
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        # Create dataset and dataloader
        train_dataset = TextDataset(
            X, y_ids, self.tokenizer, 
            self.model_params["max_length"]
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.model_params["batch_size"],
            shuffle=True
        )
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_params["learning_rate"]
        )
        
        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        
        for epoch in range(self.model_params["epochs"]):
            total_loss = 0
            
            for batch in train_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.model_params['epochs']}, Loss: {avg_loss:.4f}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Transformer model trained on {len(X)} samples")
    
    def predict(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        """Make sentiment predictions.
        
        Args:
            X: Text data to predict sentiment for.
            
        Returns:
            Array of predicted sentiment labels.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to appropriate format
        if isinstance(X, pd.Series):
            X_data = X.values
        elif isinstance(X, pd.DataFrame):
            # Assume single text column
            if len(X.columns) != 1:
                raise ValueError("For text input, DataFrame must have exactly one column")
            X_data = X.iloc[:, 0].values
        else:
            X_data = X
        
        # Handle transformer model separately
        if self.model_type == "transformer":
            return self._predict_transformer(X_data)
        
        # Use traditional ML model
        predictions = self.model.predict(X_data)
        
        return predictions
    
    def _predict_transformer(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with transformer model.
        
        Args:
            X: Text data to predict sentiment for.
            
        Returns:
            Array of predicted sentiment labels.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required for model_type='transformer'")
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Transformer model must be trained before prediction")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Make predictions in batches
        predictions = []
        
        for i in range(0, len(X), self.model_params["batch_size"]):
            batch_texts = X[i:i + self.model_params["batch_size"]]
            
            # Tokenize
            inputs = self.tokenizer(
                list(map(str, batch_texts)),
                padding=True,
                truncation=True,
                max_length=self.model_params["max_length"],
                return_tensors="pt"
            ).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get predicted class
                pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
                
                # Convert to original labels
                batch_predictions = [self.model.config.id2label[int(id_)] for id_ in pred_ids]
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def predict_proba(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        """Predict probability of each sentiment class.
        
        Args:
            X: Text data to predict probabilities for.
            
        Returns:
            Array of predicted probabilities for each class.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to appropriate format
        if isinstance(X, pd.Series):
            X_data = X.values
        elif isinstance(X, pd.DataFrame):
            # Assume single text column
            if len(X.columns) != 1:
                raise ValueError("For text input, DataFrame must have exactly one column")
            X_data = X.iloc[:, 0].values
        else:
            X_data = X
        
        # Handle transformer model separately
        if self.model_type == "transformer":
            return self._predict_proba_transformer(X_data)
        
        # Check if the model supports predict_proba
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_type} model does not support probability predictions")
        
        # Use traditional ML model
        probabilities = self.model.predict_proba(X_data)
        
        return probabilities
    
    def _predict_proba_transformer(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of each sentiment class with transformer model.
        
        Args:
            X: Text data to predict probabilities for.
            
        Returns:
            Array of predicted probabilities for each class.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required for model_type='transformer'")
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Transformer model must be trained before prediction")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Make predictions in batches
        all_probs = []
        
        for i in range(0, len(X), self.model_params["batch_size"]):
            batch_texts = X[i:i + self.model_params["batch_size"]]
            
            # Tokenize
            inputs = self.tokenizer(
                list(map(str, batch_texts)),
                padding=True,
                truncation=True,
                max_length=self.model_params["max_length"],
                return_tensors="pt"
            ).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        
        # Combine batch results
        return np.vstack(all_probs)
    
    def evaluate_classification(
        self, 
        X: Union[pd.Series, List[str], np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        plot: bool = False,
        figure_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the sentiment classification model.
        
        Args:
            X: Test text data.
            y: True sentiment labels.
            plot: Whether to generate evaluation plots.
            figure_path: Path to save the evaluation plot.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Convert to arrays
        y_true = np.array(y)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }
        
        logger.info(f"Model evaluation accuracy: {metrics['accuracy']:.4f}")
        
        # Generate evaluation plot if requested
        if plot:
            self._plot_classification_evaluation(metrics, figure_path)
        
        return metrics
    
    def _plot_classification_evaluation(
        self, 
        metrics: Dict[str, Any],
        figure_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Generate classification evaluation plots.
        
        Args:
            metrics: Evaluation metrics.
            figure_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot confusion matrix
        cm = metrics["confusion_matrix"]
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=self.classes_,
            yticklabels=self.classes_,
            ax=ax1
        )
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        ax1.set_title("Confusion Matrix")
        
        # Plot class metrics
        class_metrics = metrics["classification_report"]
        classes = [c for c in class_metrics.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        
        metrics_df = pd.DataFrame({
            'Class': classes,
            'Precision': [class_metrics[c]['precision'] for c in classes],
            'Recall': [class_metrics[c]['recall'] for c in classes],
            'F1-Score': [class_metrics[c]['f1-score'] for c in classes]
        })
        
        metrics_df = pd.melt(
            metrics_df, 
            id_vars=['Class'],
            value_vars=['Precision', 'Recall', 'F1-Score'],
            var_name='Metric',
            value_name='Score'
        )
        
        sns.barplot(
            data=metrics_df,
            x='Class',
            y='Score',
            hue='Metric',
            ax=ax2
        )
        
        ax2.set_ylim(0, 1.0)
        ax2.set_title(f"Classification Metrics (Accuracy: {metrics['accuracy']:.4f})")
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            figure_path = Path(figure_path)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classification evaluation plot saved to {figure_path}")
        
        return fig
    
    def extract_important_features(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """Extract most important features for each sentiment class.
        
        Args:
            top_n: Number of top features to extract for each class.
            
        Returns:
            Dictionary of DataFrames with important features by class.
        """
        if self.model is None:
            raise ValueError("Model must be trained before extracting features")
        
        # Only supported for traditional ML models
        if self.model_type == "transformer":
            raise ValueError("Feature extraction not supported for transformer models")
        
        # Extract vectorizer and classifier from pipeline
        vectorizer = self.model.named_steps.get('vectorizer')
        classifier = self.model.named_steps.get('classifier')
        
        if vectorizer is None or classifier is None:
            raise ValueError("Model pipeline does not have expected components")
        
        # Get feature names from vectorizer
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()
        
        # Extract coefficients from classifier
        if hasattr(classifier, 'coef_'):
            # For linear models like logistic regression
            coefficients = classifier.coef_
        elif hasattr(classifier, 'feature_importances_'):
            # For tree-based models like random forest
            coefficients = classifier.feature_importances_
            # Reshape for consistency with multi-class format
            if len(coefficients.shape) == 1:
                coefficients = coefficients.reshape(1, -1)
        else:
            raise ValueError("Classifier does not support feature importance extraction")
        
        # Extract important features for each class
        important_features = {}
        
        for i, class_label in enumerate(classifier.classes_):
            if len(coefficients.shape) > 1:
                # Multi-class case
                class_coefficients = coefficients[i]
            else:
                # Binary case
                class_coefficients = coefficients
            
            # Get indices of top coefficients
            top_indices = np.argsort(np.abs(class_coefficients))[-top_n:]
            
            # Create DataFrame with features and coefficients
            features_df = pd.DataFrame({
                'feature': feature_names[top_indices],
                'coefficient': class_coefficients[top_indices]
            })
            
            # Sort by absolute coefficient
            features_df = features_df.iloc[np.argsort(np.abs(features_df['coefficient']))[::-1]]
            
            # Add to results
            important_features[class_label] = features_df
        
        return important_features
    
    def plot_important_features(
        self,
        important_features: Dict[str, pd.DataFrame],
        figure_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot important features for sentiment classification.
        
        Args:
            important_features: Output from extract_important_features.
            figure_path: Path to save the figure.
            
        Returns:
            Matplotlib figure object.
        """
        if not important_features:
            raise ValueError("No important features provided")
        
        # Determine number of classes
        n_classes = len(important_features)
        
        # Create figure with subplots for each class
        fig, axes = plt.subplots(n_classes, 1, figsize=(10, 5*n_classes))
        
        # Handle single class case
        if n_classes == 1:
            axes = [axes]
        
        # Plot features for each class
        for i, (class_label, features_df) in enumerate(important_features.items()):
            # Sort by coefficient value for directional view
            features_df = features_df.sort_values('coefficient')
            
            # Create color map based on coefficient sign
            colors = ['red' if c < 0 else 'green' for c in features_df['coefficient']]
            
            # Plot horizontal bar chart
            axes[i].barh(
                features_df['feature'],
                features_df['coefficient'],
                color=colors
            )
            
            axes[i].set_title(f"Important Features for '{class_label}'")
            axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add labels
            for j, (_, row) in enumerate(features_df.iterrows()):
                text_color = 'black'
                axes[i].text(
                    row['coefficient'] * 0.95,
                    j,
                    f"{row['coefficient']:.4f}",
                    va='center',
                    ha='right' if row['coefficient'] > 0 else 'left',
                    color=text_color
                )
        
        plt.tight_layout()
        
        # Save figure if path provided
        if figure_path:
            figure_path = Path(figure_path)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            logger.info(f"Important features plot saved to {figure_path}")
        
        return fig