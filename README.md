# TransitVision

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

TransitVision is a basic Python package for transportation data analysis and prediction. It provides tools for processing, analyzing, and visualizing transit data, as well as building predictive models for transit ridership patterns.

## Features

### Data Processing

- **Transit Data Processing**: Clean, normalize, and prepare transit schedule, ridership, and performance data
- **Feedback Processing**: Process and analyze rider feedback and survey responses
- **Geospatial Processing**: Handle location-based transit data and satellite imagery

### Analysis

- **Transit Analysis**: Extract insights from transit data, including ridership patterns, service performance, and route comparison
- **Sentiment Analysis**: Analyze sentiment in rider feedback and extract key topics
- **Geospatial Analysis**: Analyze spatial patterns in transit data, including stop density and accessibility

### Prediction

- **Ridership Prediction**: Build machine learning models to predict transit ridership
- **Remote Work Impact**: Analyze and forecast the impact of remote work patterns on transit usage
- **Sentiment Prediction**: Predict sentiment in rider feedback

### Visualization

- Rich visualization tools for transit data, ridership patterns, sentiment analysis, and model evaluation

## Running the Project

There are multiple ways to run TransitVision, depending on your needs:

### Option 1: Standalone Demo (Easiest, No Installation)

Run the standalone Python script that demonstrates all functionality:
```bash
# Navigate to the project directory
cd TransitVision

# Run the standalone demo script
python notebooks/standalone/transit_demo.py
```

### Option 2: Standalone Jupyter Notebook (No Installation)

Open and run the standalone Jupyter notebook that works without installation:
```bash
# Navigate to the project directory
cd TransitVision

# Start Jupyter and open the notebook
jupyter notebook notebooks/standalone_notebook/TransitVision_Demo.ipynb
```

### Option 3: Full Package (Requires Installation)

```bash
# Navigate to the project directory
cd TransitVision

# Install the package in development mode
pip install -e .

# Open the full-featured notebook
jupyter notebook notebooks/transit_analysis_demo.ipynb
```

**Note:** When using Option 3, you need to add this code to the beginning of the notebook:
```python
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
```

## Installation

### Using pip

```bash
pip install transitvision
```

### From source

```bash
git clone https://github.com/zikunz/TransitVision.git
cd TransitVision
pip install -e .
```

### Dependencies

TransitVision requires the following core dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly

Additional dependencies for specific functionality:
- TensorFlow or PyTorch (for deep learning models)
- Transformers (for NLP models)
- geopandas, rasterio (for geospatial analysis)

These are handled in the setup.py file with extras_require sections, so you can install just what you need:

```bash
# Install core package
pip install -e .

# Install with ML support
pip install -e ".[ml]"

# Install with geospatial support
pip install -e ".[geo]" 

# Install everything including development tools
pip install -e ".[ml,geo,dev]"
```

## Module Structure

```
transitvision/
├── data_processing/       # Data processing modules
│   ├── base_processor.py  # Base class for data processors
│   ├── transit_data_processor.py 
│   ├── feedback_processor.py
│   └── geospatial_processor.py
├── analysis/              # Analysis modules
│   ├── transit_analyzer.py
│   ├── sentiment_analyzer.py
│   └── geospatial_analyzer.py
├── prediction/            # Prediction modules
│   ├── base_model.py      # Base class for prediction models
│   ├── ridership_model.py
│   ├── remote_work_impact.py
│   └── sentiment_predictor.py
└── utils/                 # Utility modules
    ├── config.py
    ├── data_utils.py
    ├── logger.py
    └── visualization.py
```

## Features in Detail

### 1. Transit Data Processing

- **Data Cleaning**: Handle missing values, outliers, and inconsistencies in transit data
- **Feature Engineering**: Create relevant features from date/time, route information, and other transit attributes
- **Data Normalization**: Scale and normalize data for analysis and modeling

```python
from transitvision.data_processing import TransitDataProcessor

processor = TransitDataProcessor(
    config={
        "time_columns": ["departure_time", "arrival_time"],
        "categorical_columns": ["route_id", "service_id", "trip_id"],
        "numerical_columns": ["ridership", "capacity", "delay"],
        "date_columns": ["service_date"],
        "drop_na_columns": ["route_id", "stop_id"]
    }
)

processed_data = processor.process_data(raw_data)
```

### 2. Sentiment Analysis in Transit Feedback

- **Sentiment Classification**: Analyze rider feedback to determine sentiment (positive, negative, neutral)
- **Topic Extraction**: Identify key topics and issues mentioned in feedback
- **Temporal Analysis**: Track sentiment changes over time

```python
from transitvision.analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment_results = analyzer.analyze_sentiment(feedback_data)
topic_results, topics = analyzer.extract_topics(sentiment_results)

# Visualize sentiment distribution
analyzer.plot_sentiment_distribution(sentiment_results, groupby="route_id")
```

### 3. Remote Work Impact Analysis

- **Sensitivity Analysis**: Analyze how changes in remote work patterns affect transit ridership
- **Scenario Analysis**: Compare different remote work scenarios and their impact on transit
- **Forecasting**: Predict future ridership based on evolving remote work trends

```python
from transitvision.prediction import RemoteWorkImpactModel

model = RemoteWorkImpactModel(
    model_type="elastic_net",
    remote_work_column="remote_work_percent"
)

model.fit(X_train, y_train)

# Sensitivity analysis
sensitivity_results = model.sensitivity_analysis(
    X=baseline_data,
    remote_work_values=[0, 20, 40, 60, 80, 100]
)

# Visualize impact
model.plot_remote_work_impact(sensitivity_results)
```

## Use Cases

- **Transit Agencies**: Analyze ridership patterns, optimize routes, and forecast demand
- **Urban Planners**: Study transportation patterns and accessibility
- **Researchers**: Analyze the impact of remote work on transportation systems
- **Policy Makers**: Make data-driven decisions about transportation infrastructure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Inspired by real-world transportation data analysis challenges
- Thanks to the pandas, scikit-learn, and matplotlib communities for their amazing tools