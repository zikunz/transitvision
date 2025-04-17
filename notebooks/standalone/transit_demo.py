"""
TransitVision Demo Script

This script demonstrates the key functionality of the TransitVision project
without requiring package installation.

Run this script directly to see the transit data analysis in action.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set up visualization
try:
    # For newer versions of seaborn
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        # For older versions of seaborn
        plt.style.use('seaborn-whitegrid')
    except:
        # Fallback to basic grid
        plt.grid(True)

plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)

def generate_transit_data(n_days=90, n_routes=5, n_stops_per_route=10, seed=42):
    """Generate synthetic transit data for demonstration."""
    np.random.seed(seed)
    
    # Generate dates
    start_date = pd.Timestamp('2023-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_days)]
    
    # Generate routes and stops
    routes = [f"Route_{i}" for i in range(1, n_routes + 1)]
    stops = [f"Stop_{i}" for i in range(1, n_stops_per_route + 1)]
    
    # Create base dataframe structure
    data = []
    
    for date in dates:
        # Weekend modifier for ridership
        is_weekend = date.dayofweek >= 5
        weekend_factor = 0.7 if is_weekend else 1.0
        
        # Monthly seasonality (higher in summer)
        month = date.month
        monthly_factor = 1.0 + 0.2 * np.sin((month - 1) * np.pi / 6)
        
        # Weather effect (random daily factor)
        weather_factor = np.random.uniform(0.8, 1.2)
        
        # Remote work percentage (gradually increasing over time)
        day_index = (date - start_date).days
        remote_work_pct = 20 + 10 * (day_index / n_days)
        
        for route in routes:
            # Route-specific factors
            route_idx = int(route.split('_')[1])
            route_factor = 0.8 + 0.1 * route_idx
            
            for stop in stops:
                # Stop-specific factors
                stop_idx = int(stop.split('_')[1])
                stop_factor = 0.9 + 0.02 * stop_idx
                
                # Calculate base ridership
                base_ridership = 100 * route_factor * stop_factor
                
                # Apply modifiers
                ridership = base_ridership * weekend_factor * monthly_factor * weather_factor
                
                # Apply remote work effect (more impact on commuter routes)
                remote_work_impact = 1.0 - (0.01 * remote_work_pct * route_factor)
                ridership = ridership * remote_work_impact
                
                # Add some random noise
                ridership = max(0, int(ridership * np.random.normal(1, 0.1)))
                
                # Generate capacity (somewhat correlated with ridership)
                capacity = int(max(ridership * 1.5, 150) * np.random.uniform(0.9, 1.1))
                
                # Generate delay (correlated with ridership/capacity ratio)
                utilization = ridership / capacity
                delay_base = 2 * utilization * np.random.exponential(1)
                delay = round(max(0, delay_base), 1)
                
                # Create data point
                data.append({
                    'service_date': date,
                    'route_id': route,
                    'stop_id': stop,
                    'ridership': ridership,
                    'capacity': capacity,
                    'delay': delay,
                    'temperature': round(20 + 10 * np.sin((date.dayofweek - 1) * np.pi / 7) + np.random.normal(0, 3), 1),
                    'precipitation': max(0, round(np.random.exponential(0.5), 2)),
                    'is_holiday': date.dayofweek >= 5 or np.random.random() < 0.05,
                    'remote_work_percent': round(remote_work_pct, 1),
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add time features
    df['service_month'] = df['service_date'].dt.month
    df['service_day'] = df['service_date'].dt.day
    df['service_dayofweek'] = df['service_date'].dt.dayofweek
    df['is_weekend'] = df['service_dayofweek'] >= 5
    
    return df

def generate_feedback_data(transit_data, n_feedback=500, seed=42):
    """Generate synthetic feedback data based on transit data."""
    np.random.seed(seed)
    
    # Sample from transit data to get realistic dates and routes
    sampled_data = transit_data.sample(n=n_feedback, random_state=seed)
    
    # Positive feedback templates
    positive_templates = [
        "The {route} was on time and clean. Very satisfied with the service.",
        "Driver was friendly and helpful. {route} was punctual as always.",
        "Great experience on {route} today. Comfortable ride and efficient service.",
        "Love the new schedule for {route}, makes my commute much easier.",
        "The bus was clean and not crowded. Very pleasant ride on {route}.",
        "Excellent service on {route} this morning. Right on schedule!",
        "The {route} driver was very professional and courteous.",
        "I appreciate the reliability of {route}. Always a good experience."
    ]
    
    # Negative feedback templates
    negative_templates = [
        "The {route} was late again. Very frustrating for daily commuters.",
        "Bus was overcrowded and uncomfortable. {route} needs more frequent service.",
        "Driver was rude and unhelpful. Poor experience on {route} today.",
        "The {route} was dirty and had a bad smell. Please improve cleaning.",
        "Disappointed with {route} service. Too many delays and no communication.",
        "The air conditioning wasn't working on {route}. Terrible experience in this heat.",
        "Why is {route} always late? Need better schedule adherence.",
        "The {route} bus broke down mid-journey. Needs better maintenance."
    ]
    
    # Neutral feedback templates
    neutral_templates = [
        "Average experience on {route}. Nothing special to report.",
        "The {route} was slightly delayed but overall okay.",
        "Regular service on {route} today. No issues to mention.",
        "Standard experience on {route}. Could use some minor improvements.",
        "The {route} was adequate for my needs today.",
        "Typical journey on {route}. Neither good nor bad.",
        "The {route} was moderately crowded but manageable.",
        "Satisfactory service on {route}, though there's room for improvement."
    ]
    
    # Generate feedback
    data = []
    
    for _, row in sampled_data.iterrows():
        # Determine sentiment based on delay and ridership/capacity ratio
        delay = row['delay']
        utilization = row['ridership'] / row['capacity']
        
        # Calculate base sentiment score (-1 to 1)
        sentiment_score = 0.5 - (delay / 15) - (utilization - 0.5)
        sentiment_score += np.random.normal(0, 0.3)  # Add noise
        
        # Determine sentiment category
        if sentiment_score > 0.3:
            sentiment = "positive"
            rating = np.random.choice([4, 5], p=[0.3, 0.7])
            template = np.random.choice(positive_templates)
        elif sentiment_score < -0.3:
            sentiment = "negative"
            rating = np.random.choice([1, 2], p=[0.7, 0.3])
            template = np.random.choice(negative_templates)
        else:
            sentiment = "neutral"
            rating = np.random.choice([3, 4], p=[0.7, 0.3])
            template = np.random.choice(neutral_templates)
        
        # Format feedback text
        feedback_text = template.format(route=row['route_id'])
        
        # Add some typos or variations (10% chance per feedback)
        if np.random.random() < 0.1:
            words = feedback_text.split()
            if len(words) > 3:
                word_idx = np.random.randint(0, len(words))
                if len(words[word_idx]) > 3:
                    char_idx = np.random.randint(1, len(words[word_idx]) - 1)
                    word_list = list(words[word_idx])
                    word_list[char_idx] = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
                    words[word_idx] = ''.join(word_list)
                    feedback_text = ' '.join(words)
        
        # Create feedback entry
        data.append({
            'feedback_text': feedback_text,
            'feedback_date': row['service_date'],
            'route_id': row['route_id'],
            'stop_id': row['stop_id'],
            'rating': rating,
            'sentiment': sentiment
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

def analyze_ridership_patterns(data, time_grouping="daily"):
    """Analyze ridership patterns over time."""
    df = data.copy()
    
    # Determine groupby columns based on time_grouping
    if time_grouping == "daily":
        time_grouper = ['service_date']
    elif time_grouping == "weekly":
        time_grouper = ['service_year', 'service_week']
    elif time_grouping == "monthly":
        time_grouper = ['service_year', 'service_month']
    else:
        raise ValueError(f"Unsupported time grouping: {time_grouping}")
    
    # Group by time and route to calculate ridership metrics
    group_cols = time_grouper + ['route_id']
    result = df.groupby(group_cols).agg({
        'ridership': ['sum', 'mean', 'median', 'count'],
        'capacity': ['sum', 'mean']
    }).reset_index()
    
    # Flatten MultiIndex columns
    result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
    
    # Calculate utilization
    result['utilization_rate'] = result['ridership_sum'] / result['capacity_sum']
    
    return result

def plot_ridership_trends(data, time_grouping="daily"):
    """Plot ridership trends over time."""
    df = data.copy()
    
    if time_grouping == "daily":
        # Group by date and calculate total ridership
        ridership_by_date = df.groupby('service_date')['ridership'].sum().reset_index()
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(ridership_by_date['service_date'], ridership_by_date['ridership'], marker='o')
        plt.title('Daily Ridership Trends')
        plt.xlabel('Date')
        plt.ylabel('Total Ridership')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Plot by route
        plt.figure(figsize=(12, 6))
        for route in df['route_id'].unique():
            route_data = df[df['route_id'] == route]
            ridership_by_date = route_data.groupby('service_date')['ridership'].sum().reset_index()
            plt.plot(ridership_by_date['service_date'], ridership_by_date['ridership'], 
                     marker='o', label=route)
        
        plt.title('Daily Ridership by Route')
        plt.xlabel('Date')
        plt.ylabel('Total Ridership')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Route')
        plt.tight_layout()
        plt.show()

def plot_performance_comparison(data, metric="ridership", plot_type="boxplot"):
    """Plot transit performance comparison."""
    df = data.copy()
    
    plt.figure(figsize=(12, 6))
    
    if plot_type == "boxplot":
        sns.boxplot(x='route_id', y=metric, data=df)
    elif plot_type == "violin":
        sns.violinplot(x='route_id', y=metric, data=df)
    elif plot_type == "bar":
        route_metric = df.groupby('route_id')[metric].mean().reset_index()
        sns.barplot(x='route_id', y=metric, data=route_metric)
    
    plt.title(f'{metric.capitalize()} Comparison by Route')
    plt.xlabel('Route')
    plt.ylabel(metric.capitalize())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def analyze_sentiment(feedback_data):
    """Basic sentiment analysis based on ratings."""
    df = feedback_data.copy()
    
    # Map ratings to sentiment categories
    df['sentiment_category'] = df['rating'].apply(
        lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
    )
    
    # Count sentiment by route
    sentiment_by_route = df.groupby(['route_id', 'sentiment_category']).size().unstack(fill_value=0)
    
    # Calculate percentages
    sentiment_pct = sentiment_by_route.div(sentiment_by_route.sum(axis=1), axis=0) * 100
    
    return sentiment_pct

def plot_sentiment_distribution(sentiment_data):
    """Plot sentiment distribution."""
    # Plot sentiment distribution
    plt.figure(figsize=(12, 6))
    sentiment_data.plot(kind='bar', stacked=True, 
                        color=['red', 'gray', 'green'])
    plt.title('Sentiment Distribution by Route')
    plt.xlabel('Route')
    plt.ylabel('Percentage')
    plt.legend(title='Sentiment')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def build_prediction_model(X_train, y_train):
    """Build a simple random forest model for ridership prediction."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Create pipeline with scaling and random forest
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the prediction model."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Ridership')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    """Run the TransitVision demo."""
    print("TransitVision Demo")
    print("=================")
    
    # Generate sample data
    print("\nGenerating sample transit data...")
    transit_data = generate_transit_data()
    print(f"Generated transit data: {transit_data.shape[0]} records")
    
    print("\nGenerating sample feedback data...")
    feedback_data = generate_feedback_data(transit_data)
    print(f"Generated feedback data: {feedback_data.shape[0]} records")
    
    # Display sample data
    print("\nSample transit data:")
    print(transit_data.head())
    
    print("\nSample feedback data:")
    print(feedback_data.head())
    
    # Analyze ridership patterns
    print("\nAnalyzing ridership patterns...")
    ridership_patterns = analyze_ridership_patterns(transit_data)
    print("Ridership pattern analysis complete")
    
    # Plot ridership trends
    print("\nPlotting ridership trends...")
    plot_ridership_trends(transit_data)
    
    # Plot performance comparison
    print("\nPlotting performance comparisons...")
    plot_performance_comparison(transit_data, metric="ridership", plot_type="boxplot")
    plot_performance_comparison(transit_data, metric="delay", plot_type="violin")
    
    # Analyze sentiment
    print("\nAnalyzing sentiment in feedback data...")
    sentiment_results = analyze_sentiment(feedback_data)
    print("\nSentiment distribution by route:")
    print(sentiment_results)
    
    # Plot sentiment distribution
    print("\nPlotting sentiment distribution...")
    plot_sentiment_distribution(sentiment_results)
    
    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    from sklearn.model_selection import train_test_split
    
    # Select features and target
    features = [
        'service_month', 'service_day', 'service_dayofweek', 'is_weekend',
        'temperature', 'precipitation', 'is_holiday', 'remote_work_percent'
    ]
    
    # Add route and stop dummy variables
    route_dummies = pd.get_dummies(transit_data['route_id'], prefix='route')
    stop_dummies = pd.get_dummies(transit_data['stop_id'], prefix='stop')
    
    # Combine features
    X = pd.concat([transit_data[features], route_dummies, stop_dummies], axis=1)
    y = transit_data['ridership']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build and evaluate model
    print("\nBuilding prediction model...")
    model = build_prediction_model(X_train, y_train)
    
    print("\nEvaluating model performance...")
    evaluate_model(model, X_test, y_test)
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()