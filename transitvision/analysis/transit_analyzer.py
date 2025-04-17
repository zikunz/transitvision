"""Transit data analysis module."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TransitAnalyzer:
    """Analyze transit data to extract insights.
    
    This class provides methods for analyzing transit data, including ridership
    patterns, service performance, and other transit-related metrics.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the transit analyzer.
        
        Args:
            config: Configuration parameters for the analyzer.
        """
        # Default configuration
        default_config = {
            "date_column": "service_date",
            "route_column": "route_id",
            "stop_column": "stop_id",
            "ridership_column": "ridership",
            "capacity_column": "capacity",
            "delay_column": "delay",
            "time_columns": ["departure_time", "arrival_time"],
            "plot_style": "whitegrid",
            "plot_palette": "viridis",
            "plot_figsize": (12, 8),
        }
        
        # Update default config with user-provided config
        self.config = default_config
        if config:
            for key, value in config.items():
                self.config[key] = value
        
        # Set plot style
        sns.set_style(self.config["plot_style"])
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load processed transit data for analysis.
        
        Args:
            file_path: Path to the processed transit data file.
            
        Returns:
            DataFrame containing the transit data.
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
            
            logger.info(f"Loaded {len(data)} records from {file_path}")
            
            # Convert date column to datetime if needed
            date_col = self.config["date_column"]
            if date_col in data.columns and not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                data[date_col] = pd.to_datetime(data[date_col])
                logger.info(f"Converted column {date_col} to datetime")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def analyze_ridership_patterns(
        self, 
        data: pd.DataFrame,
        time_grouping: str = "daily",
        route_filter: Optional[Union[str, List[str]]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """Analyze ridership patterns over time.
        
        Args:
            data: Transit data DataFrame.
            time_grouping: Time grouping level ("hourly", "daily", "weekly", "monthly").
            route_filter: Optional route ID(s) to filter by.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.
            
        Returns:
            DataFrame with ridership analysis results.
        """
        df = data.copy()
        
        # Apply filters
        if route_filter:
            if isinstance(route_filter, list):
                df = df[df[self.config["route_column"]].isin(route_filter)]
            else:
                df = df[df[self.config["route_column"]] == route_filter]
        
        date_col = self.config["date_column"]
        if date_col in df.columns:
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df[date_col] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df[date_col] <= end_date]
        
        # Group by time
        ridership_col = self.config["ridership_column"]
        if ridership_col not in df.columns:
            raise ValueError(f"Ridership column '{ridership_col}' not found in data")
        
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in data")
        
        # Create time grouper
        if time_grouping == "hourly":
            if not any("hour" in col for col in df.columns):
                df["hour"] = df[date_col].dt.hour
            time_grouper = ["hour"]
        elif time_grouping == "daily":
            time_grouper = [date_col]
        elif time_grouping == "weekly":
            df["week"] = df[date_col].dt.isocalendar().week
            df["year"] = df[date_col].dt.isocalendar().year
            time_grouper = ["year", "week"]
        elif time_grouping == "monthly":
            df["month"] = df[date_col].dt.month
            df["year"] = df[date_col].dt.year
            time_grouper = ["year", "month"]
        else:
            raise ValueError(f"Invalid time grouping: {time_grouping}")
        
        # Group by route if filter is not applied
        groupby_cols = time_grouper.copy()
        if not route_filter:
            groupby_cols.append(self.config["route_column"])
        
        # Group and aggregate
        result = df.groupby(groupby_cols).agg({
            ridership_col: ["sum", "mean", "median", "std", "count"]
        }).reset_index()
        
        # Flatten column names
        result.columns = [
            "_".join(col).strip("_") for col in result.columns.values
        ]
        
        # Calculate capacity utilization if capacity column exists
        capacity_col = self.config["capacity_column"]
        if capacity_col in df.columns:
            capacity_data = df.groupby(groupby_cols)[capacity_col].sum().reset_index()
            capacity_data.columns = groupby_cols + ["capacity_sum"]
            
            # Merge with result
            result = result.merge(capacity_data, on=groupby_cols)
            
            # Calculate utilization rate
            result["utilization_rate"] = result[f"{ridership_col}_sum"] / result["capacity_sum"]
        
        logger.info(f"Analyzed ridership patterns with {time_grouping} grouping")
        
        return result
    
    def analyze_performance_metrics(
        self, 
        data: pd.DataFrame,
        metric: str = "delay",
        route_filter: Optional[Union[str, List[str]]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """Analyze transit service performance metrics.
        
        Args:
            data: Transit data DataFrame.
            metric: Performance metric to analyze ("delay", "speed", etc.).
            route_filter: Optional route ID(s) to filter by.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.
            
        Returns:
            DataFrame with performance analysis results.
        """
        df = data.copy()
        
        # Apply filters
        if route_filter:
            if isinstance(route_filter, list):
                df = df[df[self.config["route_column"]].isin(route_filter)]
            else:
                df = df[df[self.config["route_column"]] == route_filter]
        
        date_col = self.config["date_column"]
        if date_col in df.columns:
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df[date_col] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df[date_col] <= end_date]
        
        # Default to delay analysis
        if metric == "delay":
            delay_col = self.config["delay_column"]
            if delay_col not in df.columns:
                raise ValueError(f"Delay column '{delay_col}' not found in data")
            
            # Group by route and date
            route_col = self.config["route_column"]
            groupby_cols = [route_col, date_col] if date_col in df.columns else [route_col]
            
            # Calculate metrics
            result = df.groupby(groupby_cols).agg({
                delay_col: ["mean", "median", "std", "min", "max", "count"]
            }).reset_index()
            
            # Flatten column names
            result.columns = [
                "_".join(col).strip("_") for col in result.columns.values
            ]
            
            # Calculate on-time performance (percentage of trips with delay <= 5 minutes)
            on_time_threshold = 5  # 5 minutes
            on_time_data = df.groupby(groupby_cols).apply(
                lambda x: (x[delay_col] <= on_time_threshold).mean()
            ).reset_index(name="on_time_percentage")
            
            # Merge with result
            result = result.merge(on_time_data, on=groupby_cols)
            
            logger.info(f"Analyzed delay performance metrics")
            
        else:
            raise ValueError(f"Unsupported performance metric: {metric}")
        
        return result
    
    def compare_routes(
        self, 
        data: pd.DataFrame,
        metric: str = "ridership",
        time_period: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compare different transit routes based on a given metric.
        
        Args:
            data: Transit data DataFrame.
            metric: Metric to compare routes by.
            time_period: Optional time period for comparison ("daily", "weekly", "monthly").
            top_n: Optional number of top routes to include.
            
        Returns:
            DataFrame with route comparison results.
        """
        df = data.copy()
        
        # Determine metric column
        if metric == "ridership":
            metric_col = self.config["ridership_column"]
        elif metric == "delay":
            metric_col = self.config["delay_column"]
        else:
            raise ValueError(f"Unsupported comparison metric: {metric}")
        
        if metric_col not in df.columns:
            raise ValueError(f"Metric column '{metric_col}' not found in data")
        
        # Define grouping
        route_col = self.config["route_column"]
        groupby_cols = [route_col]
        
        # Add time grouping if specified
        date_col = self.config["date_column"]
        if time_period and date_col in df.columns:
            if time_period == "daily":
                groupby_cols.append(date_col)
            elif time_period == "weekly":
                df["week"] = df[date_col].dt.isocalendar().week
                df["year"] = df[date_col].dt.isocalendar().year
                groupby_cols.extend(["year", "week"])
            elif time_period == "monthly":
                df["month"] = df[date_col].dt.month
                df["year"] = df[date_col].dt.year
                groupby_cols.extend(["year", "month"])
            else:
                raise ValueError(f"Invalid time period: {time_period}")
        
        # Group and aggregate
        result = df.groupby(groupby_cols).agg({
            metric_col: ["sum", "mean", "median", "std", "count"]
        }).reset_index()
        
        # Flatten column names
        result.columns = [
            "_".join(col).strip("_") for col in result.columns.values
        ]
        
        # Get route-level summary if time grouping is applied
        if len(groupby_cols) > 1:
            summary = result.groupby(route_col)[f"{metric_col}_sum"].sum().reset_index()
            summary.columns = [route_col, f"total_{metric_col}"]
            
            # Merge summary back to result
            result = result.merge(summary, on=route_col)
        else:
            # Rename for consistency
            result = result.rename(columns={f"{metric_col}_sum": f"total_{metric_col}"})
        
        # Filter to top N routes if specified
        if top_n:
            top_routes = summary.sort_values(f"total_{metric_col}", ascending=False).head(top_n)[route_col].tolist()
            result = result[result[route_col].isin(top_routes)]
        
        logger.info(f"Compared routes by {metric}")
        
        return result
    
    def analyze_stop_performance(
        self, 
        data: pd.DataFrame,
        metric: str = "ridership",
        route_filter: Optional[Union[str, List[str]]] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Analyze performance of transit stops.
        
        Args:
            data: Transit data DataFrame.
            metric: Metric to analyze stops by.
            route_filter: Optional route ID(s) to filter by.
            top_n: Optional number of top stops to include.
            
        Returns:
            DataFrame with stop performance analysis.
        """
        df = data.copy()
        
        stop_col = self.config["stop_column"]
        if stop_col not in df.columns:
            raise ValueError(f"Stop column '{stop_col}' not found in data")
        
        # Apply route filter
        if route_filter:
            if isinstance(route_filter, list):
                df = df[df[self.config["route_column"]].isin(route_filter)]
            else:
                df = df[df[self.config["route_column"]] == route_filter]
        
        # Determine metric column
        if metric == "ridership":
            metric_col = self.config["ridership_column"]
        elif metric == "delay":
            metric_col = self.config["delay_column"]
        else:
            raise ValueError(f"Unsupported stop performance metric: {metric}")
        
        if metric_col not in df.columns:
            raise ValueError(f"Metric column '{metric_col}' not found in data")
        
        # Group by stop and optionally route
        groupby_cols = [stop_col]
        if not route_filter:
            groupby_cols.append(self.config["route_column"])
        
        # Group and aggregate
        result = df.groupby(groupby_cols).agg({
            metric_col: ["sum", "mean", "median", "std", "count"]
        }).reset_index()
        
        # Flatten column names
        result.columns = [
            "_".join(col).strip("_") for col in result.columns.values
        ]
        
        # Filter to top N stops if specified
        if top_n:
            result = result.sort_values(f"{metric_col}_sum", ascending=False).head(top_n)
        
        logger.info(f"Analyzed stop performance by {metric}")
        
        return result
    
    def plot_ridership_trends(
        self,
        data: pd.DataFrame,
        time_grouping: str = "daily",
        route_filter: Optional[Union[str, List[str]]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot ridership trends over time.
        
        Args:
            data: Transit data DataFrame.
            time_grouping: Time grouping level.
            route_filter: Optional route ID(s) to filter by.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        # Get ridership patterns data
        ridership_data = self.analyze_ridership_patterns(
            data=data,
            time_grouping=time_grouping,
            route_filter=route_filter
        )
        
        # Create figure
        figsize = self.config["plot_figsize"]
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine x-axis based on time grouping
        if time_grouping == "hourly":
            x_col = "hour"
            title = "Hourly Ridership Patterns"
            x_label = "Hour of Day"
        elif time_grouping == "daily":
            x_col = self.config["date_column"]
            title = "Daily Ridership Trends"
            x_label = "Date"
        elif time_grouping == "weekly":
            # Create week label
            ridership_data["week_label"] = ridership_data.apply(
                lambda row: f"{int(row['year'])}-W{int(row['week']):02d}", axis=1
            )
            x_col = "week_label"
            title = "Weekly Ridership Trends"
            x_label = "Week"
        elif time_grouping == "monthly":
            # Create month label
            ridership_data["month_label"] = ridership_data.apply(
                lambda row: f"{int(row['year'])}-{int(row['month']):02d}", axis=1
            )
            x_col = "month_label"
            title = "Monthly Ridership Trends"
            x_label = "Month"
        
        # Plot based on presence of route filter
        y_col = f"{self.config['ridership_column']}_sum"
        
        if route_filter:
            # Simple line plot for a single route or aggregated routes
            sns.lineplot(
                data=ridership_data,
                x=x_col,
                y=y_col,
                marker='o',
                ax=ax
            )
            
            route_label = route_filter if isinstance(route_filter, str) else "Selected Routes"
            title = f"{title} - {route_label}"
            
        else:
            # Multi-line plot for route comparison
            route_col = self.config["route_column"]
            
            # Limit to top 10 routes by total ridership for readability
            route_totals = ridership_data.groupby(route_col)[y_col].sum().reset_index()
            top_routes = route_totals.sort_values(y_col, ascending=False).head(10)[route_col].tolist()
            
            plot_data = ridership_data[ridership_data[route_col].isin(top_routes)]
            
            sns.lineplot(
                data=plot_data,
                x=x_col,
                y=y_col,
                hue=route_col,
                marker='o',
                ax=ax
            )
            
            title = f"{title} - Top Routes"
            plt.legend(title="Route", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel("Total Ridership")
        ax.set_title(title)
        
        # Rotate x-axis labels if needed
        if time_grouping in ["weekly", "monthly"]:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_performance_comparison(
        self,
        data: pd.DataFrame,
        metric: str = "delay",
        route_filter: Optional[Union[str, List[str]]] = None,
        plot_type: str = "boxplot",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot transit performance comparison.
        
        Args:
            data: Transit data DataFrame.
            metric: Performance metric to plot.
            route_filter: Optional route ID(s) to filter by.
            plot_type: Type of plot ("boxplot", "violin", "bar").
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        df = data.copy()
        
        # Apply route filter
        if route_filter:
            if isinstance(route_filter, list):
                df = df[df[self.config["route_column"]].isin(route_filter)]
            else:
                df = df[df[self.config["route_column"]] == route_filter]
        
        # Determine metric column
        if metric == "delay":
            metric_col = self.config["delay_column"]
            y_label = "Delay (minutes)"
            title = "Transit Delay Comparison"
        elif metric == "ridership":
            metric_col = self.config["ridership_column"]
            y_label = "Ridership"
            title = "Transit Ridership Comparison"
        elif metric == "utilization":
            if (self.config["ridership_column"] in df.columns and 
                self.config["capacity_column"] in df.columns):
                df["utilization"] = df[self.config["ridership_column"]] / df[self.config["capacity_column"]]
                metric_col = "utilization"
                y_label = "Utilization Rate"
                title = "Transit Utilization Rate Comparison"
            else:
                raise ValueError("Ridership and capacity columns required for utilization metric")
        else:
            raise ValueError(f"Unsupported performance metric: {metric}")
        
        if metric_col not in df.columns:
            raise ValueError(f"Metric column '{metric_col}' not found in data")
        
        # Create figure
        figsize = self.config["plot_figsize"]
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine grouping - use route and stop if both available
        route_col = self.config["route_column"]
        stop_col = self.config["stop_column"]
        
        if stop_col in df.columns and not route_filter:
            # If data has too many stops, summarize by route instead
            if df[stop_col].nunique() > 15:
                groupby_col = route_col
            else:
                groupby_col = stop_col
        else:
            groupby_col = route_col
        
        # Limit to top entries for readability
        if df[groupby_col].nunique() > 15:
            # Get top values by median of metric
            top_values = df.groupby(groupby_col)[metric_col].median().nlargest(15).index.tolist()
            df = df[df[groupby_col].isin(top_values)]
        
        # Create plot based on specified type
        if plot_type == "boxplot":
            sns.boxplot(
                data=df,
                x=groupby_col,
                y=metric_col,
                ax=ax,
                palette=self.config["plot_palette"]
            )
        elif plot_type == "violin":
            sns.violinplot(
                data=df,
                x=groupby_col,
                y=metric_col,
                ax=ax,
                palette=self.config["plot_palette"]
            )
        elif plot_type == "bar":
            # Use mean as summary statistic for bar plot
            plot_data = df.groupby(groupby_col)[metric_col].mean().reset_index()
            sns.barplot(
                data=plot_data,
                x=groupby_col,
                y=metric_col,
                ax=ax,
                palette=self.config["plot_palette"]
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        # Set labels and title
        ax.set_xlabel(groupby_col.replace("_", " ").title())
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig