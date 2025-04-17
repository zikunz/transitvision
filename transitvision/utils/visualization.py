"""Visualization utilities for the TransitVision package."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def set_plot_style(
    style: str = "whitegrid",
    context: str = "notebook",
    palette: str = "viridis",
    font_scale: float = 1.0
) -> None:
    """Set the default plot style.
    
    Args:
        style: Seaborn style name.
        context: Seaborn context name.
        palette: Color palette name.
        font_scale: Scale factor for font sizes.
    """
    try:
        sns.set_theme(style=style, context=context, palette=palette, font_scale=font_scale)
        logger.debug(f"Set plot style: {style}, context: {context}, palette: {palette}")
    except Exception as e:
        logger.error(f"Error setting plot style: {str(e)}")


def plot_time_series(
    data: pd.DataFrame,
    x: str,
    y: str,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> plt.Figure:
    """Plot a time series.
    
    Args:
        data: DataFrame with data to plot.
        x: Name of the column for x-axis (usually time).
        y: Name of the column for y-axis.
        group_by: Optional column to group by.
        title: Optional plot title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure.
        **kwargs: Additional arguments for plotting.
        
    Returns:
        Matplotlib figure object.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot based on grouping
    if group_by and group_by in data.columns:
        # Get unique groups
        groups = data[group_by].unique()
        
        # Limit to top 10 groups if there are too many
        if len(groups) > 10:
            # Determine top groups by the sum of y values
            top_groups = data.groupby(group_by)[y].sum().nlargest(10).index
            plot_data = data[data[group_by].isin(top_groups)]
            logger.info(f"Limiting plot to top 10 groups out of {len(groups)}")
        else:
            plot_data = data
        
        # Plot with grouping
        sns.lineplot(
            data=plot_data,
            x=x,
            y=y,
            hue=group_by,
            marker='o',
            ax=ax,
            **kwargs
        )
        
        # Add legend with reasonable placement
        plt.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    else:
        # Simple time series plot
        sns.lineplot(
            data=data,
            x=x,
            y=y,
            marker='o',
            ax=ax,
            **kwargs
        )
    
    # Set labels and title
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{y} over {x}")
    
    # Rotate x-axis labels if there are many
    if len(data[x].unique()) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        # Create directory if it doesn't exist
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> plt.Figure:
    """Plot a bar chart.
    
    Args:
        data: DataFrame with data to plot.
        x: Name of the column for x-axis categories.
        y: Name of the column for y-axis values.
        group_by: Optional column to group by (for stacked or grouped bars).
        title: Optional plot title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure.
        **kwargs: Additional arguments for plotting.
        
    Returns:
        Matplotlib figure object.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot based on grouping
    if group_by and group_by in data.columns:
        # Get unique groups and x values
        groups = data[group_by].unique()
        x_values = data[x].unique()
        
        # Limit to top 10 categories if there are too many
        if len(x_values) > 10:
            # Determine top categories by the sum of y values
            top_x = data.groupby(x)[y].sum().nlargest(10).index
            plot_data = data[data[x].isin(top_x)]
            logger.info(f"Limiting plot to top 10 categories out of {len(x_values)}")
        else:
            plot_data = data
        
        # Plot with grouping
        sns.barplot(
            data=plot_data,
            x=x,
            y=y,
            hue=group_by,
            ax=ax,
            **kwargs
        )
        
        # Add legend with reasonable placement
        plt.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    else:
        # Limit to top 15 categories if there are too many
        if len(data[x].unique()) > 15:
            # Determine top categories by the sum of y values
            top_x = data.groupby(x)[y].sum().nlargest(15).index
            plot_data = data[data[x].isin(top_x)]
            logger.info(f"Limiting plot to top 15 categories out of {len(data[x].unique())}")
        else:
            plot_data = data
            
        # Simple bar chart
        sns.barplot(
            data=plot_data,
            x=x,
            y=y,
            ax=ax,
            **kwargs
        )
    
    # Set labels and title
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{y} by {x}")
    
    # Rotate x-axis labels if there are many
    if len(data[x].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        # Create directory if it doesn't exist
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_heatmap(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> plt.Figure:
    """Plot a heatmap.
    
    Args:
        data: DataFrame with data to plot.
        x: Name of the column for x-axis categories.
        y: Name of the column for y-axis categories.
        value: Name of the column for cell values.
        title: Optional plot title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        figsize: Figure size as (width, height).
        cmap: Colormap name.
        save_path: Optional path to save the figure.
        **kwargs: Additional arguments for plotting.
        
    Returns:
        Matplotlib figure object.
    """
    # Create pivot table
    pivot_data = data.pivot_table(index=y, columns=x, values=value, aggfunc='mean')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        pivot_data,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        ax=ax,
        **kwargs
    )
    
    # Set labels and title
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{value} by {x} and {y}")
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        # Create directory if it doesn't exist
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_distribution(
    data: pd.DataFrame,
    column: str,
    group_by: Optional[str] = None,
    kind: str = "hist",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> plt.Figure:
    """Plot a data distribution.
    
    Args:
        data: DataFrame with data to plot.
        column: Name of the column to plot distribution of.
        group_by: Optional column to group by.
        kind: Type of distribution plot ('hist', 'kde', 'ecdf', 'box', 'violin').
        title: Optional plot title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        figsize: Figure size as (width, height).
        save_path: Optional path to save the figure.
        **kwargs: Additional arguments for plotting.
        
    Returns:
        Matplotlib figure object.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set common plot elements
    ax.set_xlabel(xlabel or column)
    ylabel_text = ylabel or ("Density" if kind == "kde" else "Count")
    ax.set_ylabel(ylabel_text)
    ax.set_title(title or f"Distribution of {column}")
    
    # Plot based on type and grouping
    if kind == "hist":
        if group_by and group_by in data.columns:
            # Grouped histogram
            groups = data[group_by].unique()
            
            # Limit to top 5 groups if there are too many
            if len(groups) > 5:
                # Determine top groups by count
                top_groups = data[group_by].value_counts().nlargest(5).index
                plot_data = data[data[group_by].isin(top_groups)]
                logger.info(f"Limiting plot to top 5 groups out of {len(groups)}")
            else:
                plot_data = data
            
            # Create histogram with multiple groups
            sns.histplot(
                data=plot_data,
                x=column,
                hue=group_by,
                element="step",
                stat="density",
                common_norm=False,
                ax=ax,
                **kwargs
            )
            
            # Add legend
            plt.legend(title=group_by)
            
        else:
            # Simple histogram
            sns.histplot(
                data=data,
                x=column,
                ax=ax,
                **kwargs
            )
    
    elif kind == "kde":
        if group_by and group_by in data.columns:
            # Grouped KDE
            groups = data[group_by].unique()
            
            # Limit to top 5 groups if there are too many
            if len(groups) > 5:
                # Determine top groups by count
                top_groups = data[group_by].value_counts().nlargest(5).index
                plot_data = data[data[group_by].isin(top_groups)]
                logger.info(f"Limiting plot to top 5 groups out of {len(groups)}")
            else:
                plot_data = data
            
            # Create KDE with multiple groups
            sns.kdeplot(
                data=plot_data,
                x=column,
                hue=group_by,
                common_norm=False,
                ax=ax,
                **kwargs
            )
            
            # Add legend
            plt.legend(title=group_by)
            
        else:
            # Simple KDE
            sns.kdeplot(
                data=data,
                x=column,
                ax=ax,
                **kwargs
            )
    
    elif kind == "ecdf":
        if group_by and group_by in data.columns:
            # Grouped ECDF
            groups = data[group_by].unique()
            
            # Limit to top 5 groups if there are too many
            if len(groups) > 5:
                # Determine top groups by count
                top_groups = data[group_by].value_counts().nlargest(5).index
                plot_data = data[data[group_by].isin(top_groups)]
                logger.info(f"Limiting plot to top 5 groups out of {len(groups)}")
            else:
                plot_data = data
            
            # Create ECDF with multiple groups
            sns.ecdfplot(
                data=plot_data,
                x=column,
                hue=group_by,
                ax=ax,
                **kwargs
            )
            
            # Add legend
            plt.legend(title=group_by)
            
        else:
            # Simple ECDF
            sns.ecdfplot(
                data=data,
                x=column,
                ax=ax,
                **kwargs
            )
            
        # Override y-label
        ax.set_ylabel(ylabel or "Cumulative Probability")
    
    elif kind == "box":
        if group_by and group_by in data.columns:
            # Box plot with groups
            sns.boxplot(
                data=data,
                x=group_by,
                y=column,
                ax=ax,
                **kwargs
            )
            
            # Adjust labels
            ax.set_xlabel(xlabel or group_by)
            ax.set_ylabel(ylabel or column)
            
        else:
            # Simple box plot
            sns.boxplot(
                data=data,
                y=column,
                ax=ax,
                **kwargs
            )
            
            # Adjust labels
            ax.set_xlabel("")
    
    elif kind == "violin":
        if group_by and group_by in data.columns:
            # Violin plot with groups
            sns.violinplot(
                data=data,
                x=group_by,
                y=column,
                ax=ax,
                **kwargs
            )
            
            # Adjust labels
            ax.set_xlabel(xlabel or group_by)
            ax.set_ylabel(ylabel or column)
            
        else:
            # Simple violin plot
            sns.violinplot(
                data=data,
                y=column,
                ax=ax,
                **kwargs
            )
            
            # Adjust labels
            ax.set_xlabel("")
    
    else:
        raise ValueError(f"Unsupported distribution plot type: {kind}")
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        # Create directory if it doesn't exist
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_correlation_matrix(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "coolwarm",
    annot: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> plt.Figure:
    """Plot a correlation matrix.
    
    Args:
        data: DataFrame with data to analyze.
        columns: List of columns to include (defaults to all numeric columns).
        method: Correlation method ('pearson', 'spearman', 'kendall').
        title: Optional plot title.
        figsize: Figure size as (width, height).
        cmap: Colormap name.
        annot: Whether to show correlation values.
        save_path: Optional path to save the figure.
        **kwargs: Additional arguments for plotting.
        
    Returns:
        Matplotlib figure object.
    """
    # Filter numeric columns if not specified
    if columns is None:
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        # Remove columns with all NaN or constant values
        valid_columns = []
        for col in numeric_columns:
            if not data[col].isna().all() and data[col].nunique() > 1:
                valid_columns.append(col)
        
        plot_data = data[valid_columns]
        logger.info(f"Using {len(valid_columns)} numeric columns for correlation matrix")
    else:
        # Filter to only include specified columns that exist
        valid_columns = [col for col in columns if col in data.columns]
        plot_data = data[valid_columns]
        
        if len(valid_columns) != len(columns):
            missing = set(columns) - set(valid_columns)
            logger.warning(f"Columns not found in data: {missing}")
    
    # Compute correlation matrix
    corr_matrix = plot_data.corr(method=method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=annot,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        **kwargs
    )
    
    # Set title
    ax.set_title(title or f"Correlation Matrix ({method.capitalize()})")
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        # Create directory if it doesn't exist
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    add_regression: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> plt.Figure:
    """Plot a scatter plot.
    
    Args:
        data: DataFrame with data to plot.
        x: Name of the column for x-axis.
        y: Name of the column for y-axis.
        group_by: Optional column to group by (for color coding).
        title: Optional plot title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        figsize: Figure size as (width, height).
        add_regression: Whether to add a regression line.
        save_path: Optional path to save the figure.
        **kwargs: Additional arguments for plotting.
        
    Returns:
        Matplotlib figure object.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot based on grouping
    if group_by and group_by in data.columns:
        # Get unique groups
        groups = data[group_by].unique()
        
        # Limit to top 10 groups if there are too many
        if len(groups) > 10:
            # Determine top groups by count
            top_groups = data[group_by].value_counts().nlargest(10).index
            plot_data = data[data[group_by].isin(top_groups)]
            logger.info(f"Limiting plot to top 10 groups out of {len(groups)}")
        else:
            plot_data = data
        
        # Plot with grouping
        scatter_plot = sns.scatterplot(
            data=plot_data,
            x=x,
            y=y,
            hue=group_by,
            ax=ax,
            **kwargs
        )
        
        # Add regression line if requested
        if add_regression:
            sns.regplot(
                data=plot_data,
                x=x,
                y=y,
                scatter=False,
                ax=ax,
                line_kws={"color": "black", "linestyle": "--"}
            )
        
        # Add legend with reasonable placement
        plt.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    else:
        # Simple scatter plot
        scatter_plot = sns.scatterplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            **kwargs
        )
        
        # Add regression line if requested
        if add_regression:
            sns.regplot(
                data=data,
                x=x,
                y=y,
                scatter=False,
                ax=ax,
                line_kws={"color": "red", "linestyle": "--"}
            )
    
    # Set labels and title
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{y} vs {x}")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        # Create directory if it doesn't exist
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    return fig


def create_subplot_grid(
    n_plots: int,
    figsize: Optional[Tuple[int, int]] = None,
    max_cols: int = 3
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a grid of subplots.
    
    Args:
        n_plots: Number of plots to create.
        figsize: Optional figure size as (width, height).
        max_cols: Maximum number of columns in the grid.
        
    Returns:
        Tuple of (Figure, array of Axes).
    """
    # Calculate number of rows and columns
    n_cols = min(n_plots, max_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Convert axes to array for easier indexing
    if n_plots == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = np.array([axes]).flatten()
    else:
        axes = axes.flatten()
    
    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig, axes