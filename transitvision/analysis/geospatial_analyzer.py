"""Geospatial data analysis module for transportation data."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Check for optional dependencies
try:
    import geopandas as gpd
    import contextily as ctx
    from shapely.geometry import Point, LineString, Polygon
    import pyproj
    GEOSPATIAL_DEPS_AVAILABLE = True
except ImportError:
    logging.warning("Optional geospatial dependencies not available for full functionality")
    GEOSPATIAL_DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)


class GeospatialAnalyzer:
    """Analyze geospatial transportation data.
    
    This class provides methods for analyzing geospatial aspects of transit data,
    including spatial patterns, clustering, and visualization.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the geospatial analyzer.
        
        Args:
            config: Configuration parameters for the analyzer.
        """
        # Check dependencies
        if not GEOSPATIAL_DEPS_AVAILABLE:
            logger.warning(
                "GeospatialAnalyzer requires geopandas, contextily, shapely, and pyproj. "
                "Some functionality will be limited."
            )
        
        # Default configuration
        default_config = {
            "lat_column": "latitude",
            "lon_column": "longitude",
            "route_column": "route_id",
            "stop_column": "stop_id",
            "ridership_column": "ridership",
            "crs": "EPSG:4326",  # WGS84
            "map_crs": "EPSG:3857",  # Web Mercator for base maps
            "buffer_distance": 500,  # meters
            "plot_figsize": (12, 10),
            "plot_style": "whitegrid",
        }
        
        # Update default config with user-provided config
        self.config = default_config
        if config:
            for key, value in config.items():
                self.config[key] = value
        
        # Set plot style
        sns.set_style(self.config["plot_style"])
    
    def load_data(self, file_path: Union[str, Path]) -> Union[pd.DataFrame, 'gpd.GeoDataFrame']:
        """Load processed geospatial data for analysis.
        
        Args:
            file_path: Path to the processed geospatial data file.
            
        Returns:
            DataFrame or GeoDataFrame containing the geospatial data.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Handle GIS formats if geopandas is available
            if GEOSPATIAL_DEPS_AVAILABLE:
                if file_path.suffix in [".shp", ".geojson", ".gpkg"]:
                    data = gpd.read_file(file_path)
                    logger.info(f"Loaded GeoDataFrame with {len(data)} features from {file_path}")
                    return data
            
            # Fall back to pandas for tabular data
            if file_path.suffix == ".csv":
                data = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                data = pd.read_parquet(file_path)
            elif file_path.suffix in [".pkl", ".pickle"]:
                data = pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded {len(data)} records from {file_path}")
            
            # Convert to GeoDataFrame if coordinates are present and dependencies available
            if (GEOSPATIAL_DEPS_AVAILABLE and 
                self.config["lat_column"] in data.columns and 
                self.config["lon_column"] in data.columns):
                
                # Create point geometries
                geometry = gpd.points_from_xy(
                    data[self.config["lon_column"]], 
                    data[self.config["lat_column"]]
                )
                gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=self.config["crs"])
                logger.info(f"Converted DataFrame to GeoDataFrame with {len(gdf)} points")
                return gdf
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def _ensure_geodataframe(self, data: Union[pd.DataFrame, 'gpd.GeoDataFrame']) -> 'gpd.GeoDataFrame':
        """Ensure data is a GeoDataFrame with proper geometry.
        
        Args:
            data: Input data.
            
        Returns:
            Data as a GeoDataFrame.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            raise ImportError("Geospatial dependencies required for this operation")
        
        if isinstance(data, gpd.GeoDataFrame) and data.geometry.notna().all():
            return data
        
        df = data.copy()
        
        # Create geometry from lat/lon columns
        lat_col = self.config["lat_column"]
        lon_col = self.config["lon_column"]
        
        if lat_col in df.columns and lon_col in df.columns:
            # Filter out rows with missing coordinates
            valid_coords = df[[lat_col, lon_col]].notna().all(axis=1)
            if not valid_coords.all():
                logger.warning(
                    f"Found {(~valid_coords).sum()} rows with missing coordinates. "
                    "These rows will be excluded from spatial operations."
                )
            
            # Create point geometries
            geometry = gpd.points_from_xy(
                df.loc[valid_coords, lon_col],
                df.loc[valid_coords, lat_col]
            )
            
            gdf = gpd.GeoDataFrame(
                df.loc[valid_coords].copy(),
                geometry=geometry,
                crs=self.config["crs"]
            )
            
            return gdf
        else:
            raise ValueError(
                f"Cannot convert to GeoDataFrame: "
                f"lat/lon columns '{lat_col}/{lon_col}' not found"
            )
    
    def calculate_stop_density(
        self, 
        data: Union[pd.DataFrame, 'gpd.GeoDataFrame'],
        grid_size: float = 1.0,  # km
        crs: Optional[str] = None,
    ) -> 'gpd.GeoDataFrame':
        """Calculate transit stop density on a grid.
        
        Args:
            data: Transit stop data.
            grid_size: Size of grid cells in kilometers.
            crs: Optional coordinate reference system for the grid.
            
        Returns:
            GeoDataFrame with grid cells and stop counts.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            raise ImportError("Geospatial dependencies required for this operation")
        
        # Ensure data is a GeoDataFrame
        gdf = self._ensure_geodataframe(data)
        
        # Set CRS if provided
        if crs:
            gdf = gdf.to_crs(crs)
        
        # Create projected version for distance calculations
        if gdf.crs and not gdf.crs.is_projected:
            # Project to a suitable CRS (UTM or equal area)
            projected_gdf = gdf.to_crs(self.config["map_crs"])
        else:
            projected_gdf = gdf
        
        # Get the bounding box
        minx, miny, maxx, maxy = projected_gdf.total_bounds
        
        # Convert grid size to projected units (meters)
        grid_size_m = grid_size * 1000
        
        # Create grid cells
        x_points = np.arange(minx, maxx + grid_size_m, grid_size_m)
        y_points = np.arange(miny, maxy + grid_size_m, grid_size_m)
        
        cells = []
        for x in x_points[:-1]:
            for y in y_points[:-1]:
                cells.append(Polygon([
                    (x, y),
                    (x + grid_size_m, y),
                    (x + grid_size_m, y + grid_size_m),
                    (x, y + grid_size_m)
                ]))
        
        # Create grid GeoDataFrame
        grid = gpd.GeoDataFrame(geometry=cells, crs=projected_gdf.crs)
        
        # Count points in each grid cell
        grid["stop_count"] = grid.geometry.apply(
            lambda cell: sum(projected_gdf.geometry.intersects(cell))
        )
        
        # Calculate area in km²
        grid["area_km2"] = grid.geometry.area / 1_000_000
        
        # Calculate density
        grid["stop_density"] = grid["stop_count"] / grid["area_km2"]
        
        # Convert back to input CRS if needed
        if crs and grid.crs != gdf.crs:
            grid = grid.to_crs(gdf.crs)
        
        logger.info(f"Calculated stop density on {len(grid)} grid cells")
        
        return grid
    
    def calculate_ridership_heatmap(
        self, 
        data: Union[pd.DataFrame, 'gpd.GeoDataFrame'],
        grid_size: float = 1.0,  # km
        crs: Optional[str] = None,
    ) -> 'gpd.GeoDataFrame':
        """Calculate transit ridership heatmap on a grid.
        
        Args:
            data: Transit ridership data with stops.
            grid_size: Size of grid cells in kilometers.
            crs: Optional coordinate reference system for the grid.
            
        Returns:
            GeoDataFrame with grid cells and ridership values.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            raise ImportError("Geospatial dependencies required for this operation")
        
        # Ensure data is a GeoDataFrame
        gdf = self._ensure_geodataframe(data)
        
        # Check for ridership column
        ridership_col = self.config["ridership_column"]
        if ridership_col not in gdf.columns:
            raise ValueError(f"Ridership column '{ridership_col}' not found in data")
        
        # Set CRS if provided
        if crs:
            gdf = gdf.to_crs(crs)
        
        # Create projected version for distance calculations
        if gdf.crs and not gdf.crs.is_projected:
            # Project to a suitable CRS (UTM or equal area)
            projected_gdf = gdf.to_crs(self.config["map_crs"])
        else:
            projected_gdf = gdf
        
        # Get the bounding box
        minx, miny, maxx, maxy = projected_gdf.total_bounds
        
        # Convert grid size to projected units (meters)
        grid_size_m = grid_size * 1000
        
        # Create grid cells
        x_points = np.arange(minx, maxx + grid_size_m, grid_size_m)
        y_points = np.arange(miny, maxy + grid_size_m, grid_size_m)
        
        cells = []
        for x in x_points[:-1]:
            for y in y_points[:-1]:
                cells.append(Polygon([
                    (x, y),
                    (x + grid_size_m, y),
                    (x + grid_size_m, y + grid_size_m),
                    (x, y + grid_size_m)
                ]))
        
        # Create grid GeoDataFrame
        grid = gpd.GeoDataFrame(geometry=cells, crs=projected_gdf.crs)
        
        # Spatial join to get points in each cell
        joined = gpd.sjoin(projected_gdf, grid, how="inner", predicate="within")
        
        # Group by cell index and sum ridership
        ridership_by_cell = joined.groupby("index_right")[ridership_col].sum().reset_index()
        
        # Merge back to grid
        grid = grid.merge(
            ridership_by_cell,
            left_index=True,
            right_on="index_right",
            how="left"
        )
        
        # Fill NaN values with 0
        grid[ridership_col] = grid[ridership_col].fillna(0)
        
        # Calculate area in km²
        grid["area_km2"] = grid.geometry.area / 1_000_000
        
        # Calculate ridership density
        grid["ridership_density"] = grid[ridership_col] / grid["area_km2"]
        
        # Convert back to input CRS if needed
        if crs and grid.crs != gdf.crs:
            grid = grid.to_crs(gdf.crs)
        
        logger.info(f"Calculated ridership heatmap on {len(grid)} grid cells")
        
        return grid
    
    def analyze_accessibility(
        self, 
        stops_data: Union[pd.DataFrame, 'gpd.GeoDataFrame'],
        poi_data: Union[pd.DataFrame, 'gpd.GeoDataFrame'],
        buffer_distance: Optional[float] = None,  # meters
    ) -> pd.DataFrame:
        """Analyze accessibility of points of interest from transit stops.
        
        Args:
            stops_data: Transit stop locations.
            poi_data: Points of interest locations.
            buffer_distance: Search radius around stops in meters.
            
        Returns:
            DataFrame with accessibility metrics.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            raise ImportError("Geospatial dependencies required for this operation")
        
        # Ensure data are GeoDataFrames
        stops_gdf = self._ensure_geodataframe(stops_data)
        poi_gdf = self._ensure_geodataframe(poi_data)
        
        # Ensure both dataframes have the same CRS
        if stops_gdf.crs != poi_gdf.crs:
            poi_gdf = poi_gdf.to_crs(stops_gdf.crs)
        
        # Create projected versions for distance calculations
        if stops_gdf.crs and not stops_gdf.crs.is_projected:
            stops_projected = stops_gdf.to_crs(self.config["map_crs"])
            poi_projected = poi_gdf.to_crs(self.config["map_crs"])
        else:
            stops_projected = stops_gdf
            poi_projected = poi_gdf
        
        # Use configured buffer distance if not provided
        if buffer_distance is None:
            buffer_distance = self.config["buffer_distance"]
        
        # Create buffers around stops
        stop_buffers = stops_projected.copy()
        stop_buffers["buffer"] = stops_projected.geometry.buffer(buffer_distance)
        
        # Use buffer geometry for analysis
        stop_buffers.set_geometry("buffer", inplace=True)
        
        # Spatial join to find POIs within buffer of each stop
        joined = gpd.sjoin(poi_projected, stop_buffers, how="inner", predicate="within")
        
        # Get counts of POIs by stop
        poi_counts = joined.groupby(stops_projected.index.name or "index").size().reset_index(name="poi_count")
        
        # Merge back to stops
        result = stops_gdf.merge(
            poi_counts,
            left_index=True,
            right_on=(stops_projected.index.name or "index"),
            how="left"
        )
        
        # Fill NaN values with 0
        result["poi_count"] = result["poi_count"].fillna(0)
        
        # Calculate accessibility metrics
        total_pois = len(poi_gdf)
        result["poi_percentage"] = result["poi_count"] / total_pois * 100
        
        # Reset geometry back to points for stops
        result = result.set_geometry("geometry")
        
        logger.info(f"Analyzed accessibility with {buffer_distance}m buffer")
        
        return result
    
    def plot_stop_density(
        self,
        density_grid: 'gpd.GeoDataFrame',
        basemap: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot transit stop density map.
        
        Args:
            density_grid: Output from calculate_stop_density.
            basemap: Whether to add a basemap.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            raise ImportError("Geospatial dependencies required for this operation")
        
        # Create figure
        figsize = self.config["plot_figsize"]
        fig, ax = plt.subplots(figsize=figsize)
        
        # Project to Web Mercator for basemap compatibility
        if basemap and density_grid.crs != self.config["map_crs"]:
            plot_grid = density_grid.to_crs(self.config["map_crs"])
        else:
            plot_grid = density_grid
        
        # Plot density with colormap
        plot_grid.plot(
            column="stop_density",
            ax=ax,
            alpha=0.7,
            cmap="YlOrRd",
            legend=True,
            legend_kwds={
                "label": "Stops per km²",
                "orientation": "vertical",
                "shrink": 0.8
            }
        )
        
        # Add basemap if requested
        if basemap:
            try:
                ctx.add_basemap(
                    ax, 
                    source=ctx.providers.CartoDB.Positron,
                    attribution_size=8
                )
            except Exception as e:
                logger.warning(f"Failed to add basemap: {str(e)}")
        
        # Set title and labels
        ax.set_title("Transit Stop Density")
        
        # Remove axis labels for map
        ax.set_axis_off()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_ridership_heatmap(
        self,
        heatmap_grid: 'gpd.GeoDataFrame',
        basemap: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot transit ridership heatmap.
        
        Args:
            heatmap_grid: Output from calculate_ridership_heatmap.
            basemap: Whether to add a basemap.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            raise ImportError("Geospatial dependencies required for this operation")
        
        # Create figure
        figsize = self.config["plot_figsize"]
        fig, ax = plt.subplots(figsize=figsize)
        
        # Project to Web Mercator for basemap compatibility
        if basemap and heatmap_grid.crs != self.config["map_crs"]:
            plot_grid = heatmap_grid.to_crs(self.config["map_crs"])
        else:
            plot_grid = heatmap_grid
        
        # Plot ridership density with colormap
        plot_grid.plot(
            column="ridership_density",
            ax=ax,
            alpha=0.7,
            cmap="viridis",
            legend=True,
            legend_kwds={
                "label": "Ridership per km²",
                "orientation": "vertical",
                "shrink": 0.8
            }
        )
        
        # Add basemap if requested
        if basemap:
            try:
                ctx.add_basemap(
                    ax, 
                    source=ctx.providers.CartoDB.Positron,
                    attribution_size=8
                )
            except Exception as e:
                logger.warning(f"Failed to add basemap: {str(e)}")
        
        # Set title
        ax.set_title("Transit Ridership Heatmap")
        
        # Remove axis labels for map
        ax.set_axis_off()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_accessibility_map(
        self,
        accessibility_data: Union[pd.DataFrame, 'gpd.GeoDataFrame'],
        metric: str = "poi_count",
        basemap: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot transit accessibility map.
        
        Args:
            accessibility_data: Output from analyze_accessibility.
            metric: Accessibility metric to plot ("poi_count" or "poi_percentage").
            basemap: Whether to add a basemap.
            save_path: Optional path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            raise ImportError("Geospatial dependencies required for this operation")
        
        # Ensure data is a GeoDataFrame
        if not isinstance(accessibility_data, gpd.GeoDataFrame):
            raise ValueError("Accessibility data must be a GeoDataFrame")
        
        # Check if metric exists
        if metric not in accessibility_data.columns:
            raise ValueError(f"Metric column '{metric}' not found in data")
        
        # Create figure
        figsize = self.config["plot_figsize"]
        fig, ax = plt.subplots(figsize=figsize)
        
        # Project to Web Mercator for basemap compatibility
        if basemap and accessibility_data.crs != self.config["map_crs"]:
            plot_data = accessibility_data.to_crs(self.config["map_crs"])
        else:
            plot_data = accessibility_data
        
        # Plot accessibility with graduated symbols
        plot_data.plot(
            column=metric,
            ax=ax,
            alpha=0.7,
            cmap="YlGnBu",
            markersize=plot_data[metric] * 5 + 10,  # Scale marker size
            legend=True,
            legend_kwds={
                "label": "POIs accessible" if metric == "poi_count" else "% of POIs accessible",
                "orientation": "vertical",
                "shrink": 0.8
            }
        )
        
        # Add basemap if requested
        if basemap:
            try:
                ctx.add_basemap(
                    ax, 
                    source=ctx.providers.CartoDB.Positron,
                    attribution_size=8
                )
            except Exception as e:
                logger.warning(f"Failed to add basemap: {str(e)}")
        
        # Set title
        ax.set_title("Transit Accessibility to Points of Interest")
        
        # Remove axis labels for map
        ax.set_axis_off()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig