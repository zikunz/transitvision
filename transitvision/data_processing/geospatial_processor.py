"""Geospatial data processor for transportation data."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import os

from .base_processor import BaseDataProcessor

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask
    import pyproj
    GEOSPATIAL_DEPS_AVAILABLE = True
except ImportError:
    logger.warning("Optional geospatial dependencies (geopandas, rasterio, pyproj) not available")
    GEOSPATIAL_DEPS_AVAILABLE = False


class GeospatialProcessor(BaseDataProcessor):
    """Processor for geospatial transportation data.
    
    This class handles processing of geospatial data related to transportation,
    including satellite imagery, GIS data, and location-based transit information.
    """
    
    def __init__(
        self,
        raw_data_dir: Optional[Union[str, Path]] = None,
        processed_data_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the GeospatialProcessor.
        
        Args:
            raw_data_dir: Directory containing raw geospatial data files.
            processed_data_dir: Directory where processed data will be saved.
            config: Configuration parameters for the processor.
        """
        super().__init__(raw_data_dir, processed_data_dir, config)
        
        # Check dependencies
        if not GEOSPATIAL_DEPS_AVAILABLE:
            logger.warning(
                "GeospatialProcessor requires geopandas, rasterio, and pyproj. "
                "Some functionality will be limited."
            )
        
        # Default configuration for geospatial processing
        default_config = {
            "crs": "EPSG:4326",  # Default coordinate reference system (WGS84)
            "buffer_distance": 500,  # Default buffer distance in meters
            "raster_resolution": 30,  # Default raster resolution in meters
            "lat_column": "latitude",
            "lon_column": "longitude",
            "geometry_column": "geometry",
            "id_column": "id",
        }
        
        # Update default config with user-provided config
        if config:
            for key, value in config.items():
                default_config[key] = value
        
        self.config = default_config
    
    def load_data(self, file_path: Union[str, Path]) -> Union[pd.DataFrame, 'gpd.GeoDataFrame']:
        """Load geospatial data from specified file path.
        
        Args:
            file_path: Path to the geospatial data file.
            
        Returns:
            GeoDataFrame or DataFrame containing the loaded geospatial data.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Geospatial data file not found: {file_path}")
        
        try:
            # Handle GIS vector data formats if geopandas is available
            if GEOSPATIAL_DEPS_AVAILABLE:
                if file_path.suffix in [".shp", ".geojson", ".gpkg"]:
                    data = gpd.read_file(file_path)
                    logger.info(f"Loaded GeoDataFrame with {len(data)} features from {file_path}")
                    return data
                elif file_path.suffix == ".tif":
                    # For raster data, just return the file path for later processing
                    logger.info(f"Detected raster file: {file_path}")
                    return pd.DataFrame({"raster_path": [str(file_path)]})
            
            # Fall back to pandas for tabular data with coordinates
            if file_path.suffix == ".csv":
                data = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                data = pd.read_parquet(file_path)
            elif file_path.suffix in [".xlsx", ".xls"]:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded {len(data)} records from {file_path}")
            
            # Convert to GeoDataFrame if coordinates are present and dependencies available
            if (GEOSPATIAL_DEPS_AVAILABLE and 
                self.config["lat_column"] in data.columns and 
                self.config["lon_column"] in data.columns):
                
                try:
                    # Create point geometries
                    geometry = gpd.points_from_xy(
                        data[self.config["lon_column"]], 
                        data[self.config["lat_column"]]
                    )
                    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=self.config["crs"])
                    logger.info(f"Converted DataFrame to GeoDataFrame with {len(gdf)} points")
                    return gdf
                except Exception as e:
                    logger.warning(f"Failed to convert to GeoDataFrame: {str(e)}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def _convert_to_geodataframe(self, data: pd.DataFrame) -> Union[pd.DataFrame, 'gpd.GeoDataFrame']:
        """Convert DataFrame with lat/lon columns to GeoDataFrame.
        
        Args:
            data: DataFrame with latitude and longitude columns.
            
        Returns:
            GeoDataFrame if conversion is successful, otherwise the original DataFrame.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            logger.warning("Cannot convert to GeoDataFrame: geospatial dependencies not available")
            return data
        
        try:
            lat_col = self.config["lat_column"]
            lon_col = self.config["lon_column"]
            
            if lat_col in data.columns and lon_col in data.columns:
                # Filter out rows with missing coordinates
                valid_coords = data[[lat_col, lon_col]].notna().all(axis=1)
                if not valid_coords.all():
                    logger.warning(
                        f"Found {(~valid_coords).sum()} rows with missing coordinates. "
                        "These rows will be excluded from spatial operations."
                    )
                
                # Create point geometries for valid coordinates
                points = gpd.points_from_xy(
                    data.loc[valid_coords, lon_col],
                    data.loc[valid_coords, lat_col]
                )
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    data.loc[valid_coords].copy(), 
                    geometry=points,
                    crs=self.config["crs"]
                )
                
                logger.info(f"Created GeoDataFrame with {len(gdf)} points")
                return gdf
            else:
                logger.warning(
                    f"Coordinate columns '{lat_col}' and/or '{lon_col}' not found in DataFrame. "
                    "Cannot convert to GeoDataFrame."
                )
                return data
        
        except Exception as e:
            logger.error(f"Error converting to GeoDataFrame: {str(e)}")
            return data
    
    def _reproject_data(self, data: Union[pd.DataFrame, 'gpd.GeoDataFrame'], target_crs: str) -> Union[pd.DataFrame, 'gpd.GeoDataFrame']:
        """Reproject geospatial data to a different coordinate reference system.
        
        Args:
            data: GeoDataFrame to reproject.
            target_crs: Target coordinate reference system.
            
        Returns:
            Reprojected GeoDataFrame if input is a GeoDataFrame, otherwise the original DataFrame.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            logger.warning("Cannot reproject data: geospatial dependencies not available")
            return data
        
        if not isinstance(data, gpd.GeoDataFrame):
            logger.warning("Cannot reproject data: input is not a GeoDataFrame")
            return data
        
        try:
            # Check if reprojection is needed
            if data.crs and data.crs == target_crs:
                logger.info(f"Data already in target CRS: {target_crs}")
                return data
            
            # Reproject
            reprojected = data.to_crs(target_crs)
            logger.info(f"Reprojected data from {data.crs} to {target_crs}")
            return reprojected
        
        except Exception as e:
            logger.error(f"Error reprojecting data: {str(e)}")
            return data
    
    def _calculate_spatial_features(self, data: Union[pd.DataFrame, 'gpd.GeoDataFrame'], 
                                   pois: Optional[Union[pd.DataFrame, 'gpd.GeoDataFrame']] = None) -> pd.DataFrame:
        """Calculate spatial features for transit data.
        
        Args:
            data: GeoDataFrame with transit locations.
            pois: Optional GeoDataFrame with points of interest.
            
        Returns:
            DataFrame with additional spatial features.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            logger.warning("Cannot calculate spatial features: geospatial dependencies not available")
            return data
        
        if not isinstance(data, gpd.GeoDataFrame):
            data = self._convert_to_geodataframe(data)
            if not isinstance(data, gpd.GeoDataFrame):
                logger.warning("Cannot calculate spatial features: failed to create GeoDataFrame")
                return data
        
        try:
            # Ensure data is in a projected CRS for distance calculations
            projected_crs = "EPSG:3857"  # Web Mercator
            data_projected = self._reproject_data(data, projected_crs)
            
            # Initialize result DataFrame
            result = data.copy()
            
            # Calculate basic spatial features
            if isinstance(data_projected, gpd.GeoDataFrame):
                # Calculate area
                if 'area' in self.config.get('spatial_features', []):
                    result['area'] = data_projected.geometry.area
                    logger.info("Calculated area feature")
                
                # Calculate perimeter for polygon geometries
                if ('perimeter' in self.config.get('spatial_features', []) and 
                    data_projected.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).any()):
                    result['perimeter'] = data_projected.geometry.length
                    logger.info("Calculated perimeter feature for polygons")
            
            # Calculate distance to nearest POI if provided
            if pois is not None and isinstance(pois, gpd.GeoDataFrame):
                # Ensure POIs are in the same CRS
                pois_projected = self._reproject_data(pois, projected_crs)
                
                if isinstance(pois_projected, gpd.GeoDataFrame):
                    # Calculate distance to nearest POI
                    distances = []
                    nearest_poi_ids = []
                    
                    for idx, point in data_projected.geometry.items():
                        # Calculate distances to all POIs
                        poi_distances = pois_projected.geometry.distance(point)
                        # Find minimum distance
                        min_dist_idx = poi_distances.idxmin()
                        min_dist = poi_distances.loc[min_dist_idx]
                        # Get POI ID
                        poi_id = pois_projected.loc[min_dist_idx, self.config["id_column"]]
                        
                        distances.append(min_dist)
                        nearest_poi_ids.append(poi_id)
                    
                    result['distance_to_nearest_poi'] = distances
                    result['nearest_poi_id'] = nearest_poi_ids
                    logger.info("Calculated distance to nearest POI feature")
            
            return result
        
        except Exception as e:
            logger.error(f"Error calculating spatial features: {str(e)}")
            return data
    
    def _process_raster_data(self, data: pd.DataFrame, raster_path: str) -> pd.DataFrame:
        """Process raster data using transit locations.
        
        Args:
            data: GeoDataFrame with transit locations.
            raster_path: Path to the raster file.
            
        Returns:
            DataFrame with extracted raster values.
        """
        if not GEOSPATIAL_DEPS_AVAILABLE:
            logger.warning("Cannot process raster data: geospatial dependencies not available")
            return data
        
        if not isinstance(data, gpd.GeoDataFrame):
            data = self._convert_to_geodataframe(data)
            if not isinstance(data, gpd.GeoDataFrame):
                logger.warning("Cannot process raster data: failed to create GeoDataFrame")
                return data
        
        try:
            if not os.path.exists(raster_path):
                logger.error(f"Raster file not found: {raster_path}")
                return data
            
            # Result DataFrame
            result = data.copy()
            
            # Open raster file
            with rasterio.open(raster_path) as src:
                # Extract raster values at each point
                raster_values = []
                
                for point in data.geometry:
                    # Get pixel value at point location
                    x, y = point.x, point.y
                    
                    # Transform coordinates to raster CRS if needed
                    if data.crs and data.crs != src.crs:
                        transformer = pyproj.Transformer.from_crs(
                            data.crs, 
                            src.crs,
                            always_xy=True
                        )
                        x, y = transformer.transform(x, y)
                    
                    # Get raster row, col
                    row, col = src.index(x, y)
                    
                    # Get pixel value (handle out of bounds)
                    try:
                        value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                    except:
                        value = np.nan
                    
                    raster_values.append(value)
                
                # Add to result
                result['raster_value'] = raster_values
                
                # Add band metadata if available
                if src.tags():
                    # Extract band name or description if available
                    band_name = src.tags().get('band_name', 'band_1')
                    result.rename(columns={'raster_value': band_name}, inplace=True)
                
                logger.info(f"Extracted raster values from {raster_path}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing raster data: {str(e)}")
            return data
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process geospatial transportation data.
        
        Args:
            data: Raw geospatial data as a DataFrame or GeoDataFrame.
            
        Returns:
            Processed DataFrame with spatial features.
        """
        logger.info("Starting geospatial data processing pipeline")
        
        # Skip processing if dependencies aren't available
        if not GEOSPATIAL_DEPS_AVAILABLE:
            logger.warning(
                "Geospatial processing skipped due to missing dependencies. "
                "Install geopandas, rasterio, and pyproj for full functionality."
            )
            return data
        
        # Apply processing steps
        df = data.copy()
        
        # Convert to GeoDataFrame if needed
        if not isinstance(df, gpd.GeoDataFrame):
            df = self._convert_to_geodataframe(df)
        
        # Process based on data type
        if isinstance(df, gpd.GeoDataFrame):
            # Reproject if needed
            if self.config.get("reproject", False):
                df = self._reproject_data(df, self.config["crs"])
            
            # Calculate spatial features
            df = self._calculate_spatial_features(df)
        
        # Process raster data if path is in the data
        if "raster_path" in df.columns:
            for raster_path in df["raster_path"].unique():
                if pd.notna(raster_path):
                    df = self._process_raster_data(df, raster_path)
        
        # Log processing summary
        logger.info(f"Geospatial data processing complete. Output shape: {df.shape}")
        
        # Convert back to DataFrame for compatibility
        if isinstance(df, gpd.GeoDataFrame):
            # Add lat/lon columns from geometry
            df[self.config["lon_column"]] = df.geometry.x
            df[self.config["lat_column"]] = df.geometry.y
            
            # Convert to DataFrame
            df = pd.DataFrame(df.drop(columns="geometry"))
        
        return df