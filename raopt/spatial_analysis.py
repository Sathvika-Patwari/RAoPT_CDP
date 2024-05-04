import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
def calculate_density(data):
    """Calculate density of data points in geographic grids."""
    if 'longitude' not in data.columns or 'latitude' not in data.columns:
        raise ValueError("Data must contain 'longitude' and 'latitude' columns.")
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))
    gdf.crs = "EPSG:4326"
    grid_size = 0.01  # Decimal degrees, about 1km at the equator
    xmin, ymin, xmax, ymax = gdf.total_bounds
    cols = int((xmax - xmin) / grid_size)
    rows = int((ymax - ymin) / grid_size)
    grid = gpd.GeoDataFrame(geometry=[Point(xmin + x * grid_size, ymin + y * grid_size) 
                                      for y in range(rows) for x in range(cols)], crs=gdf.crs)
    density = gpd.sjoin(gdf, grid, how="inner", op='intersects')
    density = density.groupby(density.geometry).size().reset_index(name='count')
    
    return density

def process_folder_data(folder_path, output_path):
    """Process all CSV files in a given folder."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            density = calculate_density(data)
            density_output_path = os.path.join(output_path, f'density_{filename}')
            density.to_csv(density_output_path, index=False)
            print(f"Density map saved to {density_output_path}")