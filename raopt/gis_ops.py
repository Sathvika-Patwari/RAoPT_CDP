import osmnx as ox
import geopandas as gpd
import os
from unidecode import unidecode

# Setup OSMnx with the new settings approach
def setup_osmnx():
    ox.settings.use_cache = True
    ox.settings.log_console = True

def download_and_process_osm_data(place_name):
    buildings = ox.features_from_place(place_name, tags={'building': True})
    return buildings

def assign_sensitivity(buildings):
    def sensitivity_level(building_type):
        if building_type in ['hospital', 'school', 'kindergarten']:
            return 'high'
        elif building_type in ['commercial', 'retail']:
            return 'medium'
        return 'low'
    buildings['sensitivity'] = buildings['building'].apply(sensitivity_level)
    return buildings

def sanitize_column_names(gdf):
    sanitized_columns = {}
    for column in gdf.columns:
        sanitized_column = unidecode(column).replace(':', '_').replace(' ', '_').replace('-', '_')
        sanitized_columns[column] = sanitized_column[:10]  # Shapefile column limit
    gdf.rename(columns=sanitized_columns, inplace=True)

def clean_data_for_export(gdf):
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, list)).any():
            gdf[col] = gdf[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
    return gdf

def save_data(gdf, filepath, format='GeoJSON'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    gdf = clean_data_for_export(gdf)
    if format == 'Shapefile':
        sanitize_column_names(gdf)
        gdf.to_file(filepath + '.shp', driver='ESRI Shapefile')
    else:
        gdf.to_file(filepath + '.geojson', driver='GeoJSON')
