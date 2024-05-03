import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

def load_data(filepath):
    """Load trajectory data from a CSV file."""
    df = pd.read_csv(filepath)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    return gdf

def apply_cdp_noise(data, sensitivity_map, user_density, sparse=False):
    """Apply CDP noise based on the sensitivity and density maps."""
    noise_levels = data.apply(
        lambda row: calculate_contextual_noise(
            row['latitude'], row['longitude'], row['timestamp'], sensitivity_map, user_density, sparse), axis=1)
    noisy_data = data.copy()
    noisy_data[['latitude', 'longitude']] += np.vstack(noise_levels)
    return noisy_data

def calculate_contextual_noise(latitude, longitude, timestamp, sensitivity_map, density_map, sparse):
    """Generate noise based on CDP principles."""
    point = Point(longitude, latitude)
    sensitivity_level = sensitivity_map.loc[sensitivity_map.contains(point), 'sensitivity'].iloc[0]
    user_density = density_map.loc[
        (density_map['latitude'] == latitude) & (density_map['longitude'] == longitude), 'density'].iloc[0]
    # Adjust noise level based on density
    if sparse:
        sigma = sensitivity_level * np.sqrt(user_density) * 1.5  # Increase noise in sparse areas
    else:
        sigma = sensitivity_level * np.sqrt(user_density) * 0.5  # Reduce noise in dense areas
    
    noise = np.random.normal(0, sigma, 2)  # Noise for latitude and longitude
    return noise

def run_experiment(data, sensitivity_map, density_map, sparse=False):
    """Run the CDP experiment and evaluate privacy metrics."""
    noisy_data = apply_cdp_noise(data, sensitivity_map, density_map, sparse)
    # Evaluate privacy and utility metrics here (e.g., differential privacy, information loss)
    print("Experiment results:", noisy_data.head())

# Load data
t_drive_data = load_data('data/tdrive_output.csv')  # High-density dataset
sparse_data = load_data('t-drive.csv')  # Simulated low-density dataset

# Example sensitivity and density maps
sensitivity_map = gpd.GeoDataFrame({
    'geometry': [Point(116.397, 39.908), Point(116.398, 39.909)],  # Beijing central points
    'sensitivity': [0.5, 0.5]
})

density_map = pd.DataFrame({
    'latitude': [39.908, 39.909],
    'longitude': [116.397, 116.398],
    'density': [100, 10]  # Simulated density levels
})

# Run experiments
run_experiment(t_drive_data, sensitivity_map, density_map, sparse=False)
run_experiment(sparse_data, sensitivity_map, density_map, sparse=True)

def calculate_mean_error(original, protected):
    """Calculate the absolute error in mean between the original and protected datasets."""
    return np.abs(original.mean() - protected.mean())

def calculate_rmse(original, protected):
    """Calculate the Root Mean Squared Error between the original and protected datasets."""
    return np.sqrt(((original - protected) ** 2).mean())

def analyze_utility(data_paths):
    """Load datasets and compute utility metrics."""
    original_data = pd.read_csv(data_paths['original'])
    traditional_dp_data = pd.read_csv(data_paths['traditional'])
    cdp_data = pd.read_csv(data_paths['cdp'])

    metrics = {
        'Mean Error - Traditional DP': calculate_mean_error(original_data['data_value'], traditional_dp_data['data_value']),
        'Mean Error - CDP': calculate_mean_error(original_data['data_value'], cdp_data['data_value']),
        'RMSE - Traditional DP': calculate_rmse(original_data['data_value'], traditional_dp_data['data_value']),
        'RMSE - CDP': calculate_rmse(original_data['data_value'], cdp_data['data_value'])
    }

    for key, value in metrics.items():
        print(f"{key}: {value}")