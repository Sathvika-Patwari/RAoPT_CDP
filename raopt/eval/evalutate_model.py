import pandas as pd
import numpy as np
from dp.sdd import Model, apply_sdd_noise, apply_cdp_noise, load_data

def calculate_information_loss(original_data, protected_data):
    """Calculate the mean squared error as information loss."""
    return np.mean((original_data - protected_data) ** 2)

def evaluate_model(protected_data, original_data, model):
    """Evaluate the model and calculate accuracy and information loss."""
    predictions = model.predict(protected_data)
    accuracy = np.mean(predictions == original_data)  # Simplistic accuracy calculation
    info_loss = calculate_information_loss(original_data, protected_data)
    return accuracy, info_loss

def main_evaluation():
    # Load data
    data, sensitivity_map, density_map = load_data()
    model = Model()

    # Apply SDD
    sdd_protected_data = apply_sdd_noise(data)
    sdd_accuracy, sdd_info_loss = evaluate_model(sdd_protected_data, data, model)

    # Apply CDP
    cdp_protected_data = apply_cdp_noise(data, sensitivity_map, density_map)
    cdp_accuracy, cdp_info_loss = evaluate_model(cdp_protected_data, data, model)

    # Create a DataFrame to hold the comparison results
    results_df = pd.DataFrame({
        "Method": ["SDD", "CDP"],
        "Accuracy": [sdd_accuracy, cdp_accuracy],
        "Information Loss": [sdd_info_loss, cdp_info_loss]
    })

    print(results_df)
    return results_df

if __name__ == "__main__":
    comparison_table = main_evaluation()
    comparison_table.to_csv("privacy_comparison_results.csv", index=False)
