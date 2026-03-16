import pandas as pd
import os

def load_master_data(filepath='data/processed/master_customer_data.csv'):
    """Loads and returns the cleaned customer data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    
    df = pd.read_csv(filepath)
    # Perform any final tweaks (e.g., converting types)
    return df

if __name__ == "__main__":
    # This block only runs if you execute this file directly for testing
    data = load_master_data()
    print(f"Successfully loaded {len(data)} rows.")