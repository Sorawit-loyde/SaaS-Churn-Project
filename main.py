import sys
import os

# Import modules from src package
from src import (
    load_master_data,
    check_null_churn_data,
    preprocess_features,
    train_and_evaluate
)

def main():
    print("=== SaaS Churn Prediction Pipeline ===\n")
    
    print("[STEP 1] Loading raw data...")
    raw_df = load_master_data()
    df = raw_df.copy()
    print(f"Loaded {len(df)} rows.\n")
    
    print("[STEP 2] Cleaning and transforming data...")
    df = check_null_churn_data(df)
    df = preprocess_features(df)
    print("Data transformation complete. Features are ready.\n")
    
    print("[STEP 3] Training the model and generating reports...")
    model, predictions, y_test, full_probabilities = train_and_evaluate(df)
    
    print("\n[STEP 4] Exporting Data for Dashboard...")
    dashboard_df = raw_df.copy()
    dashboard_df['Churn_Probability'] = full_probabilities
    
    os.makedirs('data/processed', exist_ok=True)
    dashboard_file = 'data/processed/dashboard_data.csv'
    dashboard_df.to_csv(dashboard_file, index=False)
    print(f"Exported dashboard data to {dashboard_file}!")
    
    print("\n=== Pipeline Complete ===")
    print("Check the 'reports' folder for charts and 'data/processed/dashboard_data.csv' for the dashboard data.")

if __name__ == "__main__":
    main()
