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
    df = load_master_data()
    print(f"Loaded {len(df)} rows.\n")
    
    print("[STEP 2] Cleaning and transforming data...")
    df = check_null_churn_data(df)
    df = preprocess_features(df)
    print("Data transformation complete. Features are ready.\n")
    
    print("[STEP 3] Training the model and generating reports...")
    model, predictions, y_test = train_and_evaluate(df)
    
    print("\n=== Pipeline Complete ===")
    print("Check the 'reports' folder for the Feature Importance and Confusion Matrix charts.")

if __name__ == "__main__":
    main()
