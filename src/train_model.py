from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Import your custom modules
from data_loader import load_master_data
from preprocessing import check_null_churn_data, preprocess_features

def main():
    print("Loading data...")
    df = load_master_data()
    
    print("Preprocessing data...")
    df = check_null_churn_data(df)
    df = preprocess_features(df)
    
    # Define Features (X) and Target (y)
    X = df.drop(columns=['churn_flag'])
    y = df['churn_flag']
    
    # Split the dataset
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train the Model
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make Predictions and Evaluate
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    print("\n--- Model Results ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))

if __name__ == "__main__":
    main()
