import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_and_evaluate(df):
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

    # --- Feature Importance ---
    print("\nGenerating Feature Importance Chart...")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, color='skyblue')
    plt.title('Feature Importance for SaaS Churn')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/feature_importance.png')
    print("Saved Feature Importance chart to reports/feature_importance.png")
    plt.close()
    
    # --- Confusion Matrix ---
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned', 'Churned'], 
                yticklabels=['Not Churned', 'Churned'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    print("Saved Confusion Matrix to reports/confusion_matrix.png")
    plt.close()
    
    # Generate probabilities for the full dataset (for the dashboard)
    full_probabilities = model.predict_proba(X)[:, 1]
    
    return model, predictions, y_test, full_probabilities
