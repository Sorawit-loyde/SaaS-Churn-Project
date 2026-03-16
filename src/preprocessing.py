import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_churn_data(df):
    """
    Cleans the dataframe and prepares it for the model.
    """
    # 1. Fill missing values (Example: fill tenure with 0)
    df['tenure'] = df['tenure'].fillna(0)
    
    # 2. Encode categorical columns (e.g., Churn: 'Yes' -> 1, 'No' -> 0)
    le = LabelEncoder()
    if 'Churn' in df.columns:
        df['Churn'] = le.fit_transform(df['Churn'])
        
    return df

if __name__ == "__main__":
    # Test your function locally
    # df = pd.read_csv('data/processed/master_customer_data.csv')
    # clean_df = clean_churn_data(df)
    # print(clean_df.head())
    print("Preprocessing module ready.")