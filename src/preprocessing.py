import pandas as pd
from sklearn.preprocessing import LabelEncoder

def check_null_churn_data(df):
    """
    check the dataframe for null values.
    """
    if df.isnull().values.any():
        print("Missing values detected:\n", df.isnull().sum())
        return df
    else:
        print("No missing values detected.")
        return df    
def preprocess_features(df):
    """
    Encode categorical features and prepare for machine learning.
    """
    # 1. Drop identifier column
    df = df.drop(columns=['account_id'])
    
    # 2. Encode categorical text features
    le = LabelEncoder()
    categorical_cols = ['industry', 'country', 'plan_tier']
    for col in categorical_cols:
         df[col] = le.fit_transform(df[col])
         
    # 3. Convert booleans to 1/0
    df['is_trial'] = df['is_trial'].astype(int)
    df['churn_flag'] = df['churn_flag'].astype(int)
    
    return df
if __name__ == "__main__":
    df = pd.read_csv('data/processed/master_customer_data.csv')
    check_null_churn_data(df)
    df = preprocess_features(df)
    print(df.head())
    print("Preprocessing module ready.")