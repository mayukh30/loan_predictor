import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def clean_data(input_path, output_path, is_train=True):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Drop Loan_ID as it's not useful for prediction
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
        
    # Handle Missing Values
    # Categorical features -> mode
    cat_cols_with_na = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
    for col in cat_cols_with_na:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Numerical features -> median
    num_cols_with_na = ['LoanAmount']
    for col in num_cols_with_na:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # Clean Dependents column ('3+' to '3')
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)
        
    # Encoding Categorical Variables
    # Binary variables
    binary_mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in binary_mapping.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            
    # Property_Area (One-Hot Encoding)
    if 'Property_Area' in df.columns:
        df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
        # Convert boolean columns to int
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
                
    # Target Variable
    if is_train and 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
        
    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    train_path = os.path.join(dataset_dir, "train_u6lujuX_CVtuZ9i.csv")
    test_path = os.path.join(dataset_dir, "test_Y3wMUE5_7gLdaTN.csv")
    
    cleaned_train_path = os.path.join(dataset_dir, "cleaned_train.csv")
    cleaned_test_path = os.path.join(dataset_dir, "cleaned_test.csv")
    
    print("--- Cleaning Training Data ---")
    if os.path.exists(train_path):
        clean_data(train_path, cleaned_train_path, is_train=True)
    else:
        print(f"Training data not found at {train_path}")
        
    print("\n--- Cleaning Test Data ---")
    if os.path.exists(test_path):
        clean_data(test_path, cleaned_test_path, is_train=False)
    else:
        print(f"Test data not found at {test_path}")
