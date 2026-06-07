import os
import pandas as pd

def clean_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # In the synthetic dataset:
    # 1 = Default (Rejected)
    # 0 = Paid (Approved)
    # The backend expects 1 = Approved, 0 = Rejected
    if 'loan_status' in df.columns:
        df['Loan_Status'] = df['loan_status'].apply(lambda x: 0 if x == 1 else 1)
        df = df.drop('loan_status', axis=1)
        
    # Handle Missing Values (just in case)
    # Categorical
    for col in ['person_home_ownership', 'loan_intent']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Numerical
    num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_term_days']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # One-Hot Encoding for categorical variables
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)
    
    # Convert booleans to int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    input_path = os.path.join(dataset_dir, "credit_risk_dataset.csv")
    cleaned_path = os.path.join(dataset_dir, "cleaned_train.csv")
    
    if os.path.exists(input_path):
        clean_data(input_path, cleaned_path)
    else:
        print(f"Data not found at {input_path}")
