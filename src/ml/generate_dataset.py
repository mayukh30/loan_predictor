import os
import pandas as pd
import numpy as np

def generate_credit_dataset(num_rows=20000):
    np.random.seed(42)
    
    # Features
    person_age = np.random.randint(20, 70, num_rows)
    person_income = np.random.lognormal(mean=11.0, sigma=0.6, size=num_rows).astype(int) # Mean around 60k
    person_emp_length = np.where(person_age < 25, np.random.randint(0, 5, num_rows), np.random.randint(0, 30, num_rows))
    
    home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], num_rows, p=[0.4, 0.2, 0.4])
    loan_intent = np.random.choice(['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'], num_rows)
    
    loan_amnt = np.random.randint(1000, 35000, num_rows)
    loan_int_rate = np.random.uniform(5.0, 20.0, num_rows)
    loan_term_days = np.random.choice([360, 720, 1080, 1800, 3600], num_rows)
    
    # Logical Constraints & Probabilities for Risk
    # Higher income -> lower risk
    # Higher loan amount -> higher risk
    # OWN/MORTGAGE -> lower risk
    # Longer emp length -> lower risk
    # Longer term -> slightly higher risk due to extended exposure
    
    dti = loan_amnt / person_income
    
    # Base risk score (z-score like)
    risk_score = (dti * 5) + (loan_int_rate * 0.1) - (person_income / 100000) - (person_emp_length * 0.05) + (loan_term_days / 1800)
    risk_score += np.where(home_ownership == 'RENT', 0.5, -0.5)
    
    # Add some noise
    risk_score += np.random.normal(0, 1, num_rows)
    
    # Convert to probability using sigmoid
    probability = 1 / (1 + np.exp(-risk_score))
    
    # 1 = Default (Rejected), 0 = Paid (Approved)
    # Let's adjust the threshold to get roughly 20% default rate
    threshold = np.percentile(probability, 80)
    loan_status = (probability > threshold).astype(int)
    
    df = pd.DataFrame({
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_amnt': loan_amnt,
        'loan_int_rate': np.round(loan_int_rate, 2),
        'loan_term_days': loan_term_days,
        'loan_status': loan_status
    })
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, "credit_risk_dataset.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset with {num_rows} rows saved to {output_path}")
    print("Default rate:", df['loan_status'].mean())
    print(df.head())

if __name__ == "__main__":
    generate_credit_dataset()
