import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loan_advisor import PineconeAdviceVectorStore

app = FastAPI(title="Loan Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_dir = os.path.join(base_dir, "models")
dataset_dir = os.path.join(base_dir, "dataset")

train_df = None
advice_store = None
explainer = None
scaler = None
lr_model = None
feature_names = []

try:
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
        
    train_df = pd.read_csv(os.path.join(dataset_dir, "cleaned_train.csv"))
    explainer = shap.LinearExplainer(lr_model, scaler.transform(train_df.drop('Loan_Status', axis=1, errors='ignore')))
    advice_store = PineconeAdviceVectorStore()
except Exception as e:
    print(f"Error loading models or dataset: {e}")

class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: int
    loan_intent: str
    loan_amnt: float
    loan_int_rate: float
    loan_term_days: int

def get_human_readable_reason(feature, value, shap_value):
    mapping = {
        "person_age": "Applicant's age",
        "person_income": "Applicant's income level",
        "person_emp_length": "Length of continuous employment",
        "loan_amnt": "Requested loan amount",
        "loan_int_rate": "Requested interest rate",
        "loan_term_days": "Requested loan term length",
        "person_home_ownership_OWN": "Home ownership status (Owned)",
        "person_home_ownership_RENT": "Home ownership status (Renting)",
        "loan_intent_EDUCATION": "Loan purpose (Education)",
        "loan_intent_HOMEIMPROVEMENT": "Loan purpose (Home Improvement)",
        "loan_intent_MEDICAL": "Loan purpose (Medical)",
        "loan_intent_PERSONAL": "Loan purpose (Personal)",
        "loan_intent_VENTURE": "Loan purpose (Business/Venture)"
    }
    return mapping.get(feature, f"The {feature} does not align with our approval criteria.")

def get_personalized_advice(rejection_reasons: list[dict], approved: bool) -> list[dict]:
    if advice_store is None:
        return []
        
    if approved:
        # User requested no advice if accepted
        return []
        
    if not rejection_reasons:
        return advice_store.get_advice("general loan approval improvement tips", top_k=2)
        
    # Combine the top reasons into a query for Pinecone
    query = " ".join([r["explanation"] for r in rejection_reasons])
    return advice_store.get_advice(query, top_k=3)

@app.get("/")
def read_root():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict(application: LoanApplication):
    app_dict = application.dict()
    
    # Construct a row that exactly matches feature_names
    row = {}
    for f in feature_names:
        row[f] = 0 # Default for one-hot encoded columns
        
    # Fill in numerical features
    row['person_age'] = app_dict['person_age']
    row['person_income'] = app_dict['person_income']
    row['person_emp_length'] = app_dict['person_emp_length']
    row['loan_amnt'] = app_dict['loan_amnt']
    row['loan_int_rate'] = app_dict['loan_int_rate']
    row['loan_term_days'] = app_dict['loan_term_days']
    
    # Fill in one-hot encoded features
    home_ownership_key = f"person_home_ownership_{app_dict['person_home_ownership']}"
    if home_ownership_key in feature_names:
        row[home_ownership_key] = 1
        
    intent_key = f"loan_intent_{app_dict['loan_intent']}"
    if intent_key in feature_names:
        row[intent_key] = 1
        
    input_data = pd.DataFrame([row])
    
    # Apply log transformation to skewed features to match training
    skewed_cols = ['person_income', 'loan_amnt']
    for col in skewed_cols:
        input_data[col] = np.log1p(input_data[col].astype(float))
        
    input_data = input_data[feature_names] # Explicitly enforce column order to prevent scaler ValueError
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prob_approved = float(lr_model.predict_proba(input_scaled)[0][1])
    
    # We classify as APPROVED if prob_approved >= threshold
    threshold = 0.65
    approved = bool(prob_approved >= threshold)
    
    # For the frontend risk score (0-100), higher should mean more risk.
    risk_score = (1.0 - prob_approved) * 100.0
    
    # Explain prediction using SHAP
    shap_values = explainer.shap_values(input_scaled)[0]
    
    feature_importance = [
        {"feature": feature_names[i], "value": float(input_data.iloc[0, i]), "shap_value": float(shap_values[i])}
        for i in range(len(feature_names))
    ]
    feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    
    rejection_reasons = []
    if not approved:
        negative_features = [f for f in feature_importance if f["shap_value"] < 0]
        for f in negative_features[:3]:
            human_reason = get_human_readable_reason(f["feature"], f["value"], f["shap_value"])
            rejection_reasons.append({
                "feature": f["feature"],
                "explanation": human_reason
            })

    recommendations = get_personalized_advice(rejection_reasons, approved)
    
    # Summarize suggestions to strings for the UI
    suggestions = [r["advice"] for r in recommendations]
    
    return {
        "approved": approved,
        "probability": float(prob_approved),
        "risk_score": risk_score,
        "feature_importance": feature_importance,
        "rejection_reasons": rejection_reasons,
        "recommendations": recommendations,
        "suggestions": suggestions,
    }

@app.get("/stats")
def get_stats():
    if train_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
        
    total_applications = len(train_df)
    approval_rate = float(train_df['Loan_Status'].mean()) if 'Loan_Status' in train_df.columns else 0.0
        
    X = train_df.drop('Loan_Status', axis=1, errors='ignore')
    X = X[feature_names] # Ensure order
    
    skewed_cols = ['person_income', 'loan_amnt']
    for col in skewed_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
            
    X_scaled = scaler.transform(X)
    probs = lr_model.predict_proba(X_scaled)[:, 1]
    avg_risk_score = float((1.0 - probs.mean()) * 100)
    
    # Segmentation
    rent = int(train_df['person_home_ownership_RENT'].sum()) if 'person_home_ownership_RENT' in train_df.columns else 0
    own = int(train_df['person_home_ownership_OWN'].sum()) if 'person_home_ownership_OWN' in train_df.columns else 0
    mortgage = total_applications - rent - own

    averages = {}
    if 'Loan_Status' in train_df.columns:
        averages = {
            "approved": {
                "Income": float(train_df[train_df['Loan_Status'] == 1]['person_income'].mean()),
                "LoanAmount": float(train_df[train_df['Loan_Status'] == 1]['loan_amnt'].mean()),
            },
            "rejected": {
                "Income": float(train_df[train_df['Loan_Status'] == 0]['person_income'].mean()),
                "LoanAmount": float(train_df[train_df['Loan_Status'] == 0]['loan_amnt'].mean()),
            }
        }
    
    return {
        "total_applications": total_applications,
        "approval_rate": approval_rate,
        "average_risk_score": avg_risk_score,
        "segmentation": {
            "Rent": rent,
            "Own": own,
            "Mortgage": mortgage
        },
        "averages": averages
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
