import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Loan Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
models_dir = os.path.join(base_dir, "models")
dataset_dir = os.path.join(base_dir, "dataset")

train_df = None

try:
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
        
    train_df = pd.read_csv(os.path.join(dataset_dir, "cleaned_train.csv"))
    # We will use LinearExplainer for Logistic Regression
    explainer = shap.LinearExplainer(lr_model, scaler.transform(train_df.drop('Loan_Status', axis=1)))
except Exception as e:
    print(f"Error loading models or dataset: {e}")

class LoanApplication(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int
    Property_Area_Semiurban: int
    Property_Area_Urban: int

# Mapping for Human Readable Explanations
def get_human_readable_reason(feature, value, shap_value):
    if feature == "Credit_History":
        if value == 0:
            return "Poor credit history (does not meet minimum guidelines)."
        else:
            return "Credit history factors."
    elif feature == "ApplicantIncome":
        return "Applicant's income is insufficient compared to the requested loan amount."
    elif feature == "CoapplicantIncome":
        return "Co-applicant's income is too low to support the application."
    elif feature == "LoanAmount":
        return "The requested loan amount is excessively high relative to the income profile."
    elif feature == "Loan_Amount_Term":
        return "The requested loan term length is unfavorable given other financial indicators."
    elif feature == "Dependents":
        return "The number of dependents impacts the debt-to-income affordability."
    elif feature == "Education":
        if value == 0:
            return "Lack of graduate education marginally impacted the risk profile."
    elif feature == "Self_Employed":
        if value == 1:
            return "Self-employment status introduces higher income volatility risk."
    elif feature == "Property_Area_Semiurban" or feature == "Property_Area_Urban":
        return "The property location falls into a higher risk category based on historical data."
    elif feature == "Gender":
        return "Demographic statistical factors."
    elif feature == "Married":
        return "Marital status statistical factors."
    
    return f"The {feature} does not align with our approval criteria."


@app.get("/")
def read_root():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict(application: LoanApplication):
    # Convert input to dataframe
    input_data = pd.DataFrame([application.dict()])
    
    # Ensure correct column order
    input_data = input_data[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prob = lr_model.predict_proba(input_scaled)[0][1]
    prediction = int(prob > 0.5)
    
    # Explain prediction using SHAP
    shap_values = explainer.shap_values(input_scaled)[0]
    
    # Extract top reasons
    feature_importance = [
        {"feature": feature_names[i], "value": float(input_data.iloc[0, i]), "shap_value": float(shap_values[i])}
        for i in range(len(feature_names))
    ]
    
    # Sort by absolute SHAP value to find most impactful features
    feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    
    rejection_reasons = []
    suggestions = []
    if prediction == 0:
        # Filter for features that pushed the prediction towards 0 (negative SHAP)
        negative_features = [f for f in feature_importance if f["shap_value"] < 0]
        
        # Get top 3 reasons
        for f in negative_features[:3]:
            human_reason = get_human_readable_reason(f["feature"], f["value"], f["shap_value"])
            rejection_reasons.append({
                "feature": f["feature"],
                "explanation": human_reason
            })
            # Map feature to a user‑friendly suggestion
            suggestion_map = {
                "Credit_History": "Consider improving your credit history or adding a co‑applicant with a good credit record.",
                "LoanAmount": "Reduce the requested loan amount or provide additional collateral.",
                "ApplicantIncome": "Increase documented income or add a higher‑earning co‑applicant.",
                "CoapplicantIncome": "Add a co‑applicant with a higher income.",
                "Loan_Amount_Term": "Choose a shorter loan term to lower the risk perception.",
                "Dependents": "Reduce the number of dependents or increase household income.",
                "Education": "Obtain a graduate degree or provide certifications to strengthen the profile.",
                "Self_Employed": "Provide audited financial statements or add a salaried co‑applicant.",
                "Married": "If possible, add a spouse as a co‑applicant.",
                "Gender": "Gender has a minor effect; focus on other stronger factors.",
                "Property_Area_Semiurban": "Consider applying for a property in an urban area where risk is lower.",
                "Property_Area_Urban": "Property location is already urban; focus on improving other factors."
            }
            suggestion = suggestion_map.get(f["feature"], None)
            if suggestion:
                suggestions.append(suggestion)
    return {
        "approved": bool(prediction == 1),
        "probability": float(prob),
        "risk_score": float(1.0 - prob) * 100,  # 0 to 100 scale
        "feature_importance": feature_importance,
        "rejection_reasons": rejection_reasons,
        "suggestions": suggestions
    }

@app.get("/stats")
def get_stats():
    if train_df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
        
    total_applications = len(train_df)
    
    # Approval rate (Loan_Status == 1)
    if 'Loan_Status' in train_df.columns:
        approval_rate = float(train_df['Loan_Status'].mean())
    else:
        approval_rate = 0.0
        
    # Calculate average risk score by running model over dataset
    # For performance, just sample 200 rows if dataset is large, but here it's small (614)
    X = train_df.drop('Loan_Status', axis=1, errors='ignore')
    X = X[feature_names] # Ensure order
    X_scaled = scaler.transform(X)
    probs = lr_model.predict_proba(X_scaled)[:, 1]
    avg_risk_score = float((1.0 - probs.mean()) * 100)
    
    # Segmentation
    # Calculate based on one-hot encoding
    urban = int(train_df['Property_Area_Urban'].sum()) if 'Property_Area_Urban' in train_df.columns else 0
    semiurban = int(train_df['Property_Area_Semiurban'].sum()) if 'Property_Area_Semiurban' in train_df.columns else 0
    rural = total_applications - urban - semiurban
    
    return {
        "total_applications": total_applications,
        "approval_rate": approval_rate,
        "average_risk_score": avg_risk_score,
        "segmentation": {
            "Urban": urban,
            "Semiurban": semiurban,
            "Rural": rural
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
