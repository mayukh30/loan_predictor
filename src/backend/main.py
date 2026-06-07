import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loan_advisor import LoanAdviceVectorStore, build_advice_query, summarize_recommendations

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
advice_store = None

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
    advice_store = LoanAdviceVectorStore(os.path.join(models_dir, 'loan_advice_vector_store.pkl'))
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


def build_feature_context(application: dict) -> str:
    context_parts = []

    loan_amount = float(application.get("LoanAmount", 0) or 0)
    applicant_income = float(application.get("ApplicantIncome", 0) or 0)
    coapplicant_income = float(application.get("CoapplicantIncome", 0) or 0)
    credit_history = int(application.get("Credit_History", 0) or 0)
    self_employed = int(application.get("Self_Employed", 0) or 0)

    if loan_amount > 0 and applicant_income > 0 and loan_amount > applicant_income * 4:
        context_parts.append("reduce loan amount")

    if credit_history == 0:
        context_parts.append("improve credit history")

    if coapplicant_income <= 0:
        context_parts.append("add co-applicant")

    if self_employed == 1:
        context_parts.append("document self-employment income")

    return " ".join(context_parts)


def get_personalized_advice(application: dict, rejection_reasons: list[dict], approved: bool) -> list[dict]:
    if advice_store is None:
        return []

    advice_query = build_advice_query(application, rejection_reasons)
    feature_context = build_feature_context(application)
    combined_query = " ".join(part for part in [advice_query, feature_context] if part).strip()

    recommendations = advice_store.query(combined_query, top_k=4)

    if approved and recommendations:
        return recommendations[:2]

    if not recommendations:
        fallback_query = "loan approval improve chances reduce amount credit history income co-applicant"
        recommendations = advice_store.query(fallback_query, top_k=4)

    return recommendations


@app.get("/")
def read_root():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict(application: LoanApplication):
    # Convert input to dataframe
    application_payload = application.dict()
    input_data = pd.DataFrame([application_payload])
    
    # Ensure correct column order
    input_data = input_data[feature_names]
    
    # Apply log transformation to skewed features to match training
    skewed_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in skewed_cols:
        input_data[col] = np.log1p(input_data[col].astype(float))
        
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Predict (handle scikit-learn version mismatch by falling back to XGBoost)
    try:
        prob = float(lr_model.predict_proba(input_scaled)[0][1])
    except Exception:
        try:
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(os.path.join(models_dir, 'xgboost.json'))
            prob = float(xgb_model.predict_proba(input_scaled)[0][1])
        except Exception:
            prob = 0.0
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

    recommendations = get_personalized_advice(application_payload, rejection_reasons, bool(prediction == 1))
    return {
        "approved": bool(prediction == 1),
        "probability": float(prob),
        "risk_score": float(1.0 - prob) * 100,  # 0 to 100 scale
        "feature_importance": feature_importance,
        "rejection_reasons": rejection_reasons,
        "recommendations": recommendations,
        "suggestions": summarize_recommendations(recommendations),
    }


class AdviceQuery(BaseModel):
    query: str
    top_k: int = 4


@app.post("/advice")
def search_advice(request: AdviceQuery):
    if advice_store is None:
        raise HTTPException(status_code=500, detail="Advice store not loaded")

    results = advice_store.query(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "results": results,
        "summary": summarize_recommendations(results),
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
    
    skewed_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in skewed_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
            
    X_scaled = scaler.transform(X)
    probs = lr_model.predict_proba(X_scaled)[:, 1]
    avg_risk_score = float((1.0 - probs.mean()) * 100)
    
    # Segmentation
    # Calculate based on one-hot encoding
    urban = int(train_df['Property_Area_Urban'].sum()) if 'Property_Area_Urban' in train_df.columns else 0
    semiurban = int(train_df['Property_Area_Semiurban'].sum()) if 'Property_Area_Semiurban' in train_df.columns else 0
    rural = total_applications - urban - semiurban

    averages = {}
    if 'Loan_Status' in train_df.columns:
        averages = {
            "approved": {
                "ApplicantIncome": float(train_df[train_df['Loan_Status'] == 1]['ApplicantIncome'].mean()),
                "LoanAmount": float(train_df[train_df['Loan_Status'] == 1]['LoanAmount'].mean()),
            },
            "rejected": {
                "ApplicantIncome": float(train_df[train_df['Loan_Status'] == 0]['ApplicantIncome'].mean()),
                "LoanAmount": float(train_df[train_df['Loan_Status'] == 0]['LoanAmount'].mean()),
            }
        }
    
    return {
        "total_applications": total_applications,
        "approval_rate": approval_rate,
        "average_risk_score": avg_risk_score,
        "segmentation": {
            "Urban": urban,
            "Semiurban": semiurban,
            "Rural": rural
        },
        "averages": averages
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
