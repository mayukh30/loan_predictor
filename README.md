# Loan Default Prediction System

An end-to-end Machine Learning web application designed to assess loan default risk and provide explainable AI insights for rejection reasons. 

## Table of Contents
- [Architecture & System Design](#architecture--system-design)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [Machine Learning Details](#machine-learning-details)

---

## Architecture & System Design

This project uses a modern decoupled architecture:
1. **Frontend (Vite + React)**: A dynamic, single-page application (SPA) that acts as the user dashboard. It communicates via REST API to the backend.
2. **Backend (FastAPI)**: A high-performance Python backend that serves the machine learning model, calculates dynamic statistics, and computes SHAP (SHapley Additive exPlanations) values on the fly.
3. **ML Engine**: Scikit-Learn models trained on historical data, saved as serialized objects, and loaded into memory by FastAPI for real-time inference.

### Data Flow
`User Input -> React UI -> REST POST /predict -> FastAPI -> Scaler & ML Model -> SHAP Explainer -> JSON Response -> React State -> UI Rendering`

---

## Tech Stack

**Frontend**
- React 18
- Vite (Build Tool)
- Recharts (Data Visualization)
- Lucide React (Icons)
- Vanilla CSS (Styling & Animations)

**Backend / ML**
- Python 3
- FastAPI (API Framework)
- Uvicorn (ASGI Server)
- Pandas & NumPy (Data processing)
- Scikit-Learn (Logistic Regression)
- XGBoost & TensorFlow (Alternative models available)
- SHAP (Explainable AI)

---

## Features

### 1. Dynamic Dashboard Stats
The system reads directly from the processed dataset to calculate total applications, approval rates, and average risk scores dynamically, avoiding hardcoded data. 

### 2. Live Risk Assessment
Users can enter applicant data (Income, Loan Amount, Credit History, etc.) to get an instant Risk Score (0-100) and probability of repayment.

### 3. Human-Readable AI Explainability (XAI)
Instead of returning opaque "black box" decisions or raw mathematical impacts, the system uses SHAP to analyze the model's decision logic and translates it into human-readable sentences (e.g., "Poor credit history", "The requested loan amount is excessively high relative to the income profile"). 

## How Rejection Is Determined

When a loan application is submitted, the backend runs the **Logistic Regression** model to compute a **risk score** (0‑100). The model predicts the probability of default; applications with a risk score above a configurable threshold (default 50) are marked **REJECTED**, otherwise **APPROVED**. In addition to the binary decision, we use **SHAP (SHapley Additive exPlanations)** to identify which input features contributed most to a rejection. These are presented in the UI as **Primary Rejection Reasons**.

## Input Parameters – Meaning & Impact

| Parameter | Description | Typical Impact on Prediction |
|---|---|---|
| **Gender** | 1 = Male, 0 = Female | Slight effect; female applicants may have a marginally higher risk in the training data. |
| **Married** | 1 = Married, 0 = Not Married | Married status often reduces perceived risk due to assumed financial stability. |
| **Dependents** | Number of dependents (0‑3+) | More dependents increase financial burden, raising the risk score. |
| **Education** | 1 = Graduate, 0 = Not Graduate | Graduates tend to have lower default risk; non‑graduates see a modest increase. |
| **Self_Employed** | 1 = Self‑Employed, 0 = Not Self‑Employed | Self‑employment may indicate irregular income, slightly increasing risk. |
| **ApplicantIncome** | Primary applicant's monthly income (USD) | Higher income lowers risk proportionally. |
| **CoapplicantIncome** | Co‑applicant's monthly income (USD) | Additional income reduces risk, especially when primary income is low. |
| **LoanAmount** | Requested loan amount (thousands of USD) | Larger loan amounts increase risk. |
| **Loan_Amount_Term** | Loan repayment period (days) | Longer terms spread risk; very long terms can raise risk. |
| **Credit_History** | 1 = Good, 0 = Bad | The most influential factor; a bad credit history sharply raises the risk score. |
| **Property_Area** | Urban / Semiurban / Rural | Property location correlates with economic conditions; urban areas often have slightly lower risk. |

The **SHAP explanations** displayed in the UI correspond to the features above, giving applicants clear, human‑readable reasons for any rejection (e.g., "Low credit history", "High loan amount relative to income").

---

### 4. Customer Segmentation
Visualizes the distribution of loan applicants across Urban, Semiurban, and Rural regions using dynamic pie charts.

---

## Setup & Installation

### Prerequisites
- Node.js (v18+)
- Python (3.9+)

### 1. Clone the repository
```bash
git clone <repository_url>
cd Loan_Prediction
```

### 2. Backend Setup
Install the required Python dependencies:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow shap fastapi uvicorn pydantic
```
Run the FastAPI server:
```bash
python src/backend/main.py
```
*The backend will run on http://localhost:8000*

### 3. Frontend Setup
Open a new terminal window, navigate to the frontend directory, and install dependencies:
```bash
cd src/frontend
npm install
```
Start the development server:
```bash
npm run dev
```
*The frontend will run on http://localhost:5173*

---

## Project Structure

```text
Loan_Prediction/
├── dataset/                    # Raw and cleaned CSV datasets
├── models/                     # Pickled/Saved ML Models & Scalers
├── src/
│   ├── ml/
│   │   ├── data_cleaning.py    # Script for imputing/encoding dataset
│   │   └── train_models.py     # Script to train LR, XGBoost, and DNN
│   ├── backend/
│   │   └── main.py             # FastAPI Application & SHAP logic
│   └── frontend/
│       ├── package.json        # Node dependencies
│       └── src/
│           ├── App.jsx         # Main React Dashboard
│           └── index.css       # Global Vanilla CSS styling
├── .gitignore                  # Standard ignores
└── README.md                   # Project documentation
```

---

## Machine Learning Details

The pipeline involves:
1. **Imputation**: Filling missing categorical values with the mode and numerical values with the median.
2. **Encoding**: Applying binary mapping and One-Hot Encoding to categorical columns.
3. **Scaling**: Standardizing numerical features using `StandardScaler`.
4. **Modeling**: Although XGBoost and Deep Neural Networks were trained, **Logistic Regression** was chosen for deployment due to its superior validation accuracy (~78.8%) and excellent interpretability via `LinearExplainer`.
