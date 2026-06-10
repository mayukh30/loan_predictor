# IntelliLoan — AI-Powered Loan Default Prediction System

An end-to-end Machine Learning web application that assesses loan default risk using **Explainable AI (SHAP)** and delivers personalized financial advice via **Pinecone Vector Database** with semantic search.

> **No LLMs. No hallucinations. 100% deterministic, model-faithful explanations.**

---

## Table of Contents
- [Architecture & System Design](#architecture--system-design)
- [Tech Stack](#tech-stack)
- [CI/CD & Deployment Workflow](#cicd--deployment-workflow)
- [Features](#features)
- [Input Parameters](#input-parameters)
- [How SHAP Explainability Works](#how-shap-explainability-works)
- [How Pinecone Vector DB Advice Works](#how-pinecone-vector-db-advice-works)
- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [ML Pipeline Details](#ml-pipeline-details)

---

## Architecture & System Design

This project uses a **fully decoupled, three-tier architecture**:

```
┌────────────────────┐     REST API      ┌─────────────────────┐     Semantic     ┌──────────────┐
│   React Frontend   │ ◄──────────────► │   FastAPI Backend    │ ◄─────────────► │   Pinecone   │
│   (Vite + SPA)     │   JSON over HTTP  │   (Python + ML)      │    Search       │  Vector DB   │
└────────────────────┘                   └─────────────────────┘                  └──────────────┘
                                                  │
                                      ┌───────────┼───────────┐
                                      ▼           ▼           ▼
                                ┌──────────┐ ┌─────────┐ ┌─────────┐
                                │  Scaler  │ │ LR Model│ │  SHAP   │
                                │ (pkl)    │ │ (pkl)   │ │Explainer│
                                └──────────┘ └─────────┘ └─────────┘
```

### Data Flow
```
User Input → React UI → POST /predict → FastAPI → StandardScaler → Logistic Regression
→ SHAP LinearExplainer → Human-Readable Reasons → sentence-transformers Embedding
→ Pinecone Semantic Search → Top-K Financial Advice → JSON Response → React Rendering
```

### API Endpoints
| Method | Endpoint   | Description |
|--------|-----------|-------------|
| `GET`  | `/`       | Health check |
| `POST` | `/predict`| Submit loan application, returns risk score + SHAP reasons + Pinecone advice |
| `GET`  | `/stats`  | Dynamic dashboard statistics computed from the training dataset |

---

## CI/CD & Deployment Workflow

The project utilizes a fully automated CI/CD pipeline leveraging **GitHub Actions**, **Vercel**, and **Hugging Face Spaces**.

### Deployment Architecture
```
[Developer Push] ──► [GitHub Repository]
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
   [GitHub Actions CI]                 [Vercel CI/CD]
   - Runs ESLint                       - Pulls latest main branch
   - Verifies build                    - Injects VITE_API_URL
            │                          - Builds React app
            ▼                          - Deploys to Vercel Edge Network
   [GitHub Actions CD]                          │
   - Force pushes to HF Space                   ▼
            │                            [Live Frontend]
            ▼
 [Hugging Face Spaces]
   - Rebuilds Docker image (Python 3.12)
   - Installs requirements.txt
   - Exposes FastAPI on port 7860
            │
            ▼
     [Live Backend]
```

### 1. Frontend Deployment (Vercel)
- The frontend is hosted on **Vercel**.
- **Continuous Deployment**: Any push to the `main` branch automatically triggers a Vercel build.
- **Environment Variables**: Vercel injects `VITE_API_URL` during the build step, ensuring the React app natively points to the production Hugging Face backend instead of `localhost`.

### 2. Backend Deployment (Hugging Face Spaces)
- The backend is hosted as a **Docker Space on Hugging Face**.
- **Continuous Deployment**: A custom GitHub Action (`.github/workflows/huggingface-sync.yml`) is triggered on every push to `main`.
- This action forces a sync from the GitHub repository to the Hugging Face Git remote.
- Hugging Face detects the update, reads the `Dockerfile`, and automatically spins up a new container using `python:3.12-slim` to match the exact environment the Scikit-learn model was trained in.

### 3. Continuous Integration (CI)
- A GitHub Action (`frontend-ci.yml`) runs on every push and pull request.
- It enforces code quality by running `npm run lint`.
- It validates build integrity by running `npm run build` using Node 20.x, ensuring no broken code reaches production.

---

## Tech Stack

### Frontend
| Technology | Purpose |
|-----------|---------|
| **React 19** | Component-based UI framework |
| **Vite 8** | Lightning-fast build tool & dev server |
| **Recharts** | Data visualization (Pie charts, Bar charts) |
| **Lucide React** | Modern icon library |
| **Axios** | HTTP client for API communication |
| **Vanilla CSS** | Custom styling with CSS variables, glassmorphism, and animations |

### Backend & ML
| Technology | Purpose |
|-----------|---------|
| **Python 3.9+** | Core language |
| **FastAPI** | High-performance async API framework |
| **Uvicorn** | ASGI server |
| **Pandas & NumPy** | Data processing & numerical computation |
| **Scikit-Learn** | Logistic Regression (deployed model), StandardScaler |
| **XGBoost** | Alternative model (trained but not deployed) |
| **TensorFlow/Keras** | Deep Neural Network (trained but not deployed) |
| **SHAP** | SHapley Additive exPlanations for model interpretability |
| **Pinecone** | Managed vector database for semantic advice retrieval |
| **sentence-transformers** | `all-MiniLM-L6-v2` for text embedding (384-dim vectors) |
| **python-dotenv** | Environment variable management |

---

## Features

### 1. Dynamic Dashboard Statistics
The system reads directly from the processed training dataset (20,000 samples) to compute:
- Total applications processed
- Overall approval rate
- Average risk score
- Customer segmentation by home ownership (Rent / Own / Mortgage)
- Comparative averages for approved vs. rejected applicants

### 2. Live Risk Assessment
Users enter applicant data and receive an instant:
- **Risk Score** (0–100): Higher = more likely to default
- **Repayment Probability** (0–100%): Model confidence in repayment
- **Approval Decision**: Based on a configurable threshold (default: 0.65 probability)

### 3. Explainable AI (XAI) with SHAP
When a loan is rejected, the system identifies the top 3 features that pushed the prediction toward rejection using SHAP values, translated into human-readable sentences like:
- *"Applicant's income level"*
- *"Requested loan amount"*
- *"Length of continuous employment"*

### 4. Pinecone Vector DB Semantic Advice
Rejection reasons are embedded into 384-dimensional vectors using `sentence-transformers` and semantically matched against a curated catalog of 25 financial tips stored in Pinecone. This delivers personalized, actionable advice without any LLM, completely deterministic and hallucination-free.

### 5. Historical Comparison Charts
Bar charts comparing the applicant's Income and Loan Amount against the average values of historically approved and rejected applicants.

### 6. Customer Segmentation
Interactive donut chart showing the distribution of applicants by home ownership status.

---

## Input Parameters

| Parameter | Field | Description | Impact on Risk |
|-----------|-------|-------------|----------------|
| **Age** | `person_age` | Applicant's age (20–70) | Younger applicants may have shorter credit/employment histories |
| **Income** | `person_income` | Annual income in USD | Higher income → significantly lower risk |
| **Employment Length** | `person_emp_length` | Years of continuous employment | Longer employment → lower risk |
| **Home Ownership** | `person_home_ownership` | RENT / OWN / MORTGAGE | Renting increases perceived risk slightly |
| **Loan Intent** | `loan_intent` | Purpose of the loan | VENTURE/PERSONAL carry higher risk |
| **Loan Amount** | `loan_amnt` | Requested amount in USD ($1K–$35K) | Higher amounts → higher risk |
| **Interest Rate** | `loan_int_rate` | Loan interest rate (5%–20%) | Higher rates → higher risk |
| **Loan Term** | `loan_term_days` | Repayment period (360–3600 days) | Longer terms → slightly higher risk |

Available **Loan Intent** categories: `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT`

Available **Home Ownership** categories: `RENT`, `OWN`, `MORTGAGE`

---

## How SHAP Explainability Works

SHAP (SHapley Additive exPlanations) computes the **exact mathematical contribution** of each input feature to the model's final prediction — no approximation, no LLM interpretation.

1. The `LinearExplainer` is initialized with the trained Logistic Regression model and the scaled training data as background
2. For each prediction, SHAP calculates a value per feature (positive = pushes toward approval, negative = pushes toward rejection)
3. Features with the largest negative SHAP values are extracted as rejection reasons
4. These feature names are mapped to human-readable templates via a deterministic dictionary lookup

**This guarantees explanations are 100% faithful to the model's actual decision logic.**

---

## How Pinecone Vector DB Advice Works

Instead of using LangChain or a generative LLM for advice, this system uses a **Retrieval-Augmented Generation (RAG)-style** approach with Pinecone:

1. **Offline**: A curated catalog of 25 financial advice entries (in `advice_catalog.py`) is embedded using `all-MiniLM-L6-v2` (384 dimensions) and upserted into a Pinecone index
2. **Online**: When a loan is rejected, the SHAP rejection reasons are concatenated into a query string
3. The query is embedded into the same 384-dimensional vector space
4. Pinecone performs a **cosine similarity search** and returns the top-K most semantically relevant advice
5. The advice is returned to the frontend as structured JSON

**Benefits over LLM-based advice:**
- ⚡ ~50ms latency (vs. 2–5s for LLM generation)
- 🎯 100% deterministic — same input always yields same advice
- 🛡️ Zero hallucination risk — all advice is human-curated
- 💰 No per-token API costs

---

## Setup & Installation

### Prerequisites
- **Node.js** v18+
- **Python** 3.9+
- A **Pinecone** account (free tier works) with an index named `loan-advice-index` (384 dimensions, cosine metric)

### 1. Clone the Repository
```bash
git clone <repository_url>
cd Loan_Prediction
```

### 2. Environment Variables
Create a `.env` file in the project root:
```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=loan-advice-index
```

### 3. Backend Setup
```bash
# Install Python dependencies
pip install pandas numpy scikit-learn xgboost tensorflow shap fastapi uvicorn pydantic python-dotenv pinecone sentence-transformers

# Step 1: Generate synthetic dataset (20,000 samples)
python src/ml/generate_dataset.py

# Step 2: Clean and preprocess the dataset
python src/ml/data_cleaning.py

# Step 3: Train all models (Logistic Regression, XGBoost, DNN)
python src/ml/train_models.py

# Step 4: Populate Pinecone with advice embeddings
python src/backend/upsert_advice.py

# Step 5: Start the FastAPI server
python src/backend/main.py
```
*The backend runs on http://localhost:8002*

### 4. Frontend Setup
```bash
cd src/frontend
npm install
npm run dev
```
*The frontend runs on http://localhost:5173*

---

## Project Structure

```
Loan_Prediction/
├── .env                                # Pinecone API key & index name (git-ignored)
├── .gitignore
├── README.md
├── dataset/                            # Raw & cleaned CSV datasets (git-ignored)
│   ├── credit_risk_dataset.csv         # Synthetic dataset (20K rows)
│   └── cleaned_train.csv               # Preprocessed & one-hot encoded
├── models/                             # Serialized ML models (git-ignored)
│   ├── scaler.pkl                      # StandardScaler
│   ├── logistic_regression.pkl         # Deployed model
│   ├── xgboost.json                    # Alternative model
│   ├── dnn_model.h5                    # Alternative model
│   └── feature_names.pkl              # Column order for inference
├── src/
│   ├── ml/                             # Machine Learning pipeline
│   │   ├── generate_dataset.py         # Synthetic data generation (NumPy)
│   │   ├── fetch_dataset.py            # Fetch from external sources
│   │   ├── data_cleaning.py            # Imputation, encoding, cleaning
│   │   └── train_models.py             # Train LR, XGBoost, DNN
│   ├── backend/                        # FastAPI application
│   │   ├── main.py                     # API endpoints, SHAP logic, prediction
│   │   ├── loan_advisor.py             # PineconeAdviceVectorStore class
│   │   ├── advice_catalog.py           # Curated financial advice (25 entries)
│   │   └── upsert_advice.py            # One-time script to populate Pinecone
│   └── frontend/                       # React SPA
│       ├── package.json
│       └── src/
│           ├── main.jsx                # React entry point
│           ├── App.jsx                 # Main dashboard component
│           ├── App.css                 # Component-specific styles
│           └── index.css               # Global styles & CSS variables
```

---

## ML Pipeline Details

### 1. Data Generation (`generate_dataset.py`)
- Generates 20,000 synthetic credit risk samples using NumPy
- Features include realistic distributions (log-normal income, categorical intent/ownership)
- Risk score is computed from a logical formula involving DTI, interest rate, income, employment, and ownership
- Sigmoid function converts risk scores to probabilities; ~20% default rate via percentile thresholding

### 2. Data Cleaning (`data_cleaning.py`)
- Fills missing categorical values with **mode**, numerical with **median**
- Applies **One-Hot Encoding** (`pd.get_dummies`) to `person_home_ownership` and `loan_intent` with `drop_first=True`
- Inverts `loan_status` (1=Default → 0=Rejected) to match backend expectation (1=Approved)

### 3. Feature Engineering (`train_models.py`)
- **Log transformation** (`np.log1p`) applied to `person_income` and `loan_amnt` to handle right-skewed distributions
- **StandardScaler** normalizes all features to zero mean and unit variance

### 4. Model Training
Three models are trained and compared:

| Model | Framework | Accuracy | Deployed? |
|-------|-----------|----------|-----------|
| **Logistic Regression** | Scikit-Learn | ~78.8% | ✅ Yes |
| XGBoost | XGBoost | ~80%+ | ❌ No |
| Deep Neural Network | TensorFlow/Keras | ~79% | ❌ No |

**Logistic Regression was chosen for deployment** because:
- It provides the best interpretability through SHAP's `LinearExplainer`
- `LinearExplainer` computes exact SHAP values (no sampling/approximation needed)
- The marginal accuracy gain from XGBoost/DNN doesn't justify the loss of explainability

### 5. Inference
- Input is one-hot encoded to match training schema
- Log transformation and scaling are applied identically to training
- `predict_proba` gives the probability of repayment
- Risk score = `(1 - probability) × 100`
- Threshold of 0.65 determines approval/rejection
