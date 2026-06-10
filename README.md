# IntelliLoan — AI-Powered Loan Default Prediction System

IntelliLoan is an end-to-end Machine Learning web application that assesses loan default risk using Explainable AI (SHAP) and delivers personalized financial advice via a Pinecone Vector Database with semantic search.

## What It Does
IntelliLoan automates the loan application review process by calculating a risk score for each applicant based on their financial history and profile. If a loan application is rejected, the system doesn't just return a "No". Instead, it:
1. Identifies the exact reasons for rejection using mathematically sound Explainable AI (SHAP).
2. Maps these reasons to a semantic vector space.
3. Retrieves personalized, actionable financial advice from a Pinecone Vector DB to help the applicant improve their profile for future applications.

All of this happens in milliseconds without the use of generative LLMs, ensuring 100% deterministic and hallucination-free guidance.

## Features
- **Live Risk Assessment**: Instant prediction of repayment probability and risk scores based on a threshold.
- **Explainable AI (XAI)**: Understandable, human-readable rejection reasons powered by SHAP.
- **Actionable Financial Advice**: Deterministic RAG-style semantic search using Pinecone to fetch tailored financial tips.
- **Dynamic Dashboard Statistics**: Live updates on overall approval rates, average risk scores, and customer segmentation.
- **Historical Comparisons**: Interactive charts comparing applicant data (e.g., income, loan amount) against historical averages.

## Tech Stack

### Frontend
- ⚛️ **React 19**: Component-based UI framework
- ⚡ **Vite 8**: Lightning-fast build tool & dev server
- 📊 **Recharts**: Data visualization
- 🎨 **Lucide React & Vanilla CSS**: Modern icons and custom styling (glassmorphism, animations)
- 🌐 **Axios**: HTTP client for API communication

### Backend & ML
- 🐍 **Python 3.9+**: Core backend language
- 🚀 **FastAPI & Uvicorn**: High-performance async REST API and ASGI server
- 🐼 **Pandas & NumPy**: Data processing and numerical computation
- 🧠 **Scikit-Learn**: Deployed Logistic Regression model
- 🌳 **XGBoost & TensorFlow/Keras**: Alternative trained models
- 🔍 **SHAP**: Model interpretability
- 🗄️ **Pinecone**: Managed vector database for semantic search
- 📝 **sentence-transformers**: Local text embeddings (`all-MiniLM-L6-v2`)

## Installation & Setup

### Prerequisites
- **Node.js** v18+
- **Python** 3.9+
- A **Pinecone** account with an index named `loan-advice-index` (384 dimensions, cosine metric)

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
# Install dependencies
pip install pandas numpy scikit-learn xgboost tensorflow shap fastapi uvicorn pydantic python-dotenv pinecone sentence-transformers

# Run the data pipeline and populate Pinecone
python src/ml/generate_dataset.py
python src/ml/data_cleaning.py
python src/ml/train_models.py
python src/backend/upsert_advice.py

# Start the server
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
