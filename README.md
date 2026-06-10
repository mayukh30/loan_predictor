---
title: IntelliLoan API
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# IntelliLoan — AI-Powered Loan Default Prediction API

This Space hosts the FastAPI backend for the IntelliLoan system, which assesses loan default risk using Explainable AI (SHAP) and delivers personalized financial advice via Pinecone Vector Database.

## API Endpoints

| Method | Endpoint   | Description |
|--------|-----------|-------------|
| `GET`  | `/`       | Health check |
| `POST` | `/predict`| Submit loan application, returns risk score + SHAP reasons + Pinecone advice |
| `GET`  | `/stats`  | Dynamic dashboard statistics computed from the training dataset |

For full project documentation, see [PROJECT_DOCS.md](PROJECT_DOCS.md).
