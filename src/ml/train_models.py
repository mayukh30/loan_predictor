import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def train_and_save_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    data_path = os.path.join(dataset_dir, "cleaned_train.csv")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print("--- 1. Logistic Regression ---")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_val_scaled)
    print(f"Accuracy: {accuracy_score(y_val, lr_preds):.4f}")
    with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
        
    print("\n--- 2. XGBoost ---")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_val_scaled)
    print(f"Accuracy: {accuracy_score(y_val, xgb_preds):.4f}")
    xgb_model.save_model(os.path.join(models_dir, 'xgboost.json'))
    
    print("\n--- 3. Deep Neural Network (TensorFlow) ---")
    dnn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_val_scaled, y_val), verbose=0)
    
    dnn_loss, dnn_acc = dnn_model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f"Accuracy: {dnn_acc:.4f}")
    dnn_model.save(os.path.join(models_dir, 'dnn_model.h5'))
    
    print("\nModels and scaler saved successfully in the 'models' directory.")
    
    # Also save the column names for SHAP and prediction
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(list(X.columns), f)

if __name__ == "__main__":
    train_and_save_models()
