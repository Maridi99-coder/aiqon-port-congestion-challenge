import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

def train_and_evaluate_model(data_path="data/port_data.csv", model_output="src/congestion_model.pkl"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")
        
    df = pd.read_csv(data_path)
    
    # One-hot encode categorical variables like weather condition
    df_encoded = pd.get_dummies(df, columns=['weather_condition'], drop_first=True)
    
    X = df_encoded.drop(columns=['is_congested'])
    y = df_encoded['is_congested']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("--- Model Evaluation Metrics ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Explainability: Feature Importances
    importances = model.feature_importances_
    features = X.columns
    print("--- Key Contributing Factors (Explainability) ---")
    for feature, importance in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
        
    # Save model artifact
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)
    print(f"\nModel saved successfully to {model_output}")

if __name__ == "__main__":
    train_and_evaluate_model()