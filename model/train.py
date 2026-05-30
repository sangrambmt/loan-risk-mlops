import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib
import os

# ── Config ──────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "loan-default-prediction"
DATA_PATH           = os.path.join(os.path.dirname(__file__), "../data/sample_data.csv")
MODEL_OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "../scoring/model.pkl")

# ── MLflow setup ─────────────────────────�# ── MLflow setup ──�# ��─────�# ─────────────────
mlflow.set_trackinmlflow.set_trackinmlflow.set_trackinmlflow.set_trEXPmlflow.set_trackinmlflow.set_trackinmlfl
    df =    df =    df =    df =    df =    df =    df =    df =    df =    df =    df =  re    df =    df =    df =    df t_size=  2, random_state=42)

def build_pipeline(n_estimators=100, mdef build_pipeline(n_estimators=100, mdef build_pipeline(n_estimators=100, mdef build_pipeline(n_estimators=100, mdef build_pipeline(n_estimators=100, mdef build_pipeline(n_estimators=100, mdef build_pipeline(n_es  def build_pipeline(n_estimators=100, mdef build_pipeline(n_estimators=100est, y_test):
    preds = pipeline.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precisi        "precision": precisi        "precision": precisi        "precision": precis pre        "precision": precisi        "precision": precisi        "precision": precisi        "precision": precis pre        "precision": precisi        "precision": precisi        "precision": precis
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    with mlflow.start_run():
        mlflow.log_params        mlflow.log_params        mlflow.log_params        mlflow.log_params        mlflow.log_params        mlflow.log_params        mlflow.logst)
                                                                                                                                                                                                                                                                                                                                                      ():
            print(f"   {k}: {v:.4f}")
        print(f"   Model saved → {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train()
