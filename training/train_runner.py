import pandas as pd
import numpy as np
import logging
import os
import sys
import joblib
import shutil
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import mlflow.lightgbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
SCORING_DIR   = os.path.join(os.path.dirname(__file__), "..", "scoring")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "loan-default-prediction"

class TrainRunner:
    """Logs tuned LightGBM model, params, metrics and threshold to MLflow."""

    def __init__(self, target_col: str = "default"):
        self.target_col = target_col
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    def _load(self) -> tuple:
        model     = joblib.load(os.path.join(ARTIFACTS_DIR, "lightgbm_tuned.pkl"))
        threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "threshold.pkl"))
        report    = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                             "reports", "evaluation_report.csv"))
        metrics   = report.iloc[0].to_dict()
        return model, threshold, metrics

    def _save_to_scoring(self, model):
        os.makedirs(SCORING_DIR, exist_ok=True)
        out = os.path.join(SCORING_DIR, "model.pkl")
        joblib.dump(model, out)
        logger.info(f"[TrainRunner] Model saved -> {out}")
        print(f"  Model saved -> {out}")

    def run(self):
        logger.info("=== MLflow Logging Started ===")
        model, threshold, metrics = self._load()

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log params
            params = {
                "model_type":    "LightGBM",
                "threshold":     threshold,
                "n_estimators":  model.n_estimators_,
                "max_depth":     model.max_depth,
                "num_leaves":    model.num_leaves,
                "learning_rate": model.learning_rate,
                "smote":         True,
                "cv_folds":      5,
                "n_trials":      30,
            }
            mlflow.log_params(params)
            logger.info(f"[TrainRunner] Params logged")

            # Log metrics
            mlflow.log_metric("f1",        metrics["f1"])
            mlflow.log_metric("roc_auc",   metrics["roc_auc"])
            mlflow.log_metric("accuracy",  metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall",    metrics["recall"])
            logger.info(f"[TrainRunner] Metrics logged")

            # Log model
            mlflow.lightgbm.log_model(model, artifact_path="model")
            logger.info(f"[TrainRunner] Model logged to MLflow")

            # Log threshold as artifact
            threshold_path = os.path.join(ARTIFACTS_DIR, "threshold.pkl")
            mlflow.log_artifact(threshold_path, artifact_path="artifacts")

            # Log evaluation report
            report_path = os.path.join(os.path.dirname(__file__),
                                       "reports", "evaluation_report.csv")
            mlflow.log_artifact(report_path, artifact_path="reports")

        # Save to scoring
        self._save_to_scoring(model)

        # Print summary
        print(f"\n{'='*55}")
        print(f"  MLFLOW LOGGING COMPLETE")
        print(f"{'='*55}")
        print(f"  Experiment  : {EXPERIMENT_NAME}")
        print(f"  Run ID      : {run_id}")
        print(f"  F1          : {metrics['f1']}")
        print(f"  ROC-AUC     : {metrics['roc_auc']}")
        print(f"  Threshold   : {threshold}")
        print(f"  MLflow UI   : {MLFLOW_TRACKING_URI}")
        print(f"{'='*55}\n")

        logger.info(f"[TrainRunner] Run ID: {run_id}")
        logger.info("=== MLflow Logging Complete ===")
        return run_id

if __name__ == "__main__":
    TrainRunner().run()