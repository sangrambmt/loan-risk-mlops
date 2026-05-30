import pandas as pd
import numpy as np
import logging
import os
import joblib

from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
REPORTS_DIR   = os.path.join(os.path.dirname(__file__), "reports")

class Evaluator:
    """Evaluates tuned LightGBM with optimal threshold on test set."""

    def __init__(self, target_col: str = "default"):
        self.target_col = target_col
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def _load(self) -> tuple:
        model     = joblib.load(os.path.join(ARTIFACTS_DIR, "lightgbm_tuned.pkl"))
        threshold = joblib.load(os.path.join(ARTIFACTS_DIR, "threshold.pkl"))
        test      = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
        X_test    = test.drop(columns=[self.target_col])
        y_test    = test[self.target_col]
        logger.info(f"[Evaluator] Test: {X_test.shape}  Threshold: {threshold}")
        return model, threshold, X_test, y_test

    def run(self) -> dict:
        logger.info("=== Model Evaluation Started ===")
        model, threshold, X_test, y_test = self._load()

        # Predict
        probs  = model.predict_proba(X_test)[:, 1]
        preds  = (probs >= threshold).astype(int)

        # Metrics
        metrics = {
            "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, probs), 4),
            "accuracy":  round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
            "threshold": threshold,
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        # Print
        print(f"\n{'='*55}")
        print(f"  FINAL MODEL EVALUATION ON TEST SET")
        print(f"{'='*55}")
        print(f"  Model          : LightGBM (tuned)")
        print(f"  Threshold      : {threshold}")
        print(f"  Test rows      : {len(y_test):,}")
        print(f"{'='*55}")
        print(f"  F1 Score       : {metrics['f1']}")
        print(f"  ROC-AUC        : {metrics['roc_auc']}")
        print(f"  Accuracy       : {metrics['accuracy']}")
        print(f"  Precision      : {metrics['precision']}")
        print(f"  Recall         : {metrics['recall']}")
        print(f"{'='*55}")
        print(f"  CONFUSION MATRIX")
        print(f"{'='*55}")
        print(f"                  Predicted")
        print(f"                  No Default  Default")
        print(f"  Actual No Default  {tn:>6}     {fp:>6}")
        print(f"  Actual Default     {fn:>6}     {tp:>6}")
        print(f"{'='*55}")
        print(f"  True Positives  (caught defaults)    : {tp:,}")
        print(f"  False Negatives (missed defaults)    : {fn:,}")
        print(f"  False Positives (wrong alerts)       : {fp:,}")
        print(f"  True Negatives  (correct approvals)  : {tn:,}")
        print(f"{'='*55}")
        print(f"\n  CLASSIFICATION REPORT")
        print(f"{'='*55}")
        print(classification_report(y_test, preds,
                                    target_names=["No Default", "Default"]))
        print(f"{'='*55}\n")

        # Save report
        report = pd.DataFrame([metrics])
        report.to_csv(os.path.join(REPORTS_DIR, "evaluation_report.csv"), index=False)
        logger.info(f"[Evaluator] Report saved -> {REPORTS_DIR}/evaluation_report.csv")
        logger.info("=== Model Evaluation Complete ===")
        return metrics

if __name__ == "__main__":
    Evaluator().run()