import pandas as pd
import numpy as np
import logging
import os
import joblib

from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

class ThresholdTuner:
    """Finds optimal decision threshold by maximizing F1 on validation set."""

    def __init__(self, target_col: str = "default", seed: int = 42):
        self.target_col = target_col
        self.seed       = seed
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def _load(self) -> tuple:
        model = joblib.load(os.path.join(ARTIFACTS_DIR, "lightgbm_tuned.pkl"))
        val   = pd.read_csv(os.path.join(DATA_DIR, "val_data.csv"))
        X_val = val.drop(columns=[self.target_col])
        y_val = val[self.target_col]
        logger.info(f"[ThresholdTuner] Val: {X_val.shape}")
        return model, X_val, y_val

    def run(self) -> float:
        logger.info("=== Threshold Tuning Started ===")
        model, X_val, y_val = self._load()

        # Get probabilities
        probs      = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.10, 0.90, 0.01)
        results    = []

        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            results.append({
                "threshold": round(thresh, 2),
                "f1":        round(f1_score(y_val, preds, zero_division=0), 4),
                "precision": round(precision_score(y_val, preds, zero_division=0), 4),
                "recall":    round(recall_score(y_val, preds, zero_division=0), 4),
            })

        df_results   = pd.DataFrame(results)
        best_idx     = df_results["f1"].idxmax()
        best_row     = df_results.loc[best_idx]
        best_thresh  = best_row["threshold"]

        # Print table
        print(f"\n{'='*60}")
        print(f"  THRESHOLD TUNING RESULTS")
        print(f"{'='*60}")
        print(df_results.to_string(index=False))
        print(f"{'='*60}")
        print(f"\n  Best Threshold : {best_thresh}")
        print(f"  Best F1        : {best_row['f1']}")
        print(f"  Precision      : {best_row['precision']}")
        print(f"  Recall         : {best_row['recall']}")
        print(f"{'='*60}\n")

        # Save
        joblib.dump(best_thresh, os.path.join(ARTIFACTS_DIR, "threshold.pkl"))
        logger.info(f"[ThresholdTuner] Best threshold {best_thresh} saved")
        print(f"[ThresholdTuner] Threshold saved -> {ARTIFACTS_DIR}/threshold.pkl")

        logger.info("=== Threshold Tuning Complete ===")
        return best_thresh

if __name__ == "__main__":
    ThresholdTuner().run()