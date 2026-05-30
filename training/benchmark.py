import pandas as pd
import numpy as np
import os
import logging
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble        import (RandomForestClassifier, ExtraTreesClassifier,
                                     BaggingClassifier, GradientBoostingClassifier,
                                     HistGradientBoostingClassifier, AdaBoostClassifier)
from sklearn.svm             import LinearSVC
from sklearn.naive_bayes     import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (f1_score, roc_auc_score, accuracy_score,
                                     precision_score, recall_score)
from sklearn.model_selection import train_test_split
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from catboost                import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

class ModelBenchmark:
    """Benchmarks 20 classifiers on a sample and ranks by F1 & ROC-AUC."""

    MODELS = {
        "LogisticRegression":           LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RidgeClassifier":              RidgeClassifier(class_weight="balanced"),
        "SGDClassifier":                SGDClassifier(class_weight="balanced", random_state=42),
        "KNeighborsClassifier":         KNeighborsClassifier(n_jobs=-1),
        "DecisionTree":                 DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "ExtraTree":                    ExtraTreeClassifier(class_weight="balanced", random_state=42),
        "RandomForest":                 RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                               n_jobs=-1, random_state=42),
        "ExtraTrees":                   ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                                             n_jobs=-1, random_state=42),
        "BaggingClassifier":            BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "GradientBoosting":             GradientBoostingClassifier(n_estimators=100, random_state=42),
        "HistGradientBoosting":         HistGradientBoostingClassifier(random_state=42),
        "AdaBoost":                     AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost":                      XGBClassifier(n_estimators=100, scale_pos_weight=6,
                                                      use_label_encoder=False, eval_metric="logloss",
                                                      random_state=42, n_jobs=-1),
        "LightGBM":                     LGBMClassifier(n_estimators=100, class_weight="balanced",
                                                       random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost":                     CatBoostClassifier(iterations=100, class_weights=[1, 6],
                                                           random_state=42, verbose=0),
        "LinearSVC":                    LinearSVC(class_weight="balanced", max_iter=2000, random_state=42),
        "GaussianNB":                   GaussianNB(),
        "LinearDiscriminantAnalysis":   LinearDiscriminantAnalysis(),
        "QuadraticDiscriminantAnalysis":QuadraticDiscriminantAnalysis(),
        "MLPClassifier":                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                                      random_state=42),
    }

    def __init__(self, sample_frac: float = 0.20, target_col: str = "default", seed: int = 42):
        self.sample_frac = sample_frac
        self.target_col  = target_col
        self.seed        = seed
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def _load_and_sample(self) -> tuple:
        df     = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
        sample = df.sample(frac=self.sample_frac, random_state=self.seed)
        X      = sample.drop(columns=[self.target_col])
        y      = sample[self.target_col]
        logger.info(f"[Benchmark] Sample: {sample.shape}  Default rate: {y.mean():.2%}")
        print(f"[Benchmark] Running 20 models on {len(sample):,} rows ({self.sample_frac:.0%} sample) ...")
        return train_test_split(X, y, test_size=0.20, random_state=self.seed, stratify=y)

    def _evaluate(self, model, name: str, X_tr, X_te, y_tr, y_te) -> dict:
        try:
            start = time.time()
            model.fit(X_tr, y_tr)
            elapsed = round(time.time() - start, 2)

            # Handle models without predict_proba
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_te)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_te)
            else:
                y_prob = None

            y_pred  = model.predict(X_te)
            roc_auc = roc_auc_score(y_te, y_prob) if y_prob is not None else None

            return {
                "Model":     name,
                "F1":        round(f1_score(y_te, y_pred, zero_division=0), 4),
                "ROC-AUC":   round(roc_auc, 4) if roc_auc else "N/A",
                "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
                "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
                "Recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
                "Time(s)":   elapsed,
                "Status":    "OK"
            }
        except Exception as e:
            logger.warning(f"[Benchmark] {name} failed: {e}")
            return {"Model": name, "F1": 0, "ROC-AUC": 0, "Accuracy": 0,
                    "Precision": 0, "Recall": 0, "Time(s)": 0, "Status": f"FAILED: {e}"}

    def run(self) -> pd.DataFrame:
        logger.info("=== Model Benchmark Started ===")
        X_tr, X_te, y_tr, y_te = self._load_and_sample()
        results = []

        for i, (name, model) in enumerate(self.MODELS.items(), 1):
            print(f"  [{i:02d}/20] {name:<35} ...", end=" ", flush=True)
            result = self._evaluate(model, name, X_tr, X_te, y_tr, y_te)
            results.append(result)
            print(f"F1={result['F1']}  ROC-AUC={result['ROC-AUC']}  Time={result['Time(s)']}s")

        leaderboard = (pd.DataFrame(results)
                       .sort_values("F1", ascending=False)
                       .reset_index(drop=True))
        leaderboard.index += 1

        # Print leaderboard
        print(f"\n{'='*75}")
        print(f"  BENCHMARK LEADERBOARD")
        print(f"{'='*75}")
        print(leaderboard[["Model","F1","ROC-AUC","Accuracy",
                            "Precision","Recall","Time(s)"]].to_string())
        print(f"{'='*75}")

        # Top 3
        top3 = leaderboard.head(3)
        print(f"\n{'='*75}")
        print(f"  TOP 3 MODELS SELECTED FOR FULL TRAINING")
        print(f"{'='*75}")
        for i, row in top3.iterrows():
            print(f"  {i}. {row['Model']:<35} F1={row['F1']}  ROC-AUC={row['ROC-AUC']}")
        print(f"{'='*75}\n")

        # Save
        out = os.path.join(REPORTS_DIR, "benchmark_leaderboard.csv")
        leaderboard.to_csv(out, index=False)
        logger.info(f"[Benchmark] Leaderboard saved -> {out}")
        print(f"[Benchmark] Leaderboard saved -> {out}")
        logger.info("=== Model Benchmark Complete ===")
        return leaderboard

if __name__ == "__main__":
    ModelBenchmark(sample_frac=0.20).run()