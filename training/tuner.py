import pandas as pd
import numpy as np
import logging
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from lightgbm                import LGBMClassifier
from catboost                import CatBoostClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
REPORTS_DIR   = os.path.join(os.path.dirname(__file__), "reports")

class ModelTuner:
    """Tunes LightGBM, CatBoost and MLP using Optuna with cross-validation."""

    N_TRIALS = 30
    CV_FOLDS = 5

    def __init__(self, target_col: str = "default", seed: int = 42):
        self.target_col = target_col
        self.seed       = seed
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR,   exist_ok=True)

    def _load(self) -> tuple:
        train = pd.read_csv(os.path.join(DATA_DIR, "train_balanced.csv"))
        val   = pd.read_csv(os.path.join(DATA_DIR, "val_data.csv"))
        X_tr  = train.drop(columns=[self.target_col])
        y_tr  = train[self.target_col]
        X_val = val.drop(columns=[self.target_col])
        y_val = val[self.target_col]
        logger.info(f"[Tuner] Train: {X_tr.shape}  Val: {X_val.shape}")
        return X_tr, y_tr, X_val, y_val

    def _cv_score(self, model, X, y) -> float:
        cv      = StratifiedKFold(n_splits=self.CV_FOLDS, shuffle=True, random_state=self.seed)
        scores  = []
        for tr_idx, va_idx in cv.split(X, y):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = model.predict(X.iloc[va_idx])
            scores.append(f1_score(y.iloc[va_idx], preds, zero_division=0))
        return float(np.mean(scores))

    # ── LightGBM ──────────────────────────────────────────────────────────────
    def _tune_lightgbm(self, X_tr, y_tr, X_val, y_val) -> dict:
        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "class_weight":     "balanced",
                "random_state":     self.seed,
                "n_jobs":           -1,
                "verbose":          -1,
            }
            return self._cv_score(LGBMClassifier(**params), X_tr, y_tr)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.N_TRIALS, show_progress_bar=False)
        best  = study.best_params
        model = LGBMClassifier(**best, class_weight="balanced",
                               random_state=self.seed, n_jobs=-1, verbose=-1)
        model.fit(X_tr, y_tr)
        f1  = f1_score(y_val, model.predict(X_val), zero_division=0)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        return {"model": model, "params": best, "f1": round(f1, 4), "auc": round(auc, 4)}

    # ── CatBoost ──────────────────────────────────────────────────────────────
    def _tune_catboost(self, X_tr, y_tr, X_val, y_val) -> dict:
        def objective(trial):
            params = {
                "iterations":        trial.suggest_int("iterations", 100, 500),
                "depth":             trial.suggest_int("depth", 3, 10),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "bagging_temperature":trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "class_weights":     [1, 6],
                "random_state":      self.seed,
                "verbose":           0,
            }
            return self._cv_score(CatBoostClassifier(**params), X_tr, y_tr)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.N_TRIALS, show_progress_bar=False)
        best  = study.best_params
        model = CatBoostClassifier(**best, class_weights=[1, 6],
                                   random_state=self.seed, verbose=0)
        model.fit(X_tr, y_tr)
        f1  = f1_score(y_val, model.predict(X_val), zero_division=0)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        return {"model": model, "params": best, "f1": round(f1, 4), "auc": round(auc, 4)}

    # ── MLP ───────────────────────────────────────────────────────────────────
    def _tune_mlp(self, X_tr, y_tr, X_val, y_val) -> dict:
        def objective(trial):
            n_layers = trial.suggest_int("n_layers", 1, 3)
            layers   = tuple(trial.suggest_int(f"n_units_{i}", 32, 256) for i in range(n_layers))
            params   = {
                "hidden_layer_sizes": layers,
                "activation":         trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha":              trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "max_iter":           200,
                "random_state":       self.seed,
            }
            return self._cv_score(MLPClassifier(**params), X_tr, y_tr)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.N_TRIALS, show_progress_bar=False)
        best  = study.best_params
        n_layers = best.pop("n_layers")
        layers   = tuple(best.pop(f"n_units_{i}") for i in range(n_layers))
        model    = MLPClassifier(hidden_layer_sizes=layers, **best,
                                 max_iter=200, random_state=self.seed)
        model.fit(X_tr, y_tr)
        f1  = f1_score(y_val, model.predict(X_val), zero_division=0)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        return {"model": model, "params": best, "f1": round(f1, 4), "auc": round(auc, 4)}

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        logger.info("=== Optuna Tuning Started ===")
        X_tr, y_tr, X_val, y_val = self._load()
        results = {}

        for name, fn in [("LightGBM", self._tune_lightgbm),
                         ("CatBoost", self._tune_catboost),
                         ("MLP",      self._tune_mlp)]:
            print(f"\n>>> Tuning {name} ({self.N_TRIALS} trials x {self.CV_FOLDS} folds) ...")
            result        = fn(X_tr, y_tr, X_val, y_val)
            results[name] = result
            joblib.dump(result["model"], os.path.join(ARTIFACTS_DIR, f"{name.lower()}_tuned.pkl"))
            print(f"    Best F1  : {result['f1']}")
            print(f"    Best AUC : {result['auc']}")
            print(f"    Params   : {result['params']}")
            logger.info(f"[Tuner] {name} -> F1={result['f1']}  AUC={result['auc']}")

        # Summary
        print(f"\n{'='*55}")
        print(f"  TUNING SUMMARY")
        print(f"{'='*55}")
        for name, r in results.items():
            print(f"  {name:<12} F1={r['f1']}  AUC={r['auc']}")
        best_model = max(results, key=lambda x: results[x]["f1"])
        print(f"\n  Best model : {best_model}")
        print(f"{'='*55}\n")

        # Save report
        report = pd.DataFrame([{"Model": k, "F1": v["f1"], "AUC": v["auc"],
                                 "Params": str(v["params"])}
                                for k, v in results.items()])
        report.to_csv(os.path.join(REPORTS_DIR, "tuning_report.csv"), index=False)
        logger.info("=== Optuna Tuning Complete ===")
        return results

if __name__ == "__main__":
    ModelTuner().run()