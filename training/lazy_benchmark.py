import pandas as pd
import numpy as np
import os
import logging
import warnings
warnings.filterwarnings("ignore")

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
REPORTS_DIR  = os.path.join(os.path.dirname(__file__), "reports")

class LazyBenchmark:
    """Runs LazyPredict across all classifiers and returns a ranked leaderboard."""

    def __init__(self,
                 sample_frac: float = 0.20,
                 target_col:  str   = "default",
                 seed:        int   = 42):
        self.sample_frac = sample_frac
        self.target_col  = target_col
        self.seed        = seed
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def _load(self) -> pd.DataFrame:
        path = os.path.join(DATA_DIR, "train_data.csv")
        df   = pd.read_csv(path)
        logger.info(f"[LazyBenchmark] Loaded train data: {df.shape}")
        return df

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        n      = int(len(df) * self.sample_frac)
        sample = df.sample(n=n, random_state=self.seed)
        logger.info(f"[LazyBenchmark] Sampled {n:,} rows ({self.sample_frac:.0%} of train)")
        print(f"[LazyBenchmark] Running benchmark on {n:,} rows ({self.sample_frac:.0%} sample) ...")
        return sample

    def run(self) -> pd.DataFrame:
        logger.info("=== LazyPredict Benchmark Started ===")

        # Load & sample
        df     = self._load()
        sample = self._sample(df)

        X = sample.drop(columns=[self.target_col])
        y = sample[self.target_col]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=self.seed, stratify=y)

        # Run LazyClassifier
        clf     = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_tr, X_te, y_tr, y_te)

        # Sort by F1 score
        leaderboard = models.sort_values("F1 Score", ascending=False).reset_index()
        leaderboard.columns = [c.strip() for c in leaderboard.columns]

        # Print leaderboard
        print(f"\n{'='*70}")
        print(f"  LAZYPREDICT LEADERBOARD  (sample={self.sample_frac:.0%})")
        print(f"{'='*70}")
        print(leaderboard[["Model", "F1 Score", "Accuracy",
                            "Balanced Accuracy", "Time Taken"]].to_string(index=False))
        print(f"{'='*70}\n")

        # Top 3
        top3 = leaderboard.head(3)
        print(f"{'='*70}")
        print(f"  TOP 3 MODELS TO TRAIN FULLY")
        print(f"{'='*70}")
        for i, row in top3.iterrows():
            print(f"  {i+1}. {row['Model']:<40} F1: {row['F1 Score']:.4f}")
        print(f"{'='*70}\n")

        # Save
        out = os.path.join(REPORTS_DIR, "lazy_leaderboard.csv")
        leaderboard.to_csv(out, index=False)
        logger.info(f"[LazyBenchmark] Leaderboard saved -> {out}")
        print(f"[LazyBenchmark] Leaderboard saved -> {out}")

        logger.info("=== LazyPredict Benchmark Complete ===")
        return leaderboard

if __name__ == "__main__":
    LazyBenchmark(sample_frac=0.20).run()