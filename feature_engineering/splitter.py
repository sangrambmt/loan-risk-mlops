import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from typing import Tuple

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

class Splitter:
    """Splits processed data into train, val and test sets with stratification."""

    def __init__(self,
                 test_size:  float = 0.15,
                 val_size:   float = 0.15,
                 target_col: str   = "default",
                 seed:       int   = 42):
        self.test_size  = test_size
        self.val_size   = val_size
        self.target_col = target_col
        self.seed       = seed
        os.makedirs(DATA_DIR, exist_ok=True)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X       = df.drop(columns=[self.target_col])
        y       = df[self.target_col]
        val_adj = self.val_size / (1 - self.test_size)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.seed, stratify=y)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr, y_tr, test_size=val_adj,
            random_state=self.seed, stratify=y_tr)

        train = X_tr.copy(); train[self.target_col] = y_tr.values
        val   = X_va.copy(); val[self.target_col]   = y_va.values
        test  = X_te.copy(); test[self.target_col]  = y_te.values

        # Save
        train.to_csv(os.path.join(DATA_DIR, "train_data.csv"), index=False)
        val.to_csv(os.path.join(DATA_DIR,   "val_data.csv"),   index=False)
        test.to_csv(os.path.join(DATA_DIR,  "test_data.csv"),  index=False)

        # Summary
        print(f"\n{'='*55}")
        print(f"  SPLIT SUMMARY")
        print(f"{'='*55}")
        print(f"  Train : {len(train):,} rows  | Default rate: {train[self.target_col].mean():.2%}")
        print(f"  Val   : {len(val):,}  rows  | Default rate: {val[self.target_col].mean():.2%}")
        print(f"  Test  : {len(test):,} rows  | Default rate: {test[self.target_col].mean():.2%}")
        print(f"{'='*55}\n")

        logger.info(f"[Splitter] Train:{len(train):,}  Val:{len(val):,}  Test:{len(test):,}")
        return train, val, test