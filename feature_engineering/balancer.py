import pandas as pd
import numpy as np
import logging
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

class Balancer:
    """Applies SMOTE to oversample the minority class on train data only."""

    def __init__(self,
                 target_col:    str   = "default",
                 sampling_strategy: float = 0.40,
                 seed:          int   = 42):
        self.target_col         = target_col
        self.sampling_strategy  = sampling_strategy
        self.seed               = seed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X      = df.drop(columns=[self.target_col])
        y      = df[self.target_col]

        print(f"\n{'='*55}")
        print(f"  SMOTE BALANCING")
        print(f"{'='*55}")
        print(f"  Before SMOTE:")
        print(f"    Class 0 (No Default) : {(y == 0).sum():,}")
        print(f"    Class 1 (Default)    : {(y == 1).sum():,}")
        print(f"    Default rate         : {y.mean():.2%}")

        smote       = SMOTE(sampling_strategy=self.sampling_strategy,
                            random_state=self.seed, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X, y)

        df_balanced              = pd.DataFrame(X_res, columns=X.columns)
        df_balanced[self.target_col] = y_res

        print(f"\n  After SMOTE:")
        print(f"    Class 0 (No Default) : {(y_res == 0).sum():,}")
        print(f"    Class 1 (Default)    : {(y_res == 1).sum():,}")
        print(f"    Default rate         : {y_res.mean():.2%}")
        print(f"    Total rows           : {len(df_balanced):,}")
        print(f"{'='*55}\n")

        logger.info(f"[Balancer] SMOTE complete | Rows: {len(df):,} -> {len(df_balanced):,} | "
                    f"Default rate: {y.mean():.2%} -> {y_res.mean():.2%}")
        return df_balanced