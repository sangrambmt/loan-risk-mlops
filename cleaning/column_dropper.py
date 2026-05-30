import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ColumnDropper:
    """Drops columns exceeding the missing value threshold."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        miss_rate  = df.isnull().mean()
        drop_cols  = miss_rate[miss_rate > self.threshold].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logger.info(f"[ColumnDropper] Dropped columns: {drop_cols}")
            print(f"[ColumnDropper] Dropped columns: {drop_cols}")
        else:
            logger.info(f"[ColumnDropper] No columns exceeded {self.threshold:.0%} missing threshold")
            print(f"[ColumnDropper] No columns exceeded {self.threshold:.0%} missing threshold")
        return df