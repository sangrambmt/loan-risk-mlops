import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DuplicateRemover:
    """Removes exact duplicate rows from a DataFrame."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df     = df.drop_duplicates()
        after  = len(df)
        logger.info(f"[DuplicateRemover] Removed {before - after:,} duplicates | Rows: {before:,} -> {after:,}")
        print(f"[DuplicateRemover] Removed {before - after:,} duplicates | Rows: {before:,} -> {after:,}")
        return df.reset_index(drop=True)