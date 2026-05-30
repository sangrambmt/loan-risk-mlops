import pandas as pd
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eda.statistics_analyzer  import StatisticsAnalyzer
from eda.distribution_plotter import DistributionPlotter
from eda.correlation_analyzer import CorrelationAnalyzer
from eda.outlier_detector     import OutlierDetector
from eda.target_analyzer      import TargetAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

class EDARunner:
    """Orchestrates all EDA steps sequentially."""

    def __init__(self):
        self.steps = [
            ("Statistics Analyzer",  StatisticsAnalyzer()),
            ("Distribution Plotter", DistributionPlotter()),
            ("Correlation Analyzer", CorrelationAnalyzer()),
            ("Outlier Detector",     OutlierDetector()),
            ("Target Analyzer",      TargetAnalyzer()),
        ]

    def _summary(self, df: pd.DataFrame):
        print(f"\n{'='*50}")
        print(f"  DATASET OVERVIEW")
        print(f"{'='*50}")
        print(f"  Rows        : {len(df):,}")
        print(f"  Columns     : {df.shape[1]}")
        print(f"  Nulls       : {df.isnull().sum().sum():,}")
        print(f"  Duplicates  : {df.duplicated().sum():,}")
        print(f"  Dtypes      :")
        for dtype, count in df.dtypes.value_counts().items():
            print(f"    {str(dtype):<12} : {count} columns")
        print(f"{'='*50}\n")

    def run(self):
        logger.info("=== EDA Pipeline Started ===")

        # Load
        input_path = os.path.join(DATA_DIR, "clean_data.csv")
        logger.info(f"Loading clean data from {input_path}")
        df = pd.read_csv(input_path)
        self._summary(df)

        # Run each step
        for name, step in self.steps:
            logger.info(f"Running {name} ...")
            print(f"\n>>> {name}")
            if   hasattr(step, "analyze"): step.analyze(df)
            elif hasattr(step, "plot"):    step.plot(df)
            elif hasattr(step, "detect"):  step.detect(df)

        logger.info("=== EDA Pipeline Complete ===")
        print(f"\n  All reports -> eda/reports/")
        print(f"  All plots   -> eda/plots/\n")

if __name__ == "__main__":
    EDARunner().run()