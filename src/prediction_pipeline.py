import joblib
from pathlib import Path


class PredictionPipeline:

    def __init__(self):

        BASE_DIR = Path(__file__).resolve().parent.parent

        self.model = joblib.load(
            BASE_DIR / "artifacts/models/xgboost_optimized.pkl"
        )

        self.threshold = joblib.load(
            BASE_DIR / "artifacts/models/xgboost_best_threshold.pkl"
        )

    def predict(self, df):

        probability = self.model.predict_proba(df)[:, 1]

        prediction = (probability >= self.threshold).astype(int)

        return prediction, probability