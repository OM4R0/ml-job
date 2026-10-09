"""Validate, train and predict.

1. Cross-validation on 3 time-based folds (Decisions log #10): a simple
   baseline vs the gradient boosting model (Decisions log #12), error on all
   test rows and on normal rows.
2. Retrain the model on all Jan-Oct data.
3. Write validation_predictions.csv and fill data/december_chart_inputs.csv,
   ready for score.py (a copy of the December predictions goes to outputs/).

Run from the project root:  python src/train.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from features import FEATURES, build_features
from prepare import DATA_DIR, DISTANCE_BANDS, Preparer, load_data, prepare_december

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FOLDS = [
    ("May-Jun", "2025-05-01", "2025-07-01"),  # train Jan-Apr
    ("Jul-Aug", "2025-07-01", "2025-09-01"),  # train Jan-Jun
    ("Sep-Oct", "2025-09-01", "2025-11-01"),  # train Jan-Aug
]
# Tuned with src/tune.py (Decisions log #13): 81 settings on the same 3 folds.
MODEL_PARAMS = {
    "max_iter": 1000,  # upper limit; early stopping picks the real number of trees
    "learning_rate": 0.03,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "random_state": 0,
}


class BaselineModel:
    """Median rate per mile for each (distance band, equipment), times distance."""

    def fit(self, df: pd.DataFrame) -> "BaselineModel":
        rate_per_mile = df["posted_rate"] / df["distance"]
        bands = pd.cut(df["distance"], DISTANCE_BANDS)
        self.table = rate_per_mile.groupby([bands, df["equipment"]], observed=True).median().to_dict()
        self.overall = float(rate_per_mile.median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        bands = pd.cut(df["distance"], DISTANCE_BANDS)
        rate_per_mile = [self.table.get(key, self.overall) for key in zip(bands, df["equipment"])]
        return np.asarray(rate_per_mile) * df["distance"].to_numpy()


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Model features with equipment as integer codes (Dry Van 0, Flatbed 1, Reefer 2)."""
    X = build_features(df)
    X["equipment"] = X["equipment"].cat.codes
    return X


class RatePerMileModel:
    """Gradient boosting on log(rate per mile); price = exp(prediction) x distance."""

    def fit(self, df: pd.DataFrame) -> "RatePerMileModel":
        target = np.log(df["posted_rate"] / df["distance"])
        self.model = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(model_input(df), target)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(model_input(df))) * df["distance"].to_numpy()


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Mean absolute error in dollars and mean absolute percentage error."""
    abs_error = np.abs(y_true - y_pred)
    return float(abs_error.mean()), float((abs_error / y_true).mean() * 100)


def cross_validate(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, start, end in FOLDS:
        train_part = train[train["date"] < start]
        test_part = train[(train["date"] >= start) & (train["date"] < end)]

        prep = Preparer().fit(train_part)
        fit_rows = prep.transform(train_part[~prep.outlier_mask(train_part)])  # outliers out of training only
        test_rows = prep.transform(test_part)
        normal = ~prep.outlier_mask(test_part).to_numpy()
        y_true = test_rows["posted_rate"].to_numpy()

        for name, model in [("baseline", BaselineModel()), ("gradient_boosting", RatePerMileModel())]:
            y_pred = model.fit(fit_rows).predict(test_rows)
            mae_all, mape_all = error_metrics(y_true, y_pred)
            mae_normal, mape_normal = error_metrics(y_true[normal], y_pred[normal])
            rows.append({
                "test_period": label, "model": name, "mean_price_normal": float(y_true[normal].mean()),
                "mae_all": mae_all, "mape_all": mape_all,
                "mae_normal": mae_normal, "mape_normal": mape_normal,
            })
    return pd.DataFrame(rows)


def main() -> None:
    train, val, dec = load_data()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Cross-validation
    cv = cross_validate(train)
    cv.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)
    print("Cross-validation (MAE in $, MAPE in %):")
    print(cv.round(2).to_string(index=False))
    summary = cv.groupby("model")[["mae_all", "mape_all", "mae_normal", "mape_normal"]].agg(["mean", "std"]).round(2)
    print("\nMean and spread over the 3 folds:")
    print(summary.to_string())

    # 2. Final model on all Jan-Oct data
    prep = Preparer().fit(train)
    fit_rows = prep.transform(train[~prep.outlier_mask(train)])
    model = RatePerMileModel().fit(fit_rows)

    # Permutation importance: how much the error grows when one feature is shuffled.
    sample = fit_rows.sample(5000, random_state=0)
    result = permutation_importance(
        model.model, model_input(sample), np.log(sample["posted_rate"] / sample["distance"]),
        n_repeats=5, random_state=0, scoring="neg_mean_absolute_error",
    )
    importance = pd.Series(result.importances_mean, index=FEATURES).clip(lower=0)
    importance = (importance / importance.sum()).sort_values(ascending=False)
    importance.rename("importance_share").to_csv(OUTPUT_DIR / "feature_importance.csv")
    print("\nFeature importance (share, permutation on a training sample):")
    print(importance.round(3).to_string())

    # 3. Predictions
    val_rows = prep.transform(val)
    predictions = pd.DataFrame({"load_id": val_rows["load_id"], "predicted_rate": model.predict(val_rows).round(2)})
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    submission = template[["load_id"]].merge(predictions, on="load_id", how="left")
    if len(submission) != len(template) or submission["predicted_rate"].isna().any():
        raise RuntimeError("Predictions do not cover every load_id in the template")
    submission.to_csv(ROOT / "validation_predictions.csv", index=False)

    dec_rows = prepare_december(dec, train, val, prep)
    dec_out = dec.copy()
    dec_out["predicted_rate"] = model.predict(dec_rows).round(2)
    dec_out.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)  # file read by score.py
    dec_out.to_csv(OUTPUT_DIR / "december_predictions.csv", index=False)  # copy kept with the results

    print(f"\nWrote {ROOT / 'validation_predictions.csv'} ({len(submission)} rows)")
    print(f"Filled {DATA_DIR / 'december_chart_inputs.csv'} ({len(dec_out)} rows)")


if __name__ == "__main__":
    main()
