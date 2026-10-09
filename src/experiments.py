"""Experiments behind the modeling decisions.

Part 1: ways to give the model time information (Decisions log #11).
Part 2: model families on the final feature set.
Both use the 3 time-based folds (Decisions log #10) and report the mean
absolute error (MAE) in dollars per load, on all test rows and on normal
rows only (Decisions log #7). Part 2 also reports the share of normal loads
predicted within 5% of the true price.

Run from the project root:  python src/experiments.py
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import features
from prepare import Preparer, load_data

try:  # optional: pip install xgboost
    import xgboost as xgb
except ImportError:
    xgb = None

START = pd.Timestamp("2025-01-01")
FOLDS = [
    ("2025-05-01", "2025-07-01"),  # train Jan-Apr, test May-Jun
    ("2025-07-01", "2025-09-01"),  # train Jan-Jun, test Jul-Aug
    ("2025-09-01", "2025-11-01"),  # train Jan-Aug, test Sep-Oct
]
BASE_FEATURES = ["distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "equipment"]

EXPERIMENTS = {
    # name: (extra features, extrapolate a linear trend)
    "No time info, no market_index": ([], False),
    "A: + market_index": (["market_index"], False),
    "A + day_of_week": (["market_index", "day_of_week"], False),
    "B: + market_index + month": (["market_index", "month"], False),
    "C: + market_index + linear trend": (["market_index"], True),
}


def days_since_start(df: pd.DataFrame) -> pd.Series:
    return (df["date"] - START).dt.days


def build_features(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    X = df[BASE_FEATURES].copy()
    X["equipment"] = X["equipment"].astype("category")
    if "market_index" in extra:
        X["market_index"] = df["market_index"]
    if "month" in extra:
        X["month"] = df["date"].dt.month
    if "day_of_week" in extra:
        X["day_of_week"] = df["date"].dt.dayofweek
    return X


def trend_per_day(df: pd.DataFrame) -> float:
    """Daily trend of log(rate per mile) from a linear regression with basic controls."""
    X = np.column_stack([
        days_since_start(df),
        df["market_index"],
        np.log(df["distance"]),
        df["weight"],
        df["equipment"].eq("Flatbed"),
        df["equipment"].eq("Reefer"),
    ]).astype(float)
    y = np.log(df["posted_rate"] / df["distance"])
    return float(LinearRegression().fit(X, y).coef_[0])


def run_fold(train: pd.DataFrame, start: str, end: str, extra: list[str], use_trend: bool) -> tuple[float, float]:
    train_part = train[train["date"] < start]
    test_part = train[(train["date"] >= start) & (train["date"] < end)]

    prep = Preparer().fit(train_part)
    fit_rows = prep.transform(train_part[~prep.outlier_mask(train_part)])
    test_rows = prep.transform(test_part)

    # The model predicts log(rate per mile); price = rate per mile * distance.
    target = np.log(fit_rows["posted_rate"] / fit_rows["distance"])
    slope = trend_per_day(fit_rows) if use_trend else 0.0

    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, random_state=0, verbose=-1)
    model.fit(build_features(fit_rows, extra), target - slope * days_since_start(fit_rows))
    log_rpm = model.predict(build_features(test_rows, extra)) + slope * days_since_start(test_rows)
    predicted = np.exp(log_rpm) * test_rows["distance"]

    error = (test_rows["posted_rate"] - predicted).abs()
    normal = ~prep.outlier_mask(test_part)
    return float(error.mean()), float(error[normal].mean())


def as_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Equipment as integer codes, for models without native category support."""
    X = X.copy()
    X["equipment"] = X["equipment"].cat.codes
    return X


def for_linear(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot equipment and log distance, for the linear model."""
    X = X.copy()
    dummies = pd.get_dummies(X.pop("equipment"), drop_first=True).astype(float)
    X["log_distance"] = np.log(X["distance"])
    return pd.concat([X, dummies], axis=1)


MODELS = {
    # name: (model factory, input transformation)
    "Linear regression": (lambda: LinearRegression(), for_linear),
    "Decision tree": (lambda: DecisionTreeRegressor(min_samples_leaf=20, random_state=0), as_numeric),
    "Random forest": (lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0), as_numeric),
    "HistGradientBoosting": (lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=0), as_numeric),
    "LightGBM": (lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1), lambda X: X),
}
if xgb is not None:
    MODELS["XGBoost"] = (
        lambda: xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, tree_method="hist",
                                 enable_categorical=True, random_state=0),
        lambda X: X,
    )


def run_model_fold(train: pd.DataFrame, start: str, end: str, make_model, transform) -> tuple[float, float, float]:
    train_part = train[train["date"] < start]
    test_part = train[(train["date"] >= start) & (train["date"] < end)]
    prep = Preparer().fit(train_part)
    fit_rows = prep.transform(train_part[~prep.outlier_mask(train_part)])
    test_rows = prep.transform(test_part)

    model = make_model().fit(transform(features.build_features(fit_rows)), np.log(fit_rows["posted_rate"] / fit_rows["distance"]))
    predicted = np.exp(model.predict(transform(features.build_features(test_rows)))) * test_rows["distance"]
    error = (test_rows["posted_rate"] - predicted).abs()
    normal = ~prep.outlier_mask(test_part)
    within5 = (error[normal] / test_rows["posted_rate"][normal] <= 0.05).mean() * 100
    return float(error.mean()), float(error[normal].mean()), float(within5)


def print_header(title: str, with_within5: bool = False) -> None:
    print(f"\n{title}")
    extra = f" {'Within 5%':>10}" if with_within5 else ""
    print(f"{'':<34} {'MAE all':>8} {'MAE normal':>11}{extra}   normal MAE per fold")


def main() -> None:
    train, _, _ = load_data()

    print_header("Part 1: time information (LightGBM)")
    for name, (extra, use_trend) in EXPERIMENTS.items():
        results = np.array([run_fold(train, s, e, extra, use_trend) for s, e in FOLDS])
        folds = ", ".join(f"{v:.0f}" for v in results[:, 1])
        print(f"{name:<34} {results[:, 0].mean():8.1f} {results[:, 1].mean():11.1f}   [{folds}]")

    print_header("Part 2: model families (final features, may take a minute)", with_within5=True)
    for name, (make_model, transform) in MODELS.items():
        results = np.array([run_model_fold(train, s, e, make_model, transform) for s, e in FOLDS])
        folds = ", ".join(f"{v:.0f}" for v in results[:, 1])
        print(f"{name:<34} {results[:, 0].mean():8.1f} {results[:, 1].mean():11.1f} {results[:, 2].mean():9.1f}%   [{folds}]")


if __name__ == "__main__":
    main()
