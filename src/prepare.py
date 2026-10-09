"""Data preparation for the freight rate model.

Applies the decisions from notebooks/01_eda.ipynb (Decisions log #1-#9):
- abs() on weight (#4), then fill missing weight with the training median (#1, #2)
- fill missing market_index with the median of the same day (#3, #5)
- drop quote_signal (#8)
- flag price outliers so they can be removed from the training part only (#7)
- December chart file: coordinates from a city lookup, market_index from
  validation rows of the same day (#9)
Statistics that use the training data (weight median, normal rate per mile
per distance band) are learned on the training part only (#6). market_index is
an input, not the target: it is filled from same-day rows of the same table,
and the December file uses validation rows of the same day (#3, #5, #9).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DISTANCE_BANDS = [0, 150, 250, 500, 1000, 2000, float("inf")]
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 1.6
COORD_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation and December chart inputs with parsed dates."""
    train = pd.read_csv(data_dir / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(data_dir / "validation.csv", parse_dates=["date"])
    dec = pd.read_csv(data_dir / "december_chart_inputs.csv", parse_dates=["date"])
    return train, val, dec


def daily_median(df: pd.DataFrame, column: str) -> pd.Series:
    """Median of a column for each date."""
    return df.groupby("date")[column].median()


def fill_by_day(df: pd.DataFrame, column: str, reference: pd.Series | None = None) -> pd.DataFrame:
    """Fill missing values of `column` with the median of the same day.

    The daily medians come from `reference` when given (December file),
    otherwise from the rows of `df` itself.
    """
    out = df.copy()
    if column not in out:
        out[column] = float("nan")
    per_day = reference if reference is not None else daily_median(out, column)
    out[column] = out[column].fillna(out["date"].map(per_day))
    return out


def city_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city, taken from pickup and delivery columns."""
    pickup = df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    delivery = df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    return pd.concat([pickup, delivery]).drop_duplicates("city").set_index("city")


def add_coordinates(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Add pickup/delivery coordinates by looking up the city names."""
    out = df.copy()
    for side in ["pickup", "delivery"]:
        out[f"{side}_lat"] = out[side].map(coords["lat"])
        out[f"{side}_lon"] = out[side].map(coords["lon"])
    missing = out[COORD_COLUMNS].isna().any(axis=1)
    if missing.any():
        cities = sorted(set(out.loc[missing, "pickup"]) | set(out.loc[missing, "delivery"]))
        raise ValueError(f"No coordinates found for: {cities}")
    return out


class Preparer:
    """Learns statistics on a training part and applies them to any part."""

    def __init__(self) -> None:
        self.weight_median: float | None = None
        self.band_rpm_median: dict | None = None

    def fit(self, train_part: pd.DataFrame) -> "Preparer":
        self.weight_median = float(train_part["weight"].abs().median())
        rate_per_mile = train_part["posted_rate"] / train_part["distance"]
        bands = pd.cut(train_part["distance"], DISTANCE_BANDS)
        self.band_rpm_median = rate_per_mile.groupby(bands, observed=True).median().to_dict()
        return self

    def _check_fitted(self) -> None:
        if self.weight_median is None or self.band_rpm_median is None:
            raise RuntimeError("Preparer is not fitted: call fit() on the training part first")

    def transform(self, df: pd.DataFrame, market_reference: pd.Series | None = None) -> pd.DataFrame:
        self._check_fitted()
        out = df.drop(columns=["quote_signal"], errors="ignore").copy()
        out["weight"] = out["weight"].abs().fillna(self.weight_median)
        return fill_by_day(out, "market_index", market_reference)

    def outlier_mask(self, df: pd.DataFrame) -> pd.Series:
        """True for loads whose rate per mile is far from normal for their distance."""
        self._check_fitted()
        rate_per_mile = df["posted_rate"] / df["distance"]
        bands = pd.cut(df["distance"], DISTANCE_BANDS)
        expected = bands.map(self.band_rpm_median).astype(float)
        if expected.isna().any():
            raise ValueError(f"{int(expected.isna().sum())} rows have a missing or invalid distance")
        ratio = rate_per_mile / expected
        return (ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH)


def prepare_december(dec: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, preparer: Preparer) -> pd.DataFrame:
    """December chart rows with coordinates and market_index filled in."""
    out = add_coordinates(dec.drop(columns="predicted_rate"), city_coordinates(train))
    return preparer.transform(out, market_reference=daily_median(val, "market_index"))


if __name__ == "__main__":
    train, val, dec = load_data()
    preparer = Preparer().fit(train)

    outliers = preparer.outlier_mask(train)
    train_ready = preparer.transform(train[~outliers])
    val_ready = preparer.transform(val)
    dec_ready = prepare_december(dec, train, val, preparer)

    print(f"train: {len(train)} rows, {outliers.sum()} outliers removed -> {len(train_ready)}")
    print(f"weight median used for filling: {preparer.weight_median:.0f}")
    for name, df in [("train", train_ready), ("validation", val_ready), ("december", dec_ready)]:
        print(f"{name:<10} missing values: {int(df.isna().sum().sum())} | negative weight: {int((df['weight'] < 0).sum())} | quote_signal present: {'quote_signal' in df}")
    print(dec_ready[["date", "pickup", "delivery", "pickup_lat", "delivery_lat", "market_index"]].head(3).to_string(index=False))
