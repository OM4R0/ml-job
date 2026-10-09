"""Model features (Decisions log #11).

Used: distance, weight, equipment, the 4 coordinates, market_index, month.
Not used: quote_signal (#8), city names (8 validation cities never appear in
train), day of week (made validation error worse).
"""

from __future__ import annotations

import pandas as pd

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]
NUMERIC_FEATURES = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
]
FEATURES = NUMERIC_FEATURES + ["equipment", "month"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn prepared rows into the model's input table."""
    X = df[NUMERIC_FEATURES].copy()
    # Fixed categories so the codes are identical at training and prediction time.
    X["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT_TYPES)
    X["month"] = df["date"].dt.month
    return X[FEATURES]
