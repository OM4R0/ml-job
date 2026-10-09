"""Hyperparameter search for the gradient boosting model.

Grid search on the same 3 time-based folds as train.py (Decisions log #10).
Each setting is scored by the mean MAE on normal rows (Decisions log #7);
the spread across folds and the MAE on all rows are reported too.

Outputs:
    outputs/tuning_results.csv   every setting, best first
    outputs/best_params.json     the best setting

Run from the project root:  python src/tune.py
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from prepare import Preparer, load_data
from train import FOLDS, OUTPUT_DIR, error_metrics, model_input

# 3 x 3 x 3 x 3 = 81 settings, each trained on 3 folds.
GRID = {
    "learning_rate": [0.03, 0.05, 0.1],     # size of each tree's correction
    "max_leaf_nodes": [15, 31, 63],         # how complex each tree can be
    "min_samples_leaf": [20, 50, 100],      # minimum loads per leaf (bigger = smoother)
    "l2_regularization": [0.0, 0.1, 1.0],   # penalty on extreme leaf values
}
# Early stopping picks the number of trees; 1000 is only an upper limit.
FIXED = {"max_iter": 1000, "early_stopping": True, "random_state": 0}
CURRENT = {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0}


def prepare_folds(train: pd.DataFrame) -> list[dict]:
    """Prepare each fold once, so every setting is trained on identical data."""
    folds = []
    for label, start, end in FOLDS:
        train_part = train[train["date"] < start]
        test_part = train[(train["date"] >= start) & (train["date"] < end)]
        prep = Preparer().fit(train_part)
        fit_rows = prep.transform(train_part[~prep.outlier_mask(train_part)])
        test_rows = prep.transform(test_part)
        folds.append({
            "label": label,
            "X_fit": model_input(fit_rows),
            "y_fit": np.log(fit_rows["posted_rate"] / fit_rows["distance"]),
            "X_test": model_input(test_rows),
            "distance": test_rows["distance"].to_numpy(),
            "y_true": test_rows["posted_rate"].to_numpy(),
            "normal": ~prep.outlier_mask(test_part).to_numpy(),
        })
    return folds


def evaluate(params: dict, folds: list[dict]) -> dict:
    mae_normal, mae_all, trees = [], [], []
    for fold in folds:
        model = HistGradientBoostingRegressor(**FIXED, **params).fit(fold["X_fit"], fold["y_fit"])
        predicted = np.exp(model.predict(fold["X_test"])) * fold["distance"]
        normal = fold["normal"]
        mae_normal.append(error_metrics(fold["y_true"][normal], predicted[normal])[0])
        mae_all.append(error_metrics(fold["y_true"], predicted)[0])
        trees.append(model.n_iter_)
    return {
        **params,
        "mae_normal_mean": float(np.mean(mae_normal)),
        "mae_normal_std": float(np.std(mae_normal)),
        "mae_all_mean": float(np.mean(mae_all)),
        "folds_mae_normal": " / ".join(f"{v:.1f}" for v in mae_normal),
        "trees_used": " / ".join(str(t) for t in trees),
    }


def main() -> None:
    train, _, _ = load_data()
    OUTPUT_DIR.mkdir(exist_ok=True)
    folds = prepare_folds(train)

    settings = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print(f"Testing {len(settings)} settings x {len(folds)} folds...")
    start_time = time.time()
    results = []
    for i, params in enumerate(settings, start=1):
        results.append(evaluate(params, folds))
        if i % 9 == 0 or i == len(settings):
            print(f"  {i}/{len(settings)} done ({time.time() - start_time:.0f}s)")

    table = pd.DataFrame(results).sort_values(["mae_normal_mean", "mae_normal_std"]).reset_index(drop=True)
    table.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nTop 10 settings (MAE in $ per load):")
    print(table.head(10).round(2).to_string())

    current = table[(table[list(CURRENT)] == pd.Series(CURRENT)).all(axis=1)].iloc[0]
    best = table.iloc[0]
    print(f"\nCurrent setting: MAE normal {current['mae_normal_mean']:.2f} (+-{current['mae_normal_std']:.2f}), rank {current.name + 1} of {len(table)}")
    print(f"Best setting:    MAE normal {best['mae_normal_mean']:.2f} (+-{best['mae_normal_std']:.2f})")
    print(f"Improvement:     {current['mae_normal_mean'] - best['mae_normal_mean']:.2f} $ per load")

    best_params = {key: best[key].item() for key in GRID}
    (OUTPUT_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    print(f"\nBest parameters saved to {OUTPUT_DIR / 'best_params.json'}: {best_params}")


if __name__ == "__main__":
    main()
