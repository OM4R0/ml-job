# Freight Rate Prediction

Predicts the posted rate (total price in USD) of a truckload shipment from its lane, distance, equipment, weight, date and a daily market index. Built for the Spotter Machine Learning Engineer assessment.

## Results

Validated with 3 expanding-window time folds on `train_test.csv` (each fold is tested on the 2 months after its training data, which mimics predicting Nov-Dec after Oct). Errors are on held-out months the model never saw.

| Model | MAE, normal loads | MAPE, normal loads | Within 5%, normal loads | MAE, all loads | MAPE, all loads |
|---|---|---|---|---|---|
| Baseline (median rate per mile by distance band and equipment) | $106.6 (+-33.5) | 4.7% | 60.0% | $159.8 | 7.0% |
| **HistGradientBoosting (final)** | **$60.5 (+-2.1)** | **2.5%** | **89.1%** | **$114.3** | **4.9%** |

Per fold, normal loads: May-Jun $62.8, Jul-Aug $58.7, Sep-Oct $60.1. A typical load costs about $2,380.
"Within 5%" is the share of loads whose predicted price is within 5% of the true price (87.9% on all loads).
"Normal loads" excludes the 1.4% of rows whose price is far from normal for their distance (see Data quality). They are kept in the "all loads" columns because the hidden test set probably contains similar rows.

## How to run

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The assessment data is not included in this repo. Place the provided files in `data/`:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

Then, from the project root:

```bash
python src/train.py      # cross-validation, final model, predictions (about 10 seconds)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`train.py` writes:

| File | Content |
|---|---|
| `validation_predictions.csv` | 12,000 predictions (`load_id,predicted_rate`) |
| `data/december_chart_inputs.csv` | December rows with `predicted_rate` filled (read by `score.py`) |
| `outputs/december_predictions.csv` | Copy of the December predictions |
| `outputs/cv_results.csv` | Error per fold, baseline vs model |
| `outputs/feature_importance.csv` | Permutation importance of the final model |

`score.py` validates both files and creates `scorer_results/candidate_december.png`.

Optional:

```bash
python src/experiments.py   # feature and model comparisons behind the decisions (about a minute)
python src/tune.py          # hyperparameter grid search, 81 settings (several minutes)
```

## Project structure

```
notebooks/01_eda.ipynb   Exploratory analysis and the decisions log (13 decisions with evidence)
src/prepare.py           Data cleaning, fit/transform so statistics come from the training part only
src/features.py          Model features
src/train.py             Cross-validation, final model, predictions
src/experiments.py       Comparisons of time features and model families
src/tune.py              Hyperparameter search
outputs/                 Results written by the scripts
scorer_results/          December chart from score.py
```

## Approach

### Data quality

| Issue | Finding | Treatment |
|---|---|---|
| Missing `weight` (0.6%) and `market_index` (0.8%) | Missing rows match the rest on price per mile, distance, equipment and month (random) | `weight`: training median. `market_index`: median of the same day (it varies between days, not within a day) |
| Negative `weight` (0.6%) | Same magnitude distribution and price as positive weights: only the sign is wrong | `abs()` |
| Price outliers (1.4%) | Rate per mile 2-6x or 0.2-0.5x the normal for the distance, separated from normal loads by an empty gap; no pattern in any feature | Removed from training only |
| `quote_signal` | Equals the rate per mile in some months and its mirror image in others, so it nearly contains the answer in training; in Nov-Dec it shows no link to price | Dropped |
| Validation set | About twice the missing and negative rates of train | Same rules applied at prediction time |
| December chart file | No coordinates or `market_index` | Coordinates from a city lookup; `market_index` from validation rows of the same day |

### Validation

Time-based, never random: the task is to predict the months after the data ends. Three expanding folds (train Jan-Apr / test May-Jun, Jan-Jun / Jul-Aug, Jan-Aug / Sep-Oct). Every statistic used for cleaning is learned on the training part of each fold. The final model is retrained on all Jan-Oct data.

### Features and target

Distance, weight, equipment, pickup and delivery coordinates (8 validation cities never appear in train, so coordinates replace city names), `market_index` and month. The model predicts log(rate per mile); the price is the prediction times the distance. Adding month gave the lowest and most stable error among the time options compared (`src/experiments.py`).

### Model

Compared on the same folds (MAE on normal loads): decision tree $80, random forest $65, linear regression $64 (unstable across folds), LightGBM $64, XGBoost $64, HistGradientBoosting $62 (share of loads within 5% of the true price: 77% to 88.5%, highest for HistGradientBoosting). HistGradientBoosting was the most accurate and most stable and needs only scikit-learn. A grid search over 81 settings (`src/tune.py`) improved it to $60.5; every setting landed between $60.5 and $64.0, and the best ones used more regularization.

Feature importance (permutation): distance 46%, equipment 24%, weight 9%, market_index 7%, month 6%, coordinates 8%.

### December chart

For the fixed Lexington to Fort Wayne load, predictions stay between about $825 and $848, with a weekly pattern (highest on Thursdays, lowest on Sundays) that follows `market_index`.

## Limitations

- Hyperparameters were selected on the same folds used for reporting, so the tuned score is slightly optimistic.
- Nov-Dec are outside the training months; the model carries forward the latest level it learned and relies on `market_index` for day-to-day changes.
- The price outliers cannot be predicted from the available features, which is why the error on all loads is about twice the error on normal loads.
