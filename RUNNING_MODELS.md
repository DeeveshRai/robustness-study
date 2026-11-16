# Running models and experiments

Run these commands in PowerShell from the repository root. They use the
existing virtual environment. The API key for fetching data belongs in `.env`
as `API_KEY`; training from generated CSVs makes no API calls.

## 1. Which SPY data is needed?

For the full dissertation schedule, use **daily SPY observations from 1999
through 31 December 2025**, including adjusted OHLC and adjusted volume.

| Range | Purpose |
|---|---|
| 1999 | Indicator warm-up before training starts in 2000 |
| 2000--2017 | Initial historical training pool |
| 2018--2021 | Walk-forward configuration selection |
| 2022--2025 | Final walk-forward test with frozen settings |

You do **not** need data through 2026 for the current experiment. The target
uses the next two observed opens. The test requires its exit before
`2026-01-01`, so the last two observed 2025 feature rows are excluded from
scoring. The last scored feature date is the third-last observed trading date
of 2025. Its target exits on the last observed trading date of 2025.

Scoring predictions for *every* feature date through the final trading day of
2025 would require the first two trading observations of 2026 **and a change
to the declared outcome cutoff**. Merely downloading 2026 data does not change
the current policy. Do not change this policy after inspecting test results.

For validation alone, data through the final trading day of 2021 is sufficient;
the same two-observation end rule applies. A smaller fixed training window
needs only its preceding history plus indicator warm-up, but 1999--2025 is the
simple common dataset for comparing fixed and expanding windows from 2000.

### Reuse your existing cache

The existing raw file is `data/raw/spy_1999-01-03_2017-12-31.csv`.
The **missing range is 2018-01-01 through 2025-12-31**. Fetch it once:

```powershell
.\.venv\Scripts\python.exe -B main.py spy 2018-01-01 2025-12-31 daily
```

That command generates outputs for the new range too, but use the combined
history for the dissertation so indicators and targets cross 2017/2018
correctly. Combine the two **raw** caches, then regenerate all derived data:

```powershell
@'
from pathlib import Path
import pandas as pd

destination = Path("data/raw/spy_1999-01-03_2025-12-31.csv")
if not destination.exists():
    paths = [
        "data/raw/spy_1999-01-03_2017-12-31.csv",
        "data/raw/spy_2018-01-01_2025-12-31.csv",
    ]
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    combined.to_csv(destination, index=False)
    print("Created combined raw cache:", len(combined), "rows")
else:
    print("Combined raw cache already exists; reusing it")
'@ | .\.venv\Scripts\python.exe -B -

.\.venv\Scripts\python.exe -B main.py spy 1999-01-03 2025-12-31 daily
```

Alternatively, without an existing cache, fetch the complete range directly:

```powershell
.\.venv\Scripts\python.exe -B main.py spy 1999-01-03 2025-12-31 daily
```

The pipeline caches by symbol and requested dates. Separate date-range caches
are not merged automatically. For requests divided into smaller chunks, add
their raw file paths to the merge list. The common processing path sorts and
deduplicates the combined history. Do not concatenate precomputed feature
CSVs: their boundary indicators and targets would be wrong.

Only combine raw ranges with compatible schemas and adjustment bases. If older
and newer caches use different adjusted-price bases, obtain a consistent full
snapshot before calculating returns across their join.

The commands below assume these outputs:

- Supervised rows: `data/model_ready/spy_1999-01-03_2025-12-31.csv`.
- Full feature/context history: `data/features/spy_1999-01-03_2025-12-31.csv`.

## 2. Run one model on validation

These commands fit fresh models during 2018--2021 and write validation
predictions plus an adjacent `_metrics.csv`. They do not run the final test or
select hyperparameters. The output directory is created automatically.

```powershell
# Ridge (L2 regularised linear regression)
.\.venv\Scripts\python.exe -B -m training.run_training data/model_ready/spy_1999-01-03_2025-12-31.csv data/predictions/ridge.csv --model ridge --mode fixed --train-years 5 --months 3 --alpha 1.0

# Ordinary linear regression
.\.venv\Scripts\python.exe -B -m training.run_training data/model_ready/spy_1999-01-03_2025-12-31.csv data/predictions/linear.csv --model linear --mode expanding --months 12

# Elastic Net (L1/L2 regularisation)
.\.venv\Scripts\python.exe -B -m training.run_training data/model_ready/spy_1999-01-03_2025-12-31.csv data/predictions/elastic_net.csv --model elastic_net --mode fixed --train-years 5 --months 3 --parameters-file training/model_parameters/elastic_net.json

# Random forest
.\.venv\Scripts\python.exe -B -m training.run_training data/model_ready/spy_1999-01-03_2025-12-31.csv data/predictions/random_forest.csv --model random_forest --mode fixed --train-years 5 --months 3 --parameters-file training/model_parameters/random_forest.json

# Histogram gradient boosting
.\.venv\Scripts\python.exe -B -m training.run_training data/model_ready/spy_1999-01-03_2025-12-31.csv data/predictions/boosting.csv --model hist_gradient_boosting --mode fixed --train-years 5 --months 3 --parameters-file training/model_parameters/hist_gradient_boosting.json
```

These are runnable starting configurations, not claims about which model is
best. `--months` specifies both the prediction-window length and retraining
step. Omit `--train-years` for expanding windows. Stochastic built-in models
default to `random_state=42`; specify seeds explicitly for research records.
The model registry is in `training/models.py`. All built-in models are tabular
regressors. `--alpha` is Ridge-only shorthand. Edit the JSON files under
`training/model_parameters` to set constructor parameters. `--parameters-file`
avoids Windows PowerShell's removal of quotes in inline JSON arguments;
`--parameters` is also available in shells that preserve JSON quoting.

## 3. Select a configuration using validation only

Copy or edit `training/candidates.example.json` **before** final testing. Each
candidate declares its name, model parameters, training-window type, history
length, and retraining schedule. Candidate names must be unique. The example
compares Ridge, Elastic Net, and a random forest; add other registered models
using the same format.

```powershell
.\.venv\Scripts\python.exe -B -m training.run_experiment select data/model_ready/spy_1999-01-03_2025-12-31.csv training/candidates.example.json data/experiments/spy_validation --metric mae
```

Choose the metric in advance: `mae` and `rmse` are minimised;
`directional_accuracy` is maximised. Overall scores pool every validation
prediction, rather than averaging folds of different sizes. Ties break by
candidate name alphabetically. All candidates must produce the same dated
observations. Test observations and labels crossing into 2022 are excluded
before candidate fitting or scoring.

Outputs:

- `leaderboard.csv`: candidate scores, sorted using the declared metric.
- `candidates.json`: the candidate configurations used.
- `validation_predictions.csv`: dated predictions for every candidate.
- `validation_metrics.csv`: overall and per-window model/baseline metrics.
- `selected.json`: the selected model, parameters, window configuration,
  metric, validation score, and declared validation period.

`selected.json` is the frozen configuration snapshot. Keep it unchanged for
the final test. Freeze the *settings*, not the fitted weights: fitting new
weights at scheduled test cutoffs is part of the declared walk-forward method.

## 4. Run the final 2022--2025 test

```powershell
.\.venv\Scripts\python.exe -B -m training.run_experiment test data/model_ready/spy_1999-01-03_2025-12-31.csv data/experiments/spy_validation/selected.json data/experiments/spy_test
```

This command loads the selected settings and does no configuration search.
Each test window gets a fresh model and training-only scaler. Earlier test
observations can enter a later training window only after their labels become
known. Every fit requires `target_exit_date < fit_before`.

Outputs are `test_predictions.csv`, `test_metrics.csv`, `test_windows.csv`,
and a copy of `selected.json`. Fitted models/scalers remain in memory when
using the Python API; this CLI saves predictions and audit records, not model
binaries. Do not retune settings using the final test report.

## 5. Interpreting the metrics

MAE and RMSE use decimal returns: an error of `0.01` is one percentage point.
Directional accuracy is exact agreement of `sign(prediction)` and
`sign(actual)`, with zero treated as its own direction. Consequently the
zero-return baseline only gets direction correct when the actual return is
exactly zero; it is mainly a return-error baseline, not a directional trading
strategy. Each report includes both model and zero-return metrics, overall
and per retraining window.

No transaction costs, position sizing, trading returns, Sharpe ratio, or
uncertainty intervals are implemented. Predictive metrics alone do not establish
profitability. Define execution and cost assumptions before adding trading
evaluation; those assumptions must not be chosen from the final test results.

## 6. Custom models and sequence adapters

For another tabular model, use the Python runner with a factory returning a
fresh estimator on every call:

```python
import pandas as pd
from sklearn.svm import SVR
from training.data_splitter import WindowConfig
from training.walk_forward import WalkForwardRunner

ready = pd.read_csv("data/model_ready/spy_1999-01-03_2025-12-31.csv")
result = WalkForwardRunner().run(
    ready, WindowConfig("fixed", 5, 3, 3),
    model_factory=lambda: SVR(C=1.0, epsilon=0.001),
)
```

For an LSTM or Transformer, supply your own adapter whose `fit(X, y)` accepts
`samples x time x features` and whose `predict(X)` returns a 1D return vector:

```python
from training.model_training import TrainingConfig

full = pd.read_csv("data/features/spy_1999-01-03_2025-12-31.csv")
# MySequenceAdapter must be implemented separately; it is not supplied here.
result = WalkForwardRunner().run(
    ready, WindowConfig("fixed", 5, 3, 3),
    training_config=TrainingConfig("sequence", sequence_length=20),
    model_factory=lambda: MySequenceAdapter(),
    full_features=full,
)
```

The built-in selection CLI currently supports registered **tabular** models.
Custom factories and sequence adapters use the runner API. Extending automatic
selection to a custom model requires registering its construction/configuration
first. Neural-network architectures and their framework dependencies are not
implemented by this pipeline.

## 7. Checks

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Synthetic-history tests verify chronology, fitting, configuration selection,
and the CLI. They do not measure market predictive performance. All runs expect
one asset per input file; the asset column keeps outputs identifiable.
