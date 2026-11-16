# To Activate the Venv:

Within the top level \Backtester-Data-Pipeline call: .\\scripts\activate

# To deactivate the Venv:

Within the top level \Backtester-Data-Pipeline call: deactivate

## Daily data contract (stage 1)

Each row represents one asset's daily trading observation. `date` is the
feature observation date, parsed as a UTC timestamp; features are available
after that day's bar completes. It is a daily bar label, not an intraday
execution timestamp. The CLI frequency must be `daily`.

`data_pipeline/contracts.py` copies and validates observations before feature
and target generation. Each history must contain one asset, non-missing dates,
at most one row per UTC date, and ascending dates. It resets the index so
features and targets align by row. Process assets separately before combining
their model datasets. Sorting and duplicate resolution remain cleaning's job.

The feature output retains `asset` and `date`. Target generation adds
`target_entry_date` (the next observed trading date) and `target_exit_date`
(the second subsequent trading date). These follow observations, not calendar
days. The target becomes observable at the exit day's open. Missing future
observations remain missing; completeness filtering is described in stage 3,
and label-availability enforcement remains stage 4. Metadata is excluded from numeric model
inputs by the existing explicit feature list.

Run the contract checks with:
`.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v`

## Cached data processing (stage 2)

`run_pipeline` in `main.py` fetches only when the raw CSV is absent. Both fresh
and cached runs then follow the same path:

`raw cache -> combine chunks -> normalise -> clean globally -> validate contract -> features -> targets`

Normalisation selects columns by name. Cleaning sorts the complete history and
keeps the first valid occurrence of each timestamp; stable sorting preserves
raw-file order when duplicate timestamps have conflicting values. Derived
normalised, cleaned, and feature CSVs are replaced on each run, so reruns do not
append duplicate observations. Raw cached data is reused without modification.
The cleaned CSV stores cleaned prices; asset metadata is attached at the next
contract boundary and retained in the feature/target CSV.

Example using an existing cache:
`.\.venv\Scripts\python.exe -B main.py spy 1999-01-03 2017-12-31 daily`

## Feature and target eligibility (stage 3)

`features/schema.py` defines the 20 ordered numeric inputs shared by feature
generation and preprocessing. `atr14_norm` is ATR14 divided by adjusted close,
so it is dimensionless. Unavailable indicators on short histories remain NaN.

Generate features and targets from the complete available history before
selecting training dates. The first 199 observations supply the SMA200 warm-up;
earlier requested data (for example, 1999 before training starts in 2000) is
retained for this purpose. If the downloaded history is insufficient, the
pipeline excludes the affected rows; it does not fetch additional history
automatically or manufacture missing values.

`training/row_eligibility.py` checks required numeric columns and target dates,
then separates rows into:

- `data/features`: the complete feature/target history, including missing values.
- `data/model_ready`: rows after warm-up with all 20 finite features, a finite
  target, and known entry/exit dates. Use this output for supervised training.
- `data/excluded`: remaining rows with `exclusion_reason` values: `warmup`,
  `invalid_features`, and/or `incomplete_target`.

The final two observations normally have incomplete targets. Features and
targets are calculated before filtering, so exclusions cannot shift the
meaning of a target. Model-ready means numerically complete, not available at
a particular historical cutoff; the stage 4 splitter enforces label availability.
The complete feature history remains available for later sequence context and
future predictions whose targets are not yet known.

## Label availability and walk-forward boundaries (stage 4)

Pass the stage 3 `model_ready` output to `DataSplitter.split`. Dates loaded
from CSV are parsed as UTC. Each fit occurs after the last daily bar before
the next prediction window starts. Training requires both `date` and
`target_exit_date` to be strictly before that window start. This removes
trailing rows whose labels were not yet observable, even though their features
were available. No fixed calendar-day gap is assumed: actual target dates
handle weekends and missing trading observations.

Expanding training starts in 2000; fixed windows use the requested years of
history, bounded below by 2000. Tuning predictions cover 2018--2021. Tuning
labels must exit before 2022, so selection cannot use test-period prices.
Labels may cross internal validation windows; the following fit independently
checks their availability. `get_test_set` selects 2022--2025 rows with target
exits before 2026. It does not implement test-period retraining.

Window lengths must be positive integers. `step_months` must equal
`validation_months`, ensuring contiguous, non-overlapping prediction periods.
The final period is shortened to the 2022 boundary when necessary. Empty
training or validation windows raise an error rather than silently skipping
missing coverage. The current local SPY cache ends in 2017, so additional
2018--2025 data is needed for an actual validation/test run.

Example configuration:
`WindowConfig(mode="fixed", train_years=5, validation_months=3, step_months=3)`

## Preprocessing and dataset construction (stage 5)

`training/preprocessing.py` owns a `StandardScaler` fitted on the eligible
training rows for one window. Keep that `Preprocessor` instance alongside the
eventual fitted model. Transforming validation, test, or context never refits
the scaler. `Preprocessor` can still be imported from `training.data_splitter`.

`standardise_input(train, validation)` now returns two `PreparedDataset`
containers instead of four arrays. Each container has `X`, `y`, `metadata`, and
`feature_names`. Metadata carries UTC observation/target dates and asset;
metadata never enters the numeric feature array. `y` is `None` for feature-only
context or future prediction rows transformed with `require_target=False`.

```python
from training.preprocessing import Preprocessor
from training.dataset_gen import DatasetGenerator

preprocessor = Preprocessor()
train_data, validation_data = preprocessor.standardise_input(train, validation)
builder = DatasetGenerator()
tabular_train = builder.build_tabular(train_data)
sequence_train = builder.build_sequences(train_data, sequence_length=20)

# full_features is the complete feature history, not just model_ready rows.
# This includes recent rows with unknown labels but already known features.
history = preprocessor.transform(full_features, require_target=False)
sequence_validation = builder.build_sequences(
    validation_data, sequence_length=20, history=history
)
```

Sequences contain only observations up to and including their endpoint, and
use that endpoint's target and metadata. Explicit history preserves the first
validation prediction when enough context exists. It must be the complete
observed feature history, transformed using the same preprocessor, with the
same asset and feature order. Do not use label-filtered rows as context: that
would omit recent observed bars. Undefined context features cause an error,
rather than silently bridging those rows. With no explicit history, the first
`sequence_length - 1` training endpoints are omitted. Insufficient context,
invalid lengths, mismatched arrays, and mixed assets are rejected. No model is
fitted in this stage.

## Model training and walk-forward orchestration (stage 6)

`ModelTrainer` accepts a model exposing `fit(X, y)` and `predict(X)`, a
`TrainingConfig`, and prepared datasets. Tabular inputs are 2D; sequence inputs
are 3D with the configured sequence length. Predictions must be one finite
return per endpoint. Framework-specific sequence models can be wrapped in an
adapter implementing this interface; no LSTM or Transformer is included yet.

`WalkForwardRunner` runs one configuration over the stage 4 tuning windows.
For each window it selects eligible history, fits a new preprocessor, builds
datasets, creates and fits a fresh model, and predicts the validation dates.
The default model is Ridge regression with alpha=1. The runner also records a
zero-return baseline (`baseline_prediction=0`), which requires no fitting.
No hyperparameter search or final test evaluation occurs in this stage.

```python
import pandas as pd
from sklearn.linear_model import Ridge
from training.data_splitter import WindowConfig
from training.walk_forward import WalkForwardRunner

ready = pd.read_csv("data/model_ready/your_dataset.csv")
result = WalkForwardRunner().run(
    ready,
    WindowConfig("fixed", train_years=5, validation_months=3, step_months=3),
    model_factory=lambda: Ridge(alpha=1.0),
)
predictions = result.predictions
# Keep each scaler and trained model together for subsequent predictions.
first_window = result.windows[0]
```

`result.predictions` includes asset, feature date, target entry/exit dates,
predicted return, actual return, zero-return baseline, window ID, and
`fit_before` (an exclusive daily cutoff). `result.windows` retains the fitted
preprocessor and trainer, training sample count, training endpoint range, and
last training target exit. Both configuration objects are retained on the
result. Fitted objects are kept in memory; the CLI saves the prediction CSV.

Sequence runs require `TrainingConfig("sequence", sequence_length=20)`, a
model factory accepting 3D inputs, and `full_features=...`. Training sequences
stay within the selected training window. Both training and validation use
the full observed feature history for context, so label filtering cannot
silently compress time. Early training endpoints lacking context are omitted;
undefined context features raise an error. Validation retains its first date
when enough context is available.

From the repository root, run a single Ridge configuration with:

```powershell
.\.venv\Scripts\python.exe -B -m training.run_training data/model_ready/your_dataset.csv data/predictions/ridge.csv --mode fixed --train-years 5 --months 3 --alpha 1.0
```

This command requires data covering 2018--2021 and the preceding training
history. The existing local SPY cache ends in 2017, so it cannot run these
research windows yet. Training execution is verified using synthetic histories;
no predictive-performance claims follow from those checks.

## Configuration selection and final evaluation (stage 7)

See [RUNNING_MODELS.md](RUNNING_MODELS.md) for SPY data requirements, commands
for different models, candidate configuration, and the complete selection/test
workflow. The new CLI has separate `select` and `test` commands:

```powershell
.\.venv\Scripts\python.exe -B -m training.run_experiment select data/model_ready/spy_1999-01-03_2025-12-31.csv training/candidates.example.json data/experiments/spy_validation --metric mae
.\.venv\Scripts\python.exe -B -m training.run_experiment test data/model_ready/spy_1999-01-03_2025-12-31.csv data/experiments/spy_validation/selected.json data/experiments/spy_test
```

Selection pools validation predictions from 2018--2021 and freezes the winning
model parameters and window settings in `selected.json`. Test observations and
outcomes crossing into 2022 are excluded from selection. The final test reads
that snapshot and retrains at its scheduled 2022--2025 cutoffs using only known
labels; it does not tune settings. MAE, RMSE, and sign-based directional
accuracy are reported overall and per window, alongside the zero-return
baseline. Trading performance is deferred until execution/cost assumptions
are defined. Built-in selection supports tabular models; custom sequence
models continue to use the runner API.

Use daily SPY prices from 1999 through 2025-12-31 for the common full-history
experiment. Reuse the existing 1999--2017 raw cache and add 2018--2025; combine
raw history before regenerating features. **2026 prices are not required**:
test targets must exit before 2026, so the final two 2025 feature observations
are not scored. The guide explains how to merge the caches.

1. Target Generation:
    1. Predict next opening RETURN:
        
        $$
        y_t = \frac{O_{t+2} - O_{t+1}}{O_{t+1}}
        $$
        
        - Predicting the return from tomorrow's open to the following day's open, based on information available today.
        - Contract:
            
            $$
            f(X_t) = \hat r_{t+1 -> t+2}
            $$
            
2. Temporal Data Split:
    1. Preserve original boundaries
    2. Find best rolling window config in 2018-2021, then apply that config to OOS test in 2022-2025
3. Preprocessing and Scaling:
    1. Need to standardise features as they have different numerical distributions and models can be sensitive to feature scale
    2. Common:
        
        $$
        z = \frac{x - \mu_{train}}{\sigma_{train}}
        $$
        
        CRUCIAL:
        
        - Fit the scaler only on the current training window, then apply that fitted scaler to the corresponding prediction data. Otherwise, future information can leak into training
4. Dataset construction:
    1. How existing feature vectors are presented to different ML models
        1. 1) Tabular (2D) → Samples x Features
        2. 2) Sequence Models (3D) - Samples x time x Features → LSTM, Transformer

1. Model Training:
    1. Accept a Model, prepared dataset and training config
    2. Rolling window selection MUST occur before fitting:
        1. Orchestrator chooses historical observations, preps training dataset, fits model and advances to next retraining date
2. Hyperparameter Selection:
    - The process is:
        1. Define candidate configurations.
        2. Run walk-forward validation during 2018–2021.
        3. Evaluate the predictions and resulting trading performance.
        4. Select the configuration using predefined criteria.
        5. Freeze the selected configuration for 2022–2025.
3. OOS Prediction + Eval:
    1. Output:
        1. Predicted Value → Expected Return
        2. Actual Value → Realised Return
        3. Prediction date → Trading Day
        4. Asset → Stock Identifier
        5. Optional Uncertainty → Prediction Interval
    2. ML Pipeline evals predictive quality using metrics like MAE, RMSE and directional Accuracy
