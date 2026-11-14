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
