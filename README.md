# To Activate the Venv:

Within the top level \Backtester-Data-Pipeline call: .\\scripts\activate

# To deactivate the Venv:

Within the top level \Backtester-Data-Pipeline call: deactivate

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