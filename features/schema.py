"""Ordered model inputs; metadata and labels are deliberately excluded."""

FEATURE_COLUMNS = [
    'returns_1d',
    'returns_5d',
    'returns_20d',
    'returns_60d',
    'returns_lag1',
    'returns_lag2',
    'sma_20_vs_prices',
    'sma_50_vs_prices',
    'sma_200_vs_prices',
    'macd_norm',
    'rsi_14',
    'volatility_5d',
    'volatility_20d',
    'atr14_norm',
    'bollinger_width20',
    'relative_volume20',
    'volume_change_1d',
    'high_low_range',
    'open_close_return',
    'overnight_gap',
]

# SMA200 needs 200 observations, including the current bar.
FEATURE_WARMUP_ROWS = 199
