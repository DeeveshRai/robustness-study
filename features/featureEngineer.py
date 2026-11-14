import pandas as pd
import numpy as np
import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, data):
        self.data = data

        self.features = pd.DataFrame(columns=['date',
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
                                      'target_return'
                                      ]) #add rest later
        
        self.features["date"] = self.data["date"]

    def _add_return_features(self):
        self.features["returns_1d"] = np.log(self.data["adjClose"] / self.data["adjClose"].shift(1))
        self.features["returns_5d"] = np.log(self.data["adjClose"] / self.data["adjClose"].shift(5))
        self.features["returns_20d"] = np.log(self.data["adjClose"] / self.data["adjClose"].shift(20))
        self.features["returns_60d"] = np.log(self.data["adjClose"] / self.data["adjClose"].shift(60))
        self.features["returns_lag1"] = self.features["returns_1d"].shift(1)
        self.features["returns_lag2"] = self.features["returns_1d"].shift(2)

    def _add_trend_features(self):
        self.features["sma_20_vs_prices"] = (self.data["adjClose"]/self.data["adjClose"].rolling(window=20).mean())-1
        self.features["sma_50_vs_prices"] = (self.data["adjClose"]/self.data["adjClose"].rolling(window=50).mean())-1
        self.features["sma_200_vs_prices"] = (self.data["adjClose"]/self.data["adjClose"].rolling(window=200).mean())-1
        self.features["macd_norm"] = (self.data["adjClose"].ewm(span=12, adjust=False).mean() - self.data["adjClose"].ewm(span=26, adjust=False).mean())/self.data["adjClose"]

    def _add_momentum_features(self):
        self.features["rsi_14"] =  ta.rsi(self.data["adjClose"], length=14)

    def _add_volitility_features(self):
        self.features["volatility_5d"] = (
            self.features["returns_1d"]
            .rolling(window=5)
            .std()
            * np.sqrt(252)
            )

        self.features["volatility_20d"] = (
            self.features["returns_1d"]
            .rolling(window=20)
            .std()
            * np.sqrt(252)
        )

        self.features["atr14_norm"] = ta.atr(self.data["adjHigh"], self.data["adjLow"], self.data["adjClose"], length=14)

        #AI
        bbands = ta.bbands(self.data["adjClose"], length=20, std=2)
        bbu_col = next((c for c in bbands.columns if c.startswith("BBU_")), None)
        bbl_col = next((c for c in bbands.columns if c.startswith("BBL_")), None)

        if bbu_col and bbl_col:
            self.features["bollinger_width20"] = (bbands[bbu_col] - bbands[bbl_col]) / self.data["adjClose"]
        else:
            self.features["bollinger_width20"] = np.nan

        # AI END
    def _add_volume_features(self):

        self.features["relative_volume20"] = (self.data["adjVolume"] / self.data["adjVolume"].rolling(window=20).mean())
        self.features["volume_change_1d"] = (self.data["adjVolume"] / self.data["adjVolume"].shift(1)) - 1

    def _add_OHLC_features(self):

        self.features["high_low_range"] = (self.data["adjHigh"] - self.data["adjLow"]) / (self.data["adjClose"])
        self.features["open_close_return"] = (self.data["adjClose"] / self.data["adjOpen"]) - 1
        self.features["overnight_gap"] = (self.data["adjOpen"] / self.data["adjClose"].shift(1)) - 1
        


    def create_feature_dataframe(self):
        self._add_return_features()
        self._add_trend_features()
        self._add_momentum_features()
        self._add_volitility_features()
        self._add_volume_features()
        self._add_OHLC_features()

        return self.features
