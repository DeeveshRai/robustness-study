from data_pipeline.contracts import prepare_daily_observations


class TargetGenerator:
    def __init__(self, data, features):
        self.data = prepare_daily_observations(data)
        self.features = prepare_daily_observations(features)
        if not self.data[["asset", "date"]].equals(self.features[["asset", "date"]]):
            raise ValueError("Features and prices must have identical asset/date rows")

    def calculate_next_open_to_open_return(self):
        self.features["target_entry_date"] = self.data["date"].shift(-1)
        self.features["target_exit_date"] = self.data["date"].shift(-2)
        self.features["target_return"] = (self.data["adjOpen"].shift(-2) - self.data["adjOpen"].shift(-1)) / self.data["adjOpen"].shift(-1)
        return self.features
