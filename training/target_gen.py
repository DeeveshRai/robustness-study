class TargetGenerator:
    def __init__(self, data, features):
        self.data = data
        self.features = features

    def calculate_next_open_to_open_return(self):
        self.features["target_return"] = (self.data["adjOpen"].shift(-2) - self.data["adjOpen"].shift(-1)) / self.data["adjOpen"].shift(-1)
        return self.features