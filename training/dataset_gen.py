import numpy as np

class DatasetGenerator:
    def __init__(self):
        pass

    def build_tabular(self, X, y):
        return X, y #should form a dataset: no of samples(rows) * no of features(cols) 

    def build_sequences(self, X, y, sequence_length):
        X_sequences = []
        y_sequences = []

        for i in range(sequence_length - 1, len(X)):
            start = i - sequence_length + 1

            X_sequences.append(X[start:i + 1])
            y_sequences.append(y[i])

        return (
            np.array(X_sequences),
            np.array(y_sequences)
        )