import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from data_pipeline.batching.batching import batches
from data_pipeline.cleaning.cleaning import cleaning
from data_pipeline.contracts import prepare_daily_observations, require_daily_frequency
from data_pipeline.csv_saver import CSVsaver
from data_pipeline.fetcher import Fetcher
from data_pipeline.normaliser import Normaliser
from features.featureEngineer import FeatureEngineer
from training.target_gen import TargetGenerator
from training.row_eligibility import select_model_rows


def create_feature_vector(data_dataframe):
    return FeatureEngineer(data_dataframe).create_feature_dataframe()


def add_target_return(data_dataframe, feature_dataframe):
    return TargetGenerator(
        data_dataframe, feature_dataframe
    ).calculate_next_open_to_open_return()


def run_pipeline(symbol, start, end, freq, api_key=None):
    """Reuse raw data when available, then rebuild all derived datasets."""
    require_daily_frequency(freq)
    file_path = os.path.join("data", "raw", f"{symbol}_{start}_{end}.csv")
    if not os.path.exists(file_path):
        Fetcher(api_key, symbol, start, end, freq).fetch()

    # Combine chunks before cleaning so ordering and uniqueness are global.
    raw = pd.concat(batches(file_path), ignore_index=True)
    normalised = Normaliser(symbol, start, end).normalise(raw)
    cleaned = cleaning(normalised, "cleaned", symbol, start, end)
    observations = prepare_daily_observations(cleaned, symbol)
    features = create_feature_vector(observations)
    result = add_target_return(observations, features)
    model_ready, excluded = select_model_rows(result)
    CSVsaver("features", symbol, start, end).save_as_csv(result)
    CSVsaver("model_ready", symbol, start, end).save_as_csv(model_ready)
    CSVsaver("excluded", symbol, start, end).save_as_csv(excluded)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("freq", choices=["daily"])
    args = parser.parse_args()
    load_dotenv()
    run_pipeline(args.symbol, args.start, args.end, args.freq, os.getenv("API_KEY"))


if __name__ == "__main__":
    main()
