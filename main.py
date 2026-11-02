from dotenv import load_dotenv
import os
import pandas as pd
from data_pipeline.fetcher import Fetcher
import argparse
from data_pipeline.batching.batching import batches
from data_pipeline.normaliser import Normaliser
from data_pipeline.cleaning.cleaning import cleaning
from data_pipeline.formatting.formatting import formatting
from features.featureEngineer import FeatureEngineer

def create_feature_vector(data_dataframe, symbol, start, end):
    features = FeatureEngineer(data_dataframe, symbol, start, end)
    features.create_feature_dataframe()
    print("end")
    
#IMPLEMENTING CLI
parser = argparse.ArgumentParser()
parser.add_argument("symbol")
parser.add_argument("start")
parser.add_argument("end")
parser.add_argument("freq")

args = parser.parse_args()

symbol = args.symbol
start = args.start
end = args.end
freq = args.freq

# LOGIC

load_dotenv()
API_KEY = os.getenv('API_KEY')
# API_SECRET_KEY = os.getenv('API_SECRET_KEY')

data_dir = "data/raw"
os.makedirs(data_dir, exist_ok=True)

file_name = f"{symbol}_{start}_{end}.csv"
file_path = os.path.join(data_dir, file_name)
file_exists_raw = os.path.exists(file_path)

if not file_exists_raw:
    fetcher_object = Fetcher(API_KEY, symbol, start, end, freq)
    fetcher_object.fetch()
    print("HAS BEEN CREATED")

    #Only happens if file doesnt exist in cleaned and formatted and storage (shouldnt matter tbf, if it exists in storage, should exist everywhere else)
    normaliser = Normaliser(symbol, start, end)
    data_dataframe = None
    for batch in batches(file_path):
        #normalise
        normalised = normaliser.normalise(batch) # DONE
        #clean
        cleaned = cleaning(normalised, "cleaned", symbol, start, end)
        data_dataframe = pd.concat([data_dataframe, cleaned])
        #format
        #store


    #python main.py "AAPL" "5Min" "2024-01-03T00%3A00%3A00Z" "2024-01-04T00%3A00%3A00Z" "100" "raw" "sip" "asc"
    #If file exists in raw/cleaned/formatted/normalised...

    create_feature_vector(data_dataframe, symbol, start, end)

else:
    file_name = os.path.join("data/normalised", file_name)
    df = pd.read_csv(file_name)
    create_feature_vector(df, symbol, start, end)

