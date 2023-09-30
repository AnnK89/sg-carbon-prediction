import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from google.cloud import bigquery
from params import *
from ml_logic import data

def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query data from BigQuery using `get_data_with_cache` and combined for all parameters
    df = data.clean_combined_data()

    data.load_data_to_bq(
    df,
    gcp_project=PROJECT_ID,
    bq_dataset=DATASET_ID,
    table='processed_df',
    truncate=True
    )

    print("✅ preprocess() done \n")

def train(
        num_years = 5, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> float:

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
    query = f"""
        SELECT *
        FROM {PROJECT_ID}.{DATASET_ID}.processed_df
    """
    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed_df.csv")
    df = data.get_data_with_cache(
        gcp_project=PROJECT_ID,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    # Process data
    num_years = num_years
    # Use previous years except latest years for sequence length
    carbon_data = df.iloc[:, 1:-num_years].values # Exclude the 'City' column
    target_data = df.iloc[:,-num_years:].values # last year as target

    # if sequence_length > carbon_data.shape[1] - 1:
    #     sequence_length = carbon_data.shape[1] - 1

    X = []  # Input features
    y = []  # Target values

    # Create input features and target values
    for i in range(0,len(carbon_data),4): #4 parameters per planning area
        X.append(carbon_data[i:i+4].T)
        y.append(target_data[i:i+4].T)

    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split = int(len(X) *  0.8)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
