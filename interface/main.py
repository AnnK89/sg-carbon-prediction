import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from google.cloud import bigquery
from params import *
from ml_logic import data
from ml_logic import model_carbon

def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query data from BigQuery using `get_data_with_cache` and combined for all parameters
    df = data.combine_clean_data()

    data.load_data_to_bq(
    df,
    gcp_project=PROJECT_ID,
    bq_dataset=DATASET_ID,
    table='processed_df',
    truncate=True
    )

    print("✅ preprocess() done \n")

def train(
        num_years = 5, # predicting 5 years
        learning_rate=0.0005,
        patience = 5
    ) -> float:

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
    query = f"""
        SELECT *
        FROM {PROJECT_ID}.{DATASET_ID}.processed_df
        ORDER BY planning_area ASC
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

    model = data.load_model()

    if model is None:
        model = model_carbon.initialize_model(input_shape=X_train.shape[1:])

    model = model_carbon.compile_model(model, learning_rate=learning_rate)
    model, history = model_carbon.train_model(
        model, X_train, y_train,
        patience=patience,
    )

    mae = np.min(history.history['mae'])
    accuracy = np.max(history.history['accuracy'])

    params = dict(
        context="train",
        row_count=len(X_train),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    data.save_results(params=params, metrics={'mae' : mae, 'accuracy' : accuracy})

    # Save model weight on the hard drive (and optionally on GCS too!)
    data.save_model(model=model)

    # The latest model should be moved to staging
    # data.mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done \n")

    model_carbon.evaluate_model(model,X_test,y_test)

    return mae,accuracy

def pred():
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    query = f"""
        SELECT *
        FROM {PROJECT_ID}.{DATASET_ID}.processed_df
        ORDER BY planning_area ASC
    """
    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed_df.csv")
    df = data.get_data_with_cache(
        gcp_project=PROJECT_ID,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )
    model = data.load_model()

    carbon_data = df.iloc[:, -12:].values

    X_pred=[]
    for i in range(0,len(carbon_data),4): #4 parameters per planning area
        X_pred.append(carbon_data[i:i+4].T)

    X_pred = np.array(X_pred)

    y_pred = model.predict(X_pred)

    print(f"✅ pred() done")

    return y_pred

if __name__ == '__main__':
    preprocess()
    train()
    pred()
