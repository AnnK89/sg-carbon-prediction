import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from ml_logic import data, model_carbon, registry
from ml_logic.registry import mlflow_run,mlflow_transition_model
from params import *

def preprocess() -> None:
    """
    - Query the raw dataset from BigQuery dataset
    - Process query data
    - Store processed data back to BigQuery
    """

    # Query and process the raw dataset from BigQuery dataset
    df = data.combine_clean_data()

    # Store processed data back to BigQuery
    data.BigQueryDataLoader.load_processed(df, truncate=True)

    print("✅ preprocess() done \n")

@mlflow_run
def train(
        learning_rate: float=0.0005,
        patience: int = 5
    ) -> tuple:
    """
    - Download processed data from BigQuery table
    - Train on the preprocessed dataset
    - Store training results and model weights
    """

    # Download processed data from BigQuery table
    df = data.BigQueryDataRetriever().get_processed_from_bq()

    X_train, X_test, y_train, y_test = data.split_train_test_data(df)

    model = registry.load_model()

    if model is None:
        model = model_carbon.initialize_model(input_shape=X_train.shape[1:])

    model = model_carbon.compile_model(model, learning_rate=learning_rate)
    # model = model_carbon.tune_model(X_train, X_test, y_train, y_test)
    model, history = model_carbon.train_model(
        model, X_train, y_train,
        patience=patience,
    )

    mae = np.min(history.history['mae'])
    accuracy = np.max(history.history['accuracy'])

    model_carbon.evaluate_model(model,X_test,y_test)

    registry.save_model(model=model)

    params_dict = dict(
        context="train",
        row_count=len(X_train),
    )

    registry.save_results(params=params_dict, metrics={'mae' : mae, 'accuracy' : accuracy})

    # The latest model should be moved to staging
    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print(f"✅ train() done with mae = {mae} and accuracy = {accuracy} \n")

    return mae, accuracy

def pred() -> Union[List[float], np.ndarray]:
    """
    - Make a prediction using the latest trained model
    """
    # Download processed data from BigQuery table
    df = data.BigQueryDataRetriever().get_processed_from_bq()

    model = registry.load_model()

    num_train_year = TRAIN_END - TRAIN_START + 1
    carbon_data = df.iloc[:, -num_train_year:].values

    X_pred=[]
    for i in range(0,len(carbon_data), len(TRANSFORMER_MAP)):
        X_pred.append(carbon_data[i:i+len(TRANSFORMER_MAP)].T)

    X_pred = np.array(X_pred).astype(np.float32)

    y_pred = model.predict(X_pred)

    # data.BigQueryDataLoader().load_predictions(y_pred, truncate=True)

    print(f"✅ pred() done")

    return y_pred

if __name__ == "__main__":
    preprocess()
    train()
    pred()
