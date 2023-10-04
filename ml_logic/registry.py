import params
import pandas as pd
import numpy as np
import time
import pickle
import glob

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from tensorflow import keras

import mlflow
from mlflow.tracking import MlflowClient

from google.cloud import bigquery, storage
from params import *

from google.auth.exceptions import DefaultCredentialsError
from google.cloud.exceptions import NotFound

def save_model(model) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    folder_path = os.path.join(LOCAL_REGISTRY_PATH, "models")
    if os.path.isdir(folder_path):
        model_path = os.path.join(folder_path, f"{timestamp}.h5")
        model.save(model_path)
    else:
        os.mkdir(folder_path)
        model_path = os.path.join(folder_path, f"{timestamp}.h5")
        model.save(model_path)

    print("✅ Model saved to local machine")

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        folder_path = os.path.join(LOCAL_REGISTRY_PATH, "params")
        if os.path.isdir(folder_path):
            params_path = os.path.join(folder_path, timestamp + ".pickle")
            with open(params_path, "wb") as file:
                pickle.dump(params, file)
        else:
            os.mkdir(folder_path)
            params_path = os.path.join(folder_path, timestamp + ".pickle")
            with open(params_path, "wb") as file:
                pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        folder_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics")
        if os.path.isdir(folder_path):
            metrics_path = os.path.join(folder_path, timestamp + ".pickle")
            with open(metrics_path, "wb") as file:
                pickle.dump(metrics, file)
        else:
            os.mkdir(folder_path)
            metrics_path = os.path.join(folder_path, timestamp + ".pickle")
            with open(metrics_path,"wb") as file:
                pickle.dump(metrics, file)

    print("✅ Results saved locally")


def load_model():

    if MODEL_TARGET == "local":

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")
            return None
