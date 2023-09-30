import pandas as pd
import numpy as np
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from interface.main import pred
from ml_logic import data
from params import *

app = FastAPI()
app.state.model = data.load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict( ):
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
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
    carbon_data = df.iloc[:, -12:].values

    X_pred=[]
    for i in range(0,len(carbon_data),4): #4 parameters per planning area
        X_pred.append(carbon_data[i:i+4].T)

    X_pred = np.array(X_pred)

    y_pred = app.state.model.predict(X_pred)

    print(y_pred)
    res={}
    res['5_years_prediction'] = json.dumps(np.array(y_pred).tolist())

    return res

@app.get("/")
def root():
    return {'greeting':[['Hello2','Hello2','Hello2','Hello2'],['Hello2','Hello2','Hello2','Hello2'],['Hello2','Hello2','Hello2','Hello2']]}
