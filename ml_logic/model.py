import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Model, LSTM, Dense, SimpleRNN, Reshape
from tensorflow.keras.callbacks import EarlyStopping

from ml_logic import data

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def preproc_train_data():
    df = data.combine_clean_data()
    num_years = 5
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
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split = int(len(X) *  0.8)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    return X_train,X_test,y_train,y_test


def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(units=200, input_shape=input_shape))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(units=5*4,activation='linear'))
    model.add(Reshape((5,4)))


    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae','accuracy'])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=55,
        patience=5
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(patience=patience, monitor='mae', restore_best_weights=True)

    history = model.fit(
        X,
        y,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
