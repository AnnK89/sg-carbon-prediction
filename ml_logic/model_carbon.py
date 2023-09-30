import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model,optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from ml_logic import data

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


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
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse',
                optimizer=optimizer,
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

    print(f"✅ Model trained on {len(X)} rows with min val mae and accuracy: {round(np.min(history.history['mae']), 2)},{round(np.max(history.history['accuracy']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size = 55
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
        verbose=0,
        batch_size=batch_size,
        # callbacks=None,
        return_dict=True
    )

    accuracy = metrics["accuracy"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)} and Accuracy : {round(accuracy, 2)}")

    return metrics
