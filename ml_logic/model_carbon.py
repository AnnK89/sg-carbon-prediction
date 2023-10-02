import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model,optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Normalization, Dense, SimpleRNN, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor
from ml_logic import data
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    # Define the normalization layer
    normalizer = Normalization()

    # Create the LSTM model
    model = Sequential()
    model.add(normalizer)
    model.add(LSTM(units=200, input_shape=input_shape))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(units=5*4,activation='linear'))
    model.add(Reshape((5,4)))

    return model

def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'accuracy'])

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

    es = EarlyStopping(patience=patience, monitor='mae', restore_best_weights=True)

    history = model.fit(
        X,
        y,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    return model, history


from scikeras.wrappers import KerasRegressor


def tune_model1(X_train, X_test, y_train, y_test):
    model = initialize_model(input_shape=X_train.shape[1:])

    best_model = None
    best_mae = float('inf')
    best_params = {}

    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 2],
        'units': [64, 128, 256],
        'dropout_rate': [0.2, 0.5],
        'batch_size':[20, 50, 25, 32, 100],
        'epochs':[100, 200, 300, 400],
    }

    for learning_rate in param_grid['learning_rate']:
        for units in param_grid['units']:
            for dropout_rate in param_grid['dropout_rate']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        optimizer = LegacyAdam(learning_rate=learning_rate)
                        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'accuracy'])

                        model, history = train_model(model, X_train, y_train)
                        res = evaluate_model(model, X_test, y_test)
                        mae = res["mae"]

                        if mae < best_mae:
                            best_mae = mae
                            best_model = model
                            best_params = {
                                'learning_rate': learning_rate,
                                'units': units,
                                'dropout_rate': dropout_rate,
                                'batch_size': batch_size,
                                'epochs': epochs
                            }

    print(f"✅ Best hyperparameters: {best_params}")
    print(f"✅ Model trained with best hyperparameters has MAE: {best_mae}")

    return best_model

def tune_model(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_test = X_test.reshape(-1, X_test.shape[-1])
    y_train = y_train.reshape(-1, y_train.shape[-1])
    y_test = y_test.reshape(-1, y_test.shape[-1])

    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 2],
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(xgb_regressor, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_xgb_model = grid_search.best_estimator_
    y_pred = best_xgb_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, y_pred)

    best_params = grid_search.best_params_

    print(f"✅ Best hyperparameters: {best_params}")
    print(f"✅ Model trained with best hyperparameters has MAE: {best_mae}")

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    return best_xgb_model

def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size = 55
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        X,
        y,
        verbose=0,
        batch_size=batch_size,
        # callbacks=None,
        return_dict=True
    )

    accuracy = metrics["accuracy"]
    mae = metrics["mae"]

    return metrics


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size = 55
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        X,
        y,
        verbose=0,
        batch_size=batch_size,
        # callbacks=None,
        return_dict=True
    )

    accuracy = metrics["accuracy"]
    mae = metrics["mae"]

    return metrics
