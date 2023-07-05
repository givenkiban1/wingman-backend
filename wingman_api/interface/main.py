import numpy as np
import pandas as pd
import sys

from pathlib import Path
from colorama import Fore, Style

from wingman_api.params import *
from wingman_api.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from wingman_api.ml_logic.model import initialize_model, fit_model
from wingman_api.ml_logic.preprocessor import preprocess_features
from wingman_api.ml_logic.registry import load_model, save_model, save_results
from wingman_api.ml_logic.registry import mlflow_run, mlflow_transition_model


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    # if X_pred is None:
    #     X_pred = pd.DataFrame(dict(
    #     pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
    #     pickup_longitude=[-73.950655],
    #     pickup_latitude=[40.783282],
    #     dropoff_longitude=[-73.984365],
    #     dropoff_latitude=[40.769802],
    #     passenger_count=[1],
    # ))

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred
