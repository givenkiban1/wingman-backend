# imports
from tensorflow import keras
import mlflow
from params import *

# functions

def save_results(params: dict, metrics: dict) -> None:
    pass # saves results to MLflow or locally

def save_model(model: keras.Model = None) -> None:
    pass # saves model to MLflow or locally

def load_model(stage="Production") -> keras.Model:
    pass # returns model from MLflow

def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    pass # transitions model from one stage to another, returns None

def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper
