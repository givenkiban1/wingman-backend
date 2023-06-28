# imports
import os

import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

from wingman.interface.main import evaluate, preprocess, train
from wingman.ml_logic.registry import mlflow_transition_model
from wingman.params import *


# functions

@task
def preprocess_new_data():
    return preprocess()

@task
def evaluate_production_model():
    return evaluate()

@task
def re_train():
    return train()

@task
def transition_model():
    return mlflow_transition_model()


@flow()
def train_flow():
    '''
    uses the following functions:
    preprocess_new_data()
    evaluation_production_model()
    re_train()
    transition_model()
    '''
    pass # replaces best model based on added data
