# imports
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
    preprocessed = preprocess_new_data.submit()

    old_accuracy = evaluate_production_model.submit(wait_for=[preprocessed])
    new_accuracy = re_train.submit(wait_for=[preprocessed])

    old_accuracy = old_accuracy.result()
    new_accuracy = new_accuracy.result()

    if new_accuracy < old_accuracy:
        print(f"🚀 New model replacing old in production with accuracy: {new_accuracy} the Old accuracy was: {old_accuracy}")
        transition_model.submit(current_stage="Staging", new_stage="Production")
    else:
        print(f"🚀 Old model kept in place with accuracy: {old_accuracy}. The new accuracy was: {new_accuracy}")



if __name__ == "__main__":
    train_flow()
