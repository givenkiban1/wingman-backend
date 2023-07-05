# imports
import pandas as pd
import numpy as np

from ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from ml_logic.preprocessor import preprocess_features
from ml_logic.model import train_test, initialize_model, train_model, evaluate_model
from ml_logic.registry import save_results, save_model, load_model

# functions
def preprocess():
    '''
    uses the following functions:
    data.get_data_with_cache()
    data.clean_data()
    preprocessor.preprocess_features()
    data.load_data_to_bq()
    '''
    pass # collects, cleans, and preprocceses data

def train():
    '''
    uses the following functions:
    data.get_data_with_cache()
    registry.load_model()
    model.initalize_model()
    model.complie_model()
    model.train_model()
    registry.save_results()
    registry.save_model()
    '''
    pass # returns validation metrics

def evaluate():
    '''
    uses the following functions:
    registry.load_model()
    data.get_data_with_cache()
    model.evaluate_model()
    registry.save_results()
    '''
    pass # returns evaluation metrics

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    '''
    uses the following functions:
    registry.load_model()
    preprocessor.preprocess_features()
    '''
    pass # returns y_pred

if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    pred()
