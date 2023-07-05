# imports
import pandas as pd
import numpy as np

from ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from ml_logic.preprocessor import preprocess_features
from ml_logic.model import train_test, initialize_model, train_model, evaluate_model
from ml_logic.registry import save_results, save_model, load_model

from params import *
from pathlib import Path


query = f"""SELECT * FROM {GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"""
data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath('preclean', f"query_{DATA_SIZE}.csv")

# functions
def preprocess(query, cache_path, table):
    '''
    uses the following functions:
    data.get_data_with_cache()
    data.clean_data()
    preprocessor.preprocess_features()
    data.load_data_to_bq()
    '''
    data = get_data_with_cache(query, cache_path)
    data_clean = clean_data(data)
    data_preproc = preprocess_features(data_clean)
    load_data_to_bq(data_preproc, table)
    return data_preproc


def train_evaluate (query, cache_path, stage='Production'):
    data = get_data_with_cache(query, cache_path)
    X_train, X_test, y_train, y_test = train_test(data)
    model = load_model(stage)
    model = train_model(model, X_train, y_train)
    accuracy_score = evaluate_model(model, X_test, y_test)
    params = dict(context='evaluate')
    metrics=dict(accuracy=accuracy_score)
    save_results(params, metrics)
    save_model(model)
    return accuracy_score

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    '''
    uses the following functions:
    registry.load_model()
    preprocessor.preprocess_features()
    '''
    model=load_model()
    X_clean = clean_data(X_pred)
    X_preproc = preprocess_features(X_clean)

    def y_pred_top(model, X_pred):
        probabilities = model.predict_proba(X_pred)
        # Find the indices of the top three classes with highest probabilities
        top_classes_indices = np.argsort(-probabilities, axis=1)[:, :1]
        # Get the class labels corresponding to the top three classes
        top_classes = model.classes_[top_classes_indices]
        # Print the three most likely classes for each prediction
        count = 0
        for i, classes in enumerate(top_classes):
            if count < 1:
                print(f"Prediction {i+1}: {classes}")
                count += 1

    y_pred = y_pred_top(model, X_pred)
    return y_pred

if __name__ == '__main__':
    preprocess()
    train_evaluate()
    pred()
