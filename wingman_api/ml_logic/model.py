import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model() -> Model:
    """
    Initialize the RandomForestClassifier Model with set parameters
    """
    model = RandomForestClassifier(n_estimators=100, random_state=1)

    print("✅ Model initialized")

    return model

def fit_model(model: Model, X: np.ndarray, y: np.ndarray) -> Model:


    model.fit(X, y)

    print("✅ Model fitted")

    return model


def evaluate_model(model: Model, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate the model on the test set
    """
    print('🚨 🚨 🚨' + Fore.BLUE + "\nEvaluating model..." + Style.RESET_ALL)

    base = max(y.value_counts()/len(y))
    print("✅ Random Selection Accuracy: %.2f%%" % (base * 100.0))

    return base
