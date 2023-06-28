# imports
import numpy as np

from tensorflow.keras import Model
from typing import Tuple

# functions

def initialize_model(input_shape: tuple) -> Model:
    pass # returns model

def compile_model(model: Model, learning_rate=0.0005) -> Model:
    pass # returns compiled model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    pass # returns fitted model and history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    pass # returns mectris = model.evaluate()
