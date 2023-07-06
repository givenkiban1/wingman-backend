from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

from wingman.ml_logic.preprocessor import preprocess_features
from wingman.ml_logic.registry import load_model

api = FastAPI()
api.state.model = pickle.load(open("model.pkl","rb"))

# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "API connected"}


@api.get("/predict")
def predict(csv_input):

    X_input = pd.read_csv(csv_input)
    X_preproc = preprocess_features(X_input)

    y_pred = api.state.model.predict(X_preproc)[0]

    subcat_legend = {1: "Handling",
                     2: "Systems",
                     3: "Structural",
                     4: "Propeller",
                     5: "Power Plant",
                     6: "Oper/Perf/Capability",
                     7: "Fluids / Misc Hardware"}




    return dict(prediction = subcat_legend[y_pred])

#    return {'prediction': subcat_legend[y_pred]}
