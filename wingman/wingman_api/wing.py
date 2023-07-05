from fastapi import FastAPI
import pickle
import numpy as np

api = FastAPI()

# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "API connected"}


@api.get("/predict")
def predict(X_input):

    model = pickle.load_model()
    y_pred = model.predict(X_input)[0]

    subcat_legend = {1: "Handling",
                     2: "Systems",
                     3: "Structural",
                     4: "Propeller",
                     5: "Power Plant",
                     6: "Oper/Perf/Capability",
                     7: "Fluids / Misc Hardware"}

    return {'prediction': subcat_legend[y_pred]}
