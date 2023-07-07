from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
import os
from pydantic import BaseModel, Field
from typing import List

# from wingman.ml_logic.preprocessor import preprocess_features
# from wingman.ml_logic.registry import load_model

api = FastAPI()
print(os.getcwd())
try:
    api.state.model = pickle.load(open("model.pkl","rb")) # /../wingman_api/
except Exception as e:
    print(str(e))
    print(os.getcwd())

# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "API connected"}


PREPROCESSED_FIELDS = [
'total_seats',
 'afm_hrs',
 'cert_max_gr_wt',
 'dprt_time',
 'power_units',
 'flight_hours_mean',
 'type_last_insp_ANNL',
 'eng_mfgr_CONTINENTAL',
 'eng_mfgr_LYCOMING',
 'far_part_091',
 'far_part_infrequent_sklearn',
 'acft_make_CESSNA',
 'acft_make_infrequent_sklearn',
 'fixed_retractable_RETR',
 'acft_category_AIR',
 'acft_category_infrequent_sklearn',
 'homebuilt',
 'crew_category_DSTU',
 'crew_category_PLT'
]

class Prediction(BaseModel):
    prediction: str = ""
    proba: list = []
    subcat_legend: dict = {}
    y_pred: List[int] = []

class X_input(BaseModel):
   field_names: list
   values: list

@api.post("/predict", response_model=Prediction)
async def predict(x_input: X_input):

    obj2 = {}
    for field in PREPROCESSED_FIELDS:
        obj2[field] = [0]

    df2 = pd.DataFrame(data=obj2)

    for index,val in enumerate(x_input.field_names):
        df2[val] = x_input.values[index]


    # X_input = pd.DataFrame(data=obj)
    # X_preproc = preprocess_features(X_input)

    y_pred = api.state.model.predict(df2).tolist()
    proba = api.state.model.predict_proba(df2).tolist()

    print(type(y_pred))

    subcat_legend = {1: "Handling",
                     2: "Systems",
                     3: "Structural",
                     4: "Propeller",
                     5: "Power Plant",
                     6: "Oper/Perf/Capability",
                     7: "Fluids / Misc Hardware"}
    data = Prediction()
    data.prediction = subcat_legend[y_pred[0]]
    data.proba = proba
    data.subcat_legend = subcat_legend
    data.y_pred = y_pred



    # return dict(prediction = subcat_legend[y_pred], proba = proba)
    return data

#    return {'prediction': subcat_legend[y_pred]}


# original preprocessed fields
# 'num_eng',
#  'total_seats',
#  'afm_hrs',
#  'cert_max_gr_wt',
#  'dprt_time',
#  'power_units',
#  'flight_hours_mean',
#  'certs_held',
#  'second_pilot',
#  'site_seeing',
#  'air_medical',
#  'crew_sex',
#  'type_last_insp_100H',
#  'type_last_insp_AAIP',
#  'type_last_insp_ANNL',
#  'type_last_insp_COAW',
#  'type_last_insp_COND',
#  'type_last_insp_UNK',
#  '_AAPL',
#  '_AOBV',
#  '_BUS ',
#  '_FLTS',
#  '_INST',
#  '_OWRK',
#  '_Other',
#  '_PERS',
#  '_POSI',
#  '_UNK',
#  'eng_mfgr_CONTINENTAL',
#  'eng_mfgr_LYCOMING',
#  'eng_mfgr_P&W',
#  'eng_mfgr_ROTAX',
#  'eng_mfgr_infrequent_sklearn',
#  'far_part_091',
#  'far_part_infrequent_sklearn',
#  'acft_make_BEECH',
#  'acft_make_CESSNA',
#  'acft_make_PIPER',
#  'acft_make_infrequent_sklearn',
#  'fixed_retractable_RETR',
#  'acft_category_AIR',
#  'acft_category_infrequent_sklearn',
#  'homebuilt',
#  'crew_category_DSTU',
#  'crew_category_FLTI',
#  'crew_category_PLT',
#  'eng_type_REC',
#  'eng_type_infrequent_sklearn',
#  'carb_fuel_injection_CARB',
#  'carb_fuel_injection_FINJ',
#  'carb_fuel_injection_UNK',
#  'dprt_apt_id',
#  'dest_apt_id',
#  'flt_plan_filed_IFR',
#  'flt_plan_filed_NONE',
#  'flt_plan_filed_VFR',
#  'pc_profession'
