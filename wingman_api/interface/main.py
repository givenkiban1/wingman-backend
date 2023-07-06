import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from colorama import Fore, Style
import pickle
import time
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.getcwd())
print(os.getcwd() + '/../../wingman_api/')

from wingman_api.params import *
from wingman_api.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from wingman_api.ml_logic.model import initialize_model, fit_model, evaluate_model

sys.path.insert(0, os.getcwd() + '/wingman_api/ml_logic/')
from wingman_api.ml_logic.preprocessor import preprocess_features

# sys.path.insert(0, os.getcwd() + '/wingman_api/ml_logic/')
from wingman_api.ml_logic.registry import load_model, save_model, save_results
from wingman_api.ml_logic.registry import mlflow_run, mlflow_transition_model

PREPROCESSED_FIELDS = ['num_eng',
 'total_seats',
 'afm_hrs',
 'cert_max_gr_wt',
 'dprt_time',
 'power_units',
 'flight_hours_mean',
 'certs_held',
 'second_pilot',
 'site_seeing',
 'air_medical',
 'crew_sex',
 'type_last_insp_100H',
 'type_last_insp_AAIP',
 'type_last_insp_ANNL',
 'type_last_insp_COAW',
 'type_last_insp_COND',
 'type_last_insp_UNK',
 '_AAPL',
 '_AOBV',
 '_BUS ',
 '_FLTS',
 '_INST',
 '_OWRK',
 '_Other',
 '_PERS',
 '_POSI',
 '_UNK',
 'eng_mfgr_CONTINENTAL',
 'eng_mfgr_LYCOMING',
 'eng_mfgr_P&W',
 'eng_mfgr_ROTAX',
 'eng_mfgr_infrequent_sklearn',
 'far_part_091',
 'far_part_infrequent_sklearn',
 'acft_make_BEECH',
 'acft_make_CESSNA',
 'acft_make_PIPER',
 'acft_make_infrequent_sklearn',
 'fixed_retractable_RETR',
 'acft_category_AIR',
 'acft_category_infrequent_sklearn',
 'homebuilt',
 'crew_category_DSTU',
 'crew_category_FLTI',
 'crew_category_PLT',
 'eng_type_REC',
 'eng_type_infrequent_sklearn',
 'carb_fuel_injection_CARB',
 'carb_fuel_injection_FINJ',
 'carb_fuel_injection_UNK',
 'dprt_apt_id',
 'dest_apt_id',
 'flt_plan_filed_IFR',
 'flt_plan_filed_NONE',
 'flt_plan_filed_VFR',
 'pc_profession']


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    # if X_pred is None:
    #     X_pred = pd.DataFrame(dict(
    #     pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
    #     pickup_longitude=[-73.950655],
    #     pickup_latitude=[40.783282],
    #     dropoff_longitude=[-73.984365],
    #     dropoff_latitude=[40.769802],
    #     passenger_count=[1],
    # ))

    model = load_model()
    assert model is not None
    X_clean = clean_data(X_pred)
    X_processed = preprocess_features(X_clean, target_columns=[])

    values = [0 for field in PREPROCESSED_FIELDS]

    df = pd.DataFrame(values, columns=PREPROCESSED_FIELDS)

    df[X_processed.columns.to_list()] = X_processed

    print(X_processed.columns.to_list())
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


def train_n_save(wingman_data: pd.DataFrame = None) -> np.ndarray:
    print("\n🧪 Use case: Train")

    target_columns_v1 = ['phase_no', 'eventsoe_no']
    target_columns_v2 = ['category_no', 'subcategory_no', 'section_no', 'subsection_no', 'modifier_no']
    target_columns_v3 = ['category_no']
    target_columns_v4 = ['eventsoe_no']
    target_columns_v5 = ['subcategory_no']

    # Clean
    wingman_data_clean = clean_data(wingman_data)

    # Preprocess
    wingman_data_proc = preprocess_features(wingman_data_clean, target_columns=target_columns_v5)

    X = wingman_data_proc.drop(columns=["subcategory_no"])
    y = wingman_data_proc["subcategory_no"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    baseline_mod = initialize_model()

    baseline_mod = fit_model(baseline_mod, X_train, y_train)

    evaluate_model(baseline_mod, X_test, y_test)

    # Save model
    save_model(baseline_mod)

    print('✅ Done training model ⭐⭐⭐⭐⭐')


if __name__ == "__main__":

    # # ## Import data from GBQ
    # query = f"""SELECT * FROM {GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"""
    # data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("preclean", f"query_{DATA_SIZE}.csv")

    # print(data_query_cache_path)
    # if not os.path.exists(Path(LOCAL_DATA_PATH).joinpath("preclean")):
    #     # Create the directory
    #     os.makedirs(Path(LOCAL_DATA_PATH).joinpath("preclean"))

    # wingman_data = get_data_with_cache(GCP_PROJECT, query, data_query_cache_path)

    # # Train and save model
    # train_n_save(wingman_data)

    # Predict
    df = pd.read_csv('notebooks/inputs.csv')
    pred(df)
