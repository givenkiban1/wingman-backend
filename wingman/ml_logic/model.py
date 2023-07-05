# imports
import numpy as np

from tensorflow.keras import Model
from typing import Tuple

# functions
# imports
import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_test(data):
    data = data.drop(columns=['_AOBV', 'site_seeing', 'air_medical', '_BUS ', '_POSI', 'type_last_insp_COAW',
                                     'certs_held', 'type_last_insp_AAIP', '_Other', '_OWRK', 'type_last_insp_UNK',
                                     '_AAPL', 'eng_mfgr_P&W', '_FLTS', 'crew_sex', '_UNK', 'eng_type_REC',
                                     'eng_type_infrequent_sklearn', 'flt_plan_filed_IFR', 'num_eng',
                                     'flt_plan_filed_VFR', 'type_last_insp_COND', 'dprt_apt_id', 'second_pilot',
                                     '_INST', 'eng_mfgr_infrequent_sklearn', 'eng_mfgr_ROTAX', 'type_last_insp_100H',
                                     'crew_category_FLTI', 'dest_apt_id', 'carb_fuel_injection_UNK', 'flt_plan_filed_NONE',
                                     'acft_make_PIPER', 'carb_fuel_injection_FINJ', 'pc_profession', '_PERS',
                                     'acft_make_BEECH', 'carb_fuel_injection_CARB'], axis=1)
    X = data.drop('subcategory_no', axis=1)
    y = data['subcategory_no']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test

# functions

def initialize_model(input_shape: tuple):
    model = RandomForestClassifier(n_estimators= 2500, min_samples_split=5, min_samples_leaf=15,
                                        max_features= 'sqrt', max_depth=168, bootstrap=False)
    return model


def train_model(
        model: model,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
    return model.fit(X_train, y_train)

def evaluate_model(
        model: model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
