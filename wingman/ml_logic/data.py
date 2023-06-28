# imports
import pandas as pd

from google.cloud import bigquery
from pathlib import Path



# functions
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    pass # returns a df

def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    pass # returns a df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    pass # no return
