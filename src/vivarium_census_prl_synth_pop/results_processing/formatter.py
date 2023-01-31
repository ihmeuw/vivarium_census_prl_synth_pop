from typing import Dict, List

import pandas as pd


def get_year_of_birth(data: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(data["date_of_birth"]).dt.year


def get_first_name_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["first_name_id"].astype(str)


def get_middle_name_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["middle_name_id"].astype(str)


def get_prl_tracking_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["Unnamed: 0"].astype(str)


# Fixme: Add formatting functions as necessary
COLUMN_FORMATTERS = {
    "prl_tracking_id": get_prl_tracking_id,
    "year_of_birth": get_year_of_birth,
    "first_name_id": get_first_name_id,
    "middle_name_id": get_middle_name_id,
}


def format_data_for_mapping(
    index_name: str,
    obs_results: Dict[str, pd.DataFrame],
    output_columns: List[str],
) -> pd.DataFrame:

    data_to_map = [obs_data[output_columns] for obs_data in obs_results.values()]
    data = pd.concat(data_to_map).drop_duplicates()
    data = data[output_columns].set_index(index_name)

    return data
