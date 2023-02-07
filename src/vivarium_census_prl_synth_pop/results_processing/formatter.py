from typing import Dict, List

import pandas as pd

from vivarium_census_prl_synth_pop.constants import metadata


def get_year_of_birth(data: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(data["date_of_birth"]).dt.year


def get_first_name_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["first_name_id"].astype(str)


def get_middle_name_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["middle_name_id"].astype(str)


def get_last_name_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["last_name_id"].astype(str)


def format_simulant_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["simulant_id"].astype(str)


def format_address_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["address_id"].astype(str)


def get_state_name(data: pd.DataFrame) -> pd.Series:
    state_map = {state: state_id for state_id, state in metadata.CENSUS_STATE_IDS.items()}
    return data["state"].map(state_map)


# Fixme: Add formatting functions as necessary
COLUMN_FORMATTERS = {
    "simulant_id": (format_simulant_id, ["simulant_id", "random_seed"]),
    "year_of_birth": (get_year_of_birth, ["date_of_birth"]),
    "first_name_id": (get_first_name_id, ["first_name_id", "random_seed"]),
    "middle_name_id": (get_middle_name_id, ["middle_name_id", "random_seed"]),
    "last_name_id": (get_last_name_id, ["last_name_id", "random_seed"]),
    "state": (get_state_name, ["state"]),
    "address_id": (format_address_id, ["address_id", "random_seed"]),
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
