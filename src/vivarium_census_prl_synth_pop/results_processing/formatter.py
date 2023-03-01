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


def get_state_abbreviation(data: pd.DataFrame) -> pd.Series:
    state_id_map = {state: state_id for state_id, state in metadata.CENSUS_STATE_IDS.items()}
    state_name_map = data["state_id"].map(state_id_map)
    return state_name_map.map(metadata.US_STATE_ABBRV_MAP)


def get_employer_state_abbreviation(data: pd.DataFrame) -> pd.Series:
    state_id_map = {state: state_id for state_id, state in metadata.CENSUS_STATE_IDS.items()}
    state_name_map = data["employer_state_id"].map(state_id_map)
    return state_name_map.map(metadata.US_STATE_ABBRV_MAP)


def get_household_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["household_id"].astype(str)


def get_guardian_1_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["guardian_1"].astype(str)


def get_guardian_2_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["guardian_2"].astype(str)


def get_guardian_1_address_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["guardian_1_address_id"].astype(str)


def get_guardian_2_address_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["guardian_2_address_id"].astype(str)


# Fixme: Add formatting functions as necessary
COLUMN_FORMATTERS = {
    "simulant_id": (format_simulant_id, ["simulant_id", "random_seed"]),
    "year_of_birth": (get_year_of_birth, ["date_of_birth"]),
    "first_name_id": (get_first_name_id, ["first_name_id", "random_seed"]),
    "middle_name_id": (get_middle_name_id, ["middle_name_id", "random_seed"]),
    "last_name_id": (get_last_name_id, ["last_name_id", "random_seed"]),
    "state": (get_state_abbreviation, ["state_id"]),
    "address_id": (format_address_id, ["address_id", "random_seed"]),
    "employer_state": (get_employer_state_abbreviation, ["employer_state_id"]),
    "household_id": (get_household_id, ["household_id", "random_seed"]),
    "guardian_1": (get_guardian_1_id, ["guardian_1", "random_seed"]),
    "guardian_2": (get_guardian_2_id, ["guardian_2", "random_seed"]),
    "guardian_1_address_id": (get_guardian_1_id, ["guardian_1_address_id", "random_seed"]),
    "guardian_2_address_id": (get_guardian_2_id, ["guardian_2_address_id", "random_seed"]),
}


def format_data_for_mapping(
    index_name: str,
    obs_results: Dict[str, pd.DataFrame],
    output_columns: List[str],
) -> pd.DataFrame:
    data_to_map = [
        obs_data[output_columns]
        for obs_data in obs_results.values()
        if set(output_columns).issubset(set(obs_data.columns))
    ]
    data = pd.concat(data_to_map).drop_duplicates()
    data = data[output_columns].set_index(index_name)

    return data
