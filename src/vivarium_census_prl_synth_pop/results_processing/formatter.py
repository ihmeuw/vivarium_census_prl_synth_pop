from typing import Dict, List

import numpy as np
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


def format_ssn_id(data: pd.DataFrame) -> pd.Series:
    """Format ssn_id column to prepare random_seed to match simulant_id"""
    ssn_ids = pd.Series(data["ssn_id"].astype(str))
    # Only do the prepending where ssn_id points to another simulant
    ssn_ids[ssn_ids != "-1"] = (
        data["random_seed"].astype(str) + "_" + data["ssn_id"].astype(str)
    )
    return ssn_ids


def get_state_abbreviation(data: pd.DataFrame) -> pd.Series:
    state_id_map = {state: state_id for state_id, state in metadata.CENSUS_STATE_IDS.items()}
    state_name_map = data["state_id"].map(state_id_map)
    state_name_map = state_name_map.map(metadata.US_STATE_ABBRV_MAP)
    categories = sorted(list(metadata.US_STATE_ABBRV_MAP.values()))
    return state_name_map.astype(pd.CategoricalDtype(categories=categories))


def get_employer_state_abbreviation(data: pd.DataFrame) -> pd.Series:
    state_id_map = {state: state_id for state_id, state in metadata.CENSUS_STATE_IDS.items()}
    state_name_map = data["employer_state_id"].map(state_id_map)
    state_name_map = state_name_map.map(metadata.US_STATE_ABBRV_MAP)
    categories = sorted(list(metadata.US_STATE_ABBRV_MAP.values()))
    return state_name_map.astype(pd.CategoricalDtype(categories=categories))


def get_household_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["household_id"].astype(str)


def get_guardian_1_id(data: pd.DataFrame) -> pd.Series:
    no_guardian_idx = data.index[data["guardian_1"] == -1]
    column = data["random_seed"].astype(str) + "_" + data["guardian_1"].astype(str)
    column.loc[no_guardian_idx] = np.nan

    return column


def get_guardian_2_id(data: pd.DataFrame) -> pd.Series:
    no_guardian_idx = data.index[data["guardian_2"] == -1]
    column = data["random_seed"].astype(str) + "_" + data["guardian_2"].astype(str)
    column.loc[no_guardian_idx] = np.nan

    return column


def get_guardian_1_address_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["guardian_1_address_id"].astype(str)


def get_guardian_2_address_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["guardian_2_address_id"].astype(str)


def get_guardian_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["guardian_id"].astype(str)


def format_age(data: pd.DataFrame) -> pd.Series:
    return data["age"].astype(int)


def format_copy_ssn(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["copy_ssn"].astype(str)


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
    "guardian_1_address_id": (
        get_guardian_1_address_id,
        ["guardian_1_address_id", "random_seed"],
    ),
    "guardian_2_address_id": (
        get_guardian_2_address_id,
        ["guardian_2_address_id", "random_seed"],
    ),
    "guardian_id": (get_guardian_id, ["guardian_id", "random_seed"]),
    "ssn_id": (format_ssn_id, ["ssn_id", "random_seed"]),
    "age": (format_age, ["age"]),
    "ssn_copy": (format_copy_ssn, ["copy_ssn", "random_seed"]),
}


def format_columns(data: pd.DataFrame) -> pd.DataFrame:
    # Process columns to map for observers
    for output_column, (
        column_formatter,
        required_columns,
    ) in COLUMN_FORMATTERS.items():
        if set(required_columns).issubset(set(data.columns)):
            data[output_column] = column_formatter(data[required_columns])
    return data


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
