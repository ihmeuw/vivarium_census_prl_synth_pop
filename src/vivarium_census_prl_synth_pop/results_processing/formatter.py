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


def format_copy_age(data: pd.DataFrame) -> pd.Series:
    no_copy_idx = data.index[data["copy_age"].isna()]
    column = data["copy_age"].astype(str)
    column = column.apply(lambda row: row.split(".")[0])
    column.loc[no_copy_idx] = np.nan
    return column


def format_copy_ssn(data: pd.DataFrame) -> pd.Series:
    no_copy_idx = data.index[data["copy_ssn"].isna()]
    column = data["random_seed"].astype(str) + "_" + data["copy_ssn"].astype(str)
    column = column.apply(lambda row: row.split(".")[0])
    column.loc[no_copy_idx] = np.nan
    return column


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
    "copy_age": (format_copy_age, ["copy_age"]),
    "copy_ssn": (format_copy_ssn, ["copy_ssn", "random_seed"]),
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


def format_1040_dataset(processed_results: Dict[str, pd.Series]) -> pd.DataFrame:
    # Function that loads all tax datasets and formats them to the DATASET.1040 schema

    # Load data
    df_1040 = processed_results[metadata.DatasetNames.TAXES_1040]
    df_dependents = processed_results[metadata.DatasetNames.TAXES_DEPENDENTS]
    # Combine ssn and itin columns
    df_1040 = combine_ssn_and_itin_columns(df_1040)
    df_dependents = combine_ssn_and_itin_columns(df_dependents)

    # Get wide format of dependents - metadata for each guardian's dependents
    dependents_wide = flatten_data(
        data=df_dependents,
        index_cols=["guardian_id", "tax_year"],
        rank_col="simulant_id",
        value_cols=[
            "simulant_id",
            "first_name",
            "last_name",
            "ssn",
            "copy_ssn",
        ],
    )
    # Rename tax_dependents columns
    dependents_wide = dependents_wide.add_prefix("dependent_").reset_index()
    # Make sure we have all dependent columns if data does not have a guardian with 4 dependents
    for i in range(2, 5):
        if f"dependent_{i}_first_name" not in dependents_wide.columns:
            dependents_cols = ["first_name", "last_name", "ssn", "copy_ssn"]
            for column in dependents_cols:
                dependents_wide[f"dependent_{i}_{column}"] = np.nan

    # Widen 1040 data (make one row for spouses that are joint filing)
    df_joint_1040 = combine_joint_filers(df_1040)
    # Merge tax dependents onto their guardians - we must do it twice, merge onto each spouse if joint filing
    tax_1040_w_dependents = df_joint_1040.merge(
        dependents_wide,
        how="left",
        left_on=["simulant_id", "tax_year"],
        right_on=["guardian_id", "tax_year"],
    )
    # TODO: uncomment with mic-4244. Handle columns with dependents for both guardians
    # tax_1040_w_dependents = tax_1040_w_dependents.merge(
    #   dependents_wide, how="left",
    #   left_on=["COLUMNS.spouse_simulant_id.name", "COLUMNS.tax_year.name"],
    #   right_on=["COLUMNS.guardian_id.name", "COLUMNS.tax_year.name"])
    return tax_1040_w_dependents


def flatten_data(
    data: pd.DataFrame,
    index_cols: str,
    rank_col: str,
    value_cols: List[str],
    ascending: bool = False,
) -> pd.DataFrame:
    # Function that takes a dataset and widens (pivots) it to capture multiple metadata columns
    # Example: simulant_id, dependdent_1, dependent_2, dependent_1_name, dependent_2_name, etc...
    data = data.copy()
    # fixme: find a better solution than the following call since applying lambda functions is slow
    data["rank"] = (
        data.groupby(index_cols, group_keys=False)[rank_col]
        .apply(lambda x: x.rank(method="first", ascending=ascending))
        .astype(int)
    )
    # TODO: Improve via mic-4244 for random sampling of dependents
    # Choose 4 dependents
    data = data.loc[data["rank"] < 5]
    data["rank"] = data["rank"].astype(str)
    flat = data.pivot(columns="rank", index=index_cols, values=value_cols)
    flat.columns = ["_".join([pair[1], pair[0]]) for pair in flat.columns]

    return flat


def combine_joint_filers(data: pd.DataFrame) -> pd.DataFrame:
    # Get groups
    joint_filers = data.loc[data["joint_filer"] == True]
    reference_persons = data.loc[
        data["relationship_to_reference_person"] == "Reference person"
    ]
    independent_filers_index = data.index.difference(
        joint_filers.index.union(reference_persons.index)
    )
    # This is a dataframe with all independent filing individuals that are not a reference person
    independent_filers = data.loc[independent_filers_index]

    joint_filers = joint_filers.add_prefix("spouse_")
    # Merge spouses
    reference_persons_wide = reference_persons.merge(
        joint_filers,
        how="left",
        left_on=["household_id", "tax_year"],
        right_on=["spouse_household_id", "spouse_tax_year"],
    )
    joint_1040 = pd.concat([reference_persons_wide, independent_filers])

    return joint_1040


def combine_ssn_and_itin_columns(data: pd.DataFrame) -> pd.DataFrame:
    # This combines the ssn and itin columns into the ssn column.
    # Simulants can either have an ssn or an itin so we will replace
    # the nans in the ssn column with that rows corresponding itin value
    data["ssn"] = np.where(data["ssn"].notna(), data["ssn"], data["itin"])
    return data
