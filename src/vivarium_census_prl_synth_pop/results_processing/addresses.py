from typing import Dict

import pandas as pd
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import data_keys
from vivarium_census_prl_synth_pop.constants.paths import PUMA_TO_ZIP_DATA_PATH
from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)
from vivarium_census_prl_synth_pop.utilities import vectorized_choice

HOUSEHOLD_ADDRESS_COL_MAP = {
    "StreetNumber": "street_number",
    "StreetName": "street_name",
    "Unit": "unit_number",
}


def get_address_id_maps(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict:
    """
    Get all maps that are indexed by `address_id`.

    Parameters
    ----------
    column_name
        Name of the column to use as an index
    obs_data
        Observer DataFrame with key for the observer name
    artifact
        A vivarium Artifact object needed by mapper
    randomness
        RandomnessStream to use in choosing zipcodes proportionally

    Returns
    -------
    A dictionary of pd.Series suitable for pd.Series.map, indexed by `address_id`

    """
    if column_name != "address_id":
        raise ValueError(f"Expected `address_id`, got `{column_name}`")
    maps = dict()
    output_cols_superset = [column_name, "state_id", "state", "puma"]
    formatted_obs_data = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols_superset,
    )
    maps.update(get_zipcode_map(column_name, formatted_obs_data, randomness))
    maps.update(
        get_household_address_map(column_name, formatted_obs_data, artifact, randomness)
    )
    maps.update(get_city_map(column_name, formatted_obs_data, artifact, randomness))
    return maps


def get_zipcode_map(
    column_name: str,
    obs_data: pd.DataFrame,
    randomness: RandomnessStream,
) -> Dict[str, pd.Series]:
    """Gets a mapper for `address_id` to zipcode, based on state, puma, and proportion.

    Parameters
    ----------
    column_name
        Name of the column to use as an index
    obs_data
        Observer DataFrame with key for the observer name
    randomness
        RandomnessStream to use in choosing zipcodes proportionally

    Returns
    -------
    A pd.Series suitable for pd.Series.map, indexed by column_name, with key "zipcode"

    """
    zip_map_dict = {}
    output_cols = [column_name, "state_id", "puma"]  # columns in the output we use to map
    simulation_addresses = (
        obs_data.reset_index()[output_cols].drop_duplicates().set_index("address_id")
    )
    zip_map = pd.Series(index=simulation_addresses.index)

    # Read in CSV and normalize
    map_data = pd.read_csv(PUMA_TO_ZIP_DATA_PATH)
    proportions = (
        map_data.groupby(["state", "puma"])
        .sum()["proportion"]
        .reset_index()
        .set_index(["state", "puma"])
    )
    normalized_groupby = (
        (map_data.set_index(["state", "puma", "zipcode"]) / proportions)
        .reset_index()
        .groupby(["state", "puma"])
    )

    for (state_id, puma), df_locale in simulation_addresses.groupby(["state_id", "puma"]):
        locale_group = normalized_groupby.get_group((state_id, puma))
        zip_map.loc[df_locale.index] = vectorized_choice(
            options=locale_group["zipcode"],
            n_to_choose=len(df_locale),
            randomness_stream=randomness,
            weights=locale_group["proportion"],
            additional_key=f"zip_map_{state_id}_{puma}",
        ).to_numpy()

    # Map against obs_data
    zip_map_dict["zipcode"] = zip_map.astype(int)
    return zip_map_dict


def get_household_address_map(
    column_name: str,
    obs_data: pd.DataFrame,
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict[str, pd.Series]:
    # This will return address_id mapped to address number, street name, and unit number.

    address_map = {}
    output_cols = [column_name]
    address_ids = (
        obs_data.reset_index()[output_cols].drop_duplicates().set_index("address_id")
    )
    address_data = pd.DataFrame(index=address_ids.index)

    # Load address data from artifact
    synthetic_address_data = artifact.load(data_keys.SYNTHETIC_DATA.ADDRESSES).reset_index()
    # Generate addresses
    for artifact_column, obs_column in HOUSEHOLD_ADDRESS_COL_MAP.items():
        address_details = vectorized_choice(
            options=synthetic_address_data[artifact_column],
            n_to_choose=len(address_data),
            randomness_stream=randomness,
            additional_key=obs_column,
        ).to_numpy()
        address_data[obs_column] = address_details
        address_data.fillna("", inplace=True)
        # Update map
        address_map[obs_column] = address_data[obs_column]

    return address_map


def get_city_map(
    column_name: str,
    obs_data: pd.DataFrame,
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict[str, pd.Series]:
    # Load addresses data from artifact
    addresses = artifact.load(data_keys.SYNTHETIC_DATA.ADDRESSES).reset_index()
    # Get observer data to map
    output_cols = [column_name, "state"]
    city_data = obs_data.reset_index()[output_cols].drop_duplicates().set_index("address_id")

    for state in city_data["state"].str.lower().unique():
        cities = vectorized_choice(
            options=addresses.loc[addresses["Province"] == state, "Municipality"],
            n_to_choose=len(city_data.loc[city_data["state"] == state.upper()]),
            randomness_stream=randomness,
            additional_key="city",
        ).to_numpy()
        city_data.loc[city_data["state"] == state.upper(), "city"] = cities

    city_map = {"city": city_data["city"]}
    return city_map
