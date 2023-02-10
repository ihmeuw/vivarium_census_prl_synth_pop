from typing import Dict

import numpy as np
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
):
    """
    Get all maps that are indexed by `address_id`.
    """
    if column_name != "address_id":
        raise ValueError(f"Expected `address_id`, got `{column_name}`")
    maps = dict()
    maps.update(get_zip_map(column_name, obs_data, randomness))
    maps.update(get_household_address_map(column_name, obs_data, artifact, randomness))
    return maps


def get_zip_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    randomness: RandomnessStream,
) -> Dict[str, pd.Series]:
    """Adds a logging sink to the global process logger.

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
    output_cols = [column_name, "state", "puma"]  # columns in the output we use to map
    address_ids = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols,
    )
    zip_map = pd.Series(index=address_ids.index)

    # Read in CSV and normalize
    map_data = pd.read_csv(PUMA_TO_ZIP_DATA_PATH)
    proportions = (
        map_data.groupby(["state", "puma"])
        .apply(sum)["proportion"]
        .reset_index()
        .set_index(["state", "puma"])
    )
    normalized_groupby = (
        (map_data.set_index(["state", "puma", "zipcode"]) / proportions)
        .reset_index()
        .groupby(["state", "puma"])
    )

    for (state, puma), df_locale in address_ids.groupby(["state", "puma"]):
        locale_group = normalized_groupby.get_group((state, puma))
        zip_map.loc[df_locale.index] = vectorized_choice(
            options=locale_group["zipcode"],
            n_to_choose=len(df_locale),
            randomness_stream=randomness,
            weights=locale_group["proportion"],
            additional_key=f"zip_map_{state}_{puma}",
        ).to_numpy()

    # Map against obs_data
    zip_map_dict["zipcode"] = zip_map.astype(int)
    return zip_map_dict


def get_household_address_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict[str, Dict[str, pd.Series]]:
    # This will return address_id mapped to address number, street name, and unit number.

    address_map = {}
    output_cols = [column_name]
    address_ids = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols,
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