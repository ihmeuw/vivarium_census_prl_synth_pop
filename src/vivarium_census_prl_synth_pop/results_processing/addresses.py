from typing import Any, Dict

import pandas as pd
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import data_keys, data_values
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

    if column_name == "address_id":
        output_cols_superset = [column_name, "state_id", "state", "puma", "po_box"]
    elif column_name == "employer_address_id":
        output_cols_superset = [
            column_name,
            "employer_state_id",
            "employer_state",
            "employer_puma",
        ]
    else:
        raise ValueError(
            f"Expected `address_id` or 'employer_address_id', got `{column_name}`"
        )

    maps = dict()
    formatted_obs_data = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols_superset,
    )
    maps.update(get_zipcode_map(column_name, formatted_obs_data, randomness))
    maps.update(get_street_details_map(column_name, formatted_obs_data, artifact, randomness))
    maps.update(get_city_map(column_name, formatted_obs_data, artifact, randomness))
    if column_name == "address_id":
        maps.update(
            get_mailing_address_map(
                column_name, formatted_obs_data, artifact, randomness, maps
            )
        )
    return maps


def get_zipcode_map(
    column_name: str,
    obs_data: pd.DataFrame,
    randomness: RandomnessStream,
    additional_key: str = "",
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
    additional_key
        Key for RandomnessStream.  The use case here is for resampling of mailing addresses.

    Returns
    -------
    A pd.Series suitable for pd.Series.map, indexed by column_name, with key "zipcode"

    """
    zip_map_dict = {}
    if column_name == "address_id":
        output_cols = [column_name, "state_id", "puma"]  # columns in the output we use to map
        groupby_cols = ["state_id", "puma"]
    else:
        output_cols = [column_name, "employer_state_id", "employer_puma"]
        groupby_cols = ["employer_state_id", "employer_puma"]
    simulation_addresses = (
        obs_data.reset_index()[output_cols].drop_duplicates().set_index(column_name)
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

    for (state_id, puma), df_locale in simulation_addresses.groupby(groupby_cols):
        locale_group = normalized_groupby.get_group((state_id, puma))
        zip_map.loc[df_locale.index] = vectorized_choice(
            options=locale_group["zipcode"],
            n_to_choose=len(df_locale),
            randomness_stream=randomness,
            weights=locale_group["proportion"],
            additional_key=f"{additional_key}{column_name}_zip_map_{state_id}_{puma}",
        ).to_numpy()

    # Map against obs_data
    if column_name == "address_id":
        zip_map_dict["zipcode"] = zip_map.astype(int)
    else:
        zip_map_dict["employer_zipcode"] = zip_map.astype(int)
    return zip_map_dict


def get_street_details_map(
    column_name: str,
    obs_data: pd.DataFrame,
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict[str, pd.Series]:
    # This will return address_id mapped to address number, street name, and unit number.
    # This function will be used and get the same columns for address_id and employer_address_id

    address_map = {}
    output_cols = [column_name]
    address_ids = obs_data.reset_index()[output_cols].drop_duplicates().set_index(column_name)
    address_data = pd.DataFrame(index=address_ids.index)

    # Load address data from artifact
    synthetic_address_data = artifact.load(data_keys.SYNTHETIC_DATA.ADDRESSES).reset_index()
    # Generate addresses
    for artifact_column, obs_column in HOUSEHOLD_ADDRESS_COL_MAP.items():
        address_details = vectorized_choice(
            options=synthetic_address_data[artifact_column],
            n_to_choose=len(address_data),
            randomness_stream=randomness,
            additional_key=f"{column_name}_{obs_column}",
        ).to_numpy()
        address_data[obs_column] = address_details
        address_data.fillna("", inplace=True)
        # Update map
        if column_name == "address_id":
            address_map[obs_column] = address_data[obs_column]
        else:
            address_map[f"employer_{obs_column}"] = address_data[obs_column]

    return address_map


def get_city_map(
    column_name: str,
    obs_data: pd.DataFrame,
    artifact: Artifact,
    randomness: RandomnessStream,
    additional_key: str = "",
) -> Dict[str, pd.Series]:
    # Load addresses data from artifact
    addresses = artifact.load(data_keys.SYNTHETIC_DATA.ADDRESSES).reset_index()
    # Get observer data to map
    if column_name == "address_id":
        state_col = "state"
    else:
        state_col = "employer_state"
    output_cols = [column_name, state_col]
    city_data = obs_data.reset_index()[output_cols].drop_duplicates().set_index(column_name)

    for state in city_data[state_col].str.lower().unique():
        cities = vectorized_choice(
            options=addresses.loc[addresses["Province"] == state, "Municipality"],
            n_to_choose=len(city_data.loc[city_data[state_col] == state.upper()]),
            randomness_stream=randomness,
            additional_key=f"{additional_key}{column_name}_city_map",
        ).to_numpy()
        city_data.loc[city_data[state_col] == state.upper(), "city"] = cities

    if column_name == "address_id":
        city_map = {"city": city_data["city"]}
    else:
        city_map = {"employer_city": city_data["city"]}
    return city_map


def get_mailing_address_map(
    column_name: str,
    formatted_obs_data: pd.DataFrame,
    artifact: Artifact,
    randomness: RandomnessStream,
    maps: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    """
    Returns a dict with mailing addresses mapped to address_id.  These mailing addresses will have the same columns that
    physical addresses have - street number, street name, and unit number will be blanked for addresses WITH A PO box.
    Note: We need to generate mailing addresses after physical addresses for address_ids that do not have a different
    mailing address.  In this case, we will just copy the values over for each column.
    Note: This must occur after a physical address has been generated for address_ids.  In other words, maps.keys()
    must contain columns for physical address ["street_number", "street_name", etc].
    """

    # Get address_ds that have a PO box
    po_box_address_ids = formatted_obs_data["po_box"] != data_values.NO_PO_BOX
    # Setup mailing address map
    mailing_address_map = {}
    # Copy address line one columns and blank out columns for PO box address_ids.
    for column in HOUSEHOLD_ADDRESS_COL_MAP.values():
        mailing_address_map[f"mailing_address_{column}"] = pd.Series(
            "", index=formatted_obs_data.index
        )
        # Move over address details for non-PO box addresses
        mailing_address_map[f"mailing_address_{column}"][~po_box_address_ids] = maps[column][
            ~po_box_address_ids
        ]

    # Copy over address detail columns that will remain the same for all address_ids or be updated for just address_ids
    #  with a PO box.
    for column in ["po_box", "state"]:
        mailing_address_map[f"mailing_address_{column}"] = formatted_obs_data[column].copy()
    for column in ["city", "zipcode"]:
        mailing_address_map[f"mailing_address_{column}"] = maps[column].copy()

    # Resample city and zipcode for mailing address
    # Note: Address line 1 is already blanked with creation of mailing_address_map
    # Get subset of observer data that is just PO boxes for resampling
    po_boxes = formatted_obs_data[po_box_address_ids]
    zipcode_map = get_zipcode_map(column_name, po_boxes, randomness, "mailing_")
    mailing_address_map["mailing_address_zipcode"][po_box_address_ids] = zipcode_map[
        "zipcode"
    ]
    city_map = get_city_map(column_name, po_boxes, artifact, randomness, "mailing_")
    mailing_address_map["mailing_address_city"][po_box_address_ids] = city_map["city"]

    return mailing_address_map
