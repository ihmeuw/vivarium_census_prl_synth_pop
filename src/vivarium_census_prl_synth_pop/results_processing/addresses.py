from typing import Dict

import numpy as np
import pandas as pd
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import data_keys
from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)
from vivarium_census_prl_synth_pop.utilities import vectorized_choice

HOUSEHOLD_ADDRESS_COL_MAP = {
    "StreetNumber": "street_number",
    "StreetName": "street_name",
    "Unit": "unit_number",
}


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

    # Load address dazta from artifact
    synthetic_address_data = artifact.load(data_keys.SYNTHETIC_DATA.ADDRESSES).reset_index()
    # Generate addresses
    for artifact_column, obs_column in HOUSEHOLD_ADDRESS_COL_MAP.items():
        address_details = vectorized_choice(
            options=synthetic_address_data[artifact_column],
            n_to_choose=len(address_data),
            randomness_stream=randomness,
            additional_key=obs_column,
        )
        address_data[artifact_column] = address_details.fillna("").values
        # Update map
        address_map[obs_column] = address_data[obs_column]

    return {column_name: address_map}
