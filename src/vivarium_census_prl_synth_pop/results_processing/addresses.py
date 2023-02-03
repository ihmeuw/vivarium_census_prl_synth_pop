from typing import Dict

import numpy as np
import pandas as pd
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import data_keys, data_values
from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)
from vivarium_census_prl_synth_pop.utilities import vectorized_choice


def get_household_address_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict[str, Dict[str, pd.Series]]:
    # This will return address_id mapped to address number, street name, and unit number.

    output_cols = [column_name]
    address_data = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols,
    )
    address_map = pd.Series(index=address_data.index, dtype=object)

    # Load address dazta from artifact
    synthetic_address_data = artifact.load(data_keys.SYNTHETIC_DATA.ADDRESSES)
    # Generate addresses
    for col in ["StreetNumber", "StreetName", "Unit"]:
        chosen_indices = vectorized_choice(
            options=synthetic_address_data.index,
            n_to_choose=len(address_map),
            randomness_stream=randomness,
        )
        address_map += synthetic_address_data.loc[chosen_indices, col].fillna("").values
        address_map += " "

    # todo: Separate address strings into necessary columns and return address map


