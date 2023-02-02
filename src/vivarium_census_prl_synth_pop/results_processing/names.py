from typing import Any, Dict

import numpy as np
import pandas as pd
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import data_keys
from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)
from vivarium_census_prl_synth_pop.utilities import vectorized_choice


def get_given_name_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict[str, pd.Series]:
    """
    Parameters:
    -------
    column_name: Column name to map (example: first_name_id)
    obs_data: Dict of key observer names and value being dataframe of all that observers concated results.
    artifact: Artifact containing synthetic name data
    randomness: Randomness stream for post-processing

    Returns
    -------
    Dict with column name as key and pd.Series with name_ids as index and string names as values
    """

    output_cols = [column_name, "year_of_birth", "sex"]
    name_data = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols,
    )

    names_map = pd.Series(index=name_data.index, dtype=object)
    for (yob, sex), df_age in name_data.groupby(["year_of_birth", "sex"]):
        n = len(df_age)
        names_map.loc[df_age.index] = random_first_names(
            yob, sex, n, column_name, artifact, randomness
        )

    return {column_name.removesuffix("_id"): names_map}


def get_middle_initial_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict[str, pd.Series]:
    middle_name_map = get_given_name_map(column_name, obs_data, artifact, randomness)
    middle_initial_map = middle_name_map[column_name.removesuffix("_id")].str[0]

    return {"middle_initial": middle_initial_map}


def random_first_names(
    yob: int,
    sex: str,
    size: int,
    additional_key: Any,
    artifact: Artifact,
    randomness: RandomnessStream,
) -> np.ndarray:
    """

    Parameters
    ----------
    yob: the year of birth of the sims for whom to sample a name
    sex: the sex of the sims for whom to sample a name
    size: the number of sample to return
    additional_key: additional randomness key to pass vivarium.randomness
    artifact: Artifact with synthetic data
    randomness: Randomness stream for results processing

    Returns
    -------
    if [yob] <= 2020:
        nd.ndarray of [size] names sampled from the first names of people of sex [sex], born in the year [yob]
    if [yob] > 2020:
        nd.ndarray of [size] names sampled from the first names of people of sex [sex], born in the year 2020
    """
    # we only have data up to 2020; for younger children, sample from 2020 names.
    if yob > 2020:
        yob = 2020
    grouped_name_data = (
        artifact.load(data_keys.SYNTHETIC_DATA.FIRST_NAMES)
        .reset_index()
        .groupby(["yob", "sex"])
    )
    age_sex_specific_names = grouped_name_data.get_group((yob, sex))
    name_probabilities = age_sex_specific_names["freq"] / age_sex_specific_names["freq"].sum()
    return vectorized_choice(
        options=age_sex_specific_names.name,
        n_to_choose=size,
        randomness_stream=randomness,
        weights=name_probabilities,
        additional_key=additional_key,
    ).to_numpy()
