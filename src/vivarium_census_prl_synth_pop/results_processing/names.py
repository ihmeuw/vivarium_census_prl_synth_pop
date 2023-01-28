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


def get_first_name_map(
    raw_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
) -> Dict[str, pd.Series]:
    first_name_data = get_data_for_first_middle_name_mapping(
        "first_name_id",
        raw_data,
    )

    first_name_map = generate_first_and_middle_names(
        first_name_data, "first_name", artifact, randomness
    )

    return {"first_name": first_name_map}


def get_middle_name_map(
    raw_data: Dict[str, pd.DataFrame], artifact: Artifact, randomness: RandomnessStream
) -> Dict[str, pd.Series]:
    middle_name_data = get_data_for_first_middle_name_mapping(
        "middle_name_id",
        raw_data,
    )

    middle_name_map = generate_first_and_middle_names(
        middle_name_data, "middle_name", artifact, randomness
    )

    return {"middle_name": middle_name_map}


def get_data_for_first_middle_name_mapping(
    index_name: str, raw_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:

    input_cols = [index_name, "date_of_birth", "sex", "random_seed"]
    output_cols = [index_name, "year_of_birth", "sex"]

    name_data = format_data_for_mapping(
        index_name=index_name,
        raw_results=raw_data,
        columns_required=input_cols,
        output_columns=output_cols,
    )

    return name_data


def generate_first_and_middle_names(
    df_in: pd.DataFrame, additional_key: Any, artifact: Artifact, randomness: RandomnessStream
) -> pd.Series:
    """Generate synthetic names for individuals
    Parameters
    ----------
    df_in : pd.DataFrame, with columns sex, year_of_birth
    additional_key: key to pass to randomness stream
    artifact: Artifact with synthetic data
    randomness: Randomness stream for results processing
    Returns
    -------
    returns pd.Series with name_ids as index and string names as values.
    """
    # first and middle names
    # strategy: calculate year of birth based on age, use it with sex and state to find a representative name
    names = pd.Series(index=df_in.index, dtype=object)
    for (yob, sex), df_age in df_in.groupby(["year_of_birth", "sex"]):
        n = len(df_age)
        names.loc[df_age.index] = random_first_names(
            yob, sex, n, additional_key, artifact, randomness
        )

    return names


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
    ).to_numpy()  # TODO: include spaces and hyphens
