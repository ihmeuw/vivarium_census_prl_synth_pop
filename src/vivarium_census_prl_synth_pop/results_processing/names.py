from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import data_keys, data_values
from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)
from vivarium_census_prl_synth_pop.utilities import vectorized_choice


def get_given_name_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
    seed: str,
    *_: Any,
) -> Dict[str, pd.Series]:
    """
    Parameters:
    -------
    column_name: Column name to map (example: first_name_id)
    obs_data: Dict of key observer names and value being dataframe of all that observers concatenated results.
    artifact: Artifact containing synthetic name data
    randomness: Randomness stream for post-processing
    seed: vivarium random seed this is being run for

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
            yob, sex, n, column_name, artifact, randomness, seed
        )

    return {column_name.removesuffix("_id"): names_map}


def get_middle_initial_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
    seed: str,
    *_: Any,
) -> Dict[str, pd.Series]:
    """
    Parameters:
    -------
    column_name: Column name to map (example: first_name_id)
    obs_data: Dict of key observer names and value being dataframe of all that observers concatenated results.
    artifact: Artifact containing synthetic name data
    randomness: Randomness stream for post-processing
    seed: vivarium random seed this is being run for

    Returns
    -------
    Dict with column name as key and pd.Series with middle initial as index and
    string names as values
    """
    middle_name_map = get_given_name_map(column_name, obs_data, artifact, randomness, seed)
    middle_initial_map = middle_name_map[column_name.removesuffix("_id")].str[0]

    return {"middle_initial": middle_initial_map}


def get_last_name_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
    seed: str,
    *_: Any,
) -> Dict[str, pd.Series]:
    """
    Parameters:
    -------
    column_name: Column name to map (example: first_name_id)
    obs_data: Dict of key observer names and value being dataframe of all that observers concated results.
    artifact: Artifact containing synthetic name data
    randomness: Randomness stream for post-processing
    seed: vivarium random seed this is being run for
    Returns
    -------
    Dict with column name as key and pd.Series with name_ids as index and string names as values
    """
    output_cols = [column_name, "race_ethnicity", "date_of_birth"]
    name_data = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols,
    )

    # Get oldest simulant for each existing last_name_id
    name_data = name_data.groupby("last_name_id", group_keys=False).apply(
        pd.DataFrame.sort_values, "date_of_birth"
    )
    oldest = name_data.reset_index().drop_duplicates("last_name_id").set_index("last_name_id")

    last_names_map = pd.Series("", index=oldest.index)
    for race_eth, df_race_eth in oldest.groupby("race_ethnicity"):
        last_names_map.loc[df_race_eth.index] = random_last_names(
            race_eth, len(df_race_eth), column_name, artifact, randomness, seed
        )

    return {column_name.removesuffix("_id"): last_names_map}


def random_first_names(
    yob: int,
    sex: str,
    size: int,
    additional_key: Any,
    artifact: Artifact,
    randomness: RandomnessStream,
    seed: str,
) -> np.ndarray:
    """

    Parameters
    ----------
    yob: the year of birth of the sims for whom to sample a name
    sex: the sex of the sims for whom to sample a name
    size: the number of sample to return
    additional_key: additional randomness key to pass to RandomnessStream
    artifact: Artifact with synthetic data
    randomness: Randomness stream for results processing
    seed: vivarium random seed this is being run for

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
        additional_key=f"{additional_key}_{seed}",
    ).to_numpy()


def random_last_names(
    race_eth: str,
    size: int,
    additional_key: Any,
    artifact: Artifact,
    randomness: RandomnessStream,
    seed: str,
) -> np.ndarray:
    """
    Parameters
    ----------
    race_eth: the race_ethnicity category (string) of the sims for whom to sample a name
    size: the number of samples to return
    additional_key: additional randomness key to pass to RandomnessStream
    artifact: Artifact containing synthetic names data
    randomness: RandomnessStream for post-processing
    seed: vivarium random seed this is being run for
    Returns
    -------
    nd.ndarray of [size] last names sampled from people of race and ethnicity [race_eth]
    """
    df_census_names = artifact.load(data_keys.SYNTHETIC_DATA.LAST_NAMES).reset_index()
    l = len(df_census_names)
    # fixme: Nan in artifact data for last names.
    df_census_names = df_census_names.loc[~df_census_names["name"].isnull()]
    if len(df_census_names) < l:
        logger.info(
            "Artifact contains missing values for last names data.  "
            "Removing null values from last name data..."
        )

    # randomly sample last names
    last_names = vectorized_choice(
        options=df_census_names.name,
        n_to_choose=size,
        randomness_stream=randomness,
        weights=df_census_names[race_eth],
        additional_key=f"{additional_key}_{seed}",
    )

    # Last names sometimes also include spaces or hyphens, and abie has
    # come up with race/ethnicity specific space and hyphen
    # probabilities from an analysis of voter registration data (from
    # publicly available data from North Carolina, filename
    # VR_Snapshot_20220101.txt; see
    # 2022_06_02b_prl_code_for_probs_of_spaces_and_hyphens_in_last_and_first_names.ipynb
    # for computation details.)

    # for some names, add a hyphen between two randomly samples last names
    probability_of_hyphen = data_values.PROBABILITY_OF_HYPHEN_IN_NAME[race_eth]
    hyphen_rows = (
        randomness.get_draw(last_names.index, f"choose_hyphen_sims_{seed}")
        < probability_of_hyphen
    )
    if hyphen_rows.sum() > 0:
        last_names[hyphen_rows] += (
            "-"
            + vectorized_choice(
                options=df_census_names.name,
                n_to_choose=hyphen_rows.sum(),
                randomness_stream=randomness,
                weights=df_census_names[race_eth],
                additional_key=f"hyphen_last_names{seed}",
            ).to_numpy()
        )

    # add spaces to some names
    probability_of_space = data_values.PROBABILITY_OF_SPACE_IN_NAME[race_eth]
    space_rows = randomness.get_draw(
        last_names.index, f"choose_space_sims_{seed}"
    ) < probability_of_space * (
        1 - hyphen_rows
    )  # HACK: don't put spaces in names that are already hyphenated
    if space_rows.sum() > 0:
        last_names[space_rows] += (
            " "
            + vectorized_choice(
                options=df_census_names.name,
                n_to_choose=space_rows.sum(),
                randomness_stream=randomness,
                weights=df_census_names[race_eth],
                additional_key=f"space_last_names_{seed}",
            ).to_numpy()
        )

    return last_names.to_numpy()


def get_employer_name_map(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    *_: Any,
) -> Dict[str, pd.Series]:
    """
    column_name: Name of column that is being mapped - employer_id
    obs_data: Raw results from observer outputs
    randomness: randomness stream used to assign names

    Returns: Dict with key "employer_name" and value is series of names.
    Note:  For clarity on variable names, business names refers to the generated
    business_names.  Employer names will be the chosen names that will be
    assigned to employer_ids for final results.

    """

    known_employer_names = pd.Series(
        [employer.employer_name for employer in data_values.KNOWN_EMPLOYERS],
        index=[employer.employer_id for employer in data_values.KNOWN_EMPLOYERS],
    )

    business_names_data = artifact.load(data_keys.SYNTHETIC_DATA.BUSINESS_NAMES)
    business_names_data.index = business_names_data.index + len(known_employer_names)
    business_names_data = pd.concat([known_employer_names, business_names_data])

    output_cols = [column_name]
    employer_ids = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols,
    )
    employer_names = business_names_data.loc[employer_ids.index]
    employer_name_map = {"employer_name": employer_names}
    return employer_name_map
