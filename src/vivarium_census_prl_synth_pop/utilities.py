import os
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.randomness import RandomnessStream, get_hash
from vivarium.framework.values import Pipeline

from vivarium_census_prl_synth_pop.constants import (
    data_keys,
    data_values,
    metadata,
    paths,
)

SeededDistribution = Tuple[str, stats.rv_continuous]


def len_longest_location() -> int:
    """Returns the length of the longest location in the project.

    Returns
    -------
       Length of the longest location in the project.
    """
    return len(max(metadata.LOCATIONS, key=len))


def sanitize_location(location: str):
    """Cleans up location formatting for writing and reading from file names.

    Parameters
    ----------
    location
        The unsanitized location name.

    Returns
    -------
        The sanitized location name (lower-case with white-space and
        special characters removed.

    """
    # FIXME: Should make this a reversible transformation.
    return location.replace(" ", "_").replace("'", "_").lower()


def delete_if_exists(*paths: Union[Path, List[Path]], confirm=False):
    paths = paths[0] if isinstance(paths[0], list) else paths
    existing_paths = [p for p in paths if p.exists()]
    if existing_paths:
        if confirm:
            # Assumes all paths have the same root dir
            root = existing_paths[0].parent
            names = [p.name for p in existing_paths]
            click.confirm(
                f"Existing files {names} found in directory {root}. Do you want to delete and replace?",
                abort=True,
            )
        for p in existing_paths:
            logger.info(f"Deleting artifact at {str(p)}.")
            p.unlink()


def get_norm(
    mean: float,
    sd: float = None,
    ninety_five_pct_confidence_interval: Tuple[float, float] = None,
) -> stats.norm:
    sd = _get_standard_deviation(mean, sd, ninety_five_pct_confidence_interval)
    return stats.norm(loc=mean, scale=sd)


def get_truncnorm(
    mean: float,
    sd: float = None,
    ninety_five_pct_confidence_interval: Tuple[float, float] = None,
    lower_clip: float = 0.0,
    upper_clip: float = 1.0,
) -> stats.norm:
    sd = _get_standard_deviation(mean, sd, ninety_five_pct_confidence_interval)
    a = (lower_clip - mean) / sd if sd else mean - 1e-03
    b = (upper_clip - mean) / sd if sd else mean + 1e03
    return stats.truncnorm(loc=mean, scale=sd, a=a, b=b)


def _get_standard_deviation(
    mean: float, sd: float, ninety_five_pct_confidence_interval: Tuple[float, float]
) -> float:
    if sd is None and ninety_five_pct_confidence_interval is None:
        raise ValueError(
            "Must provide either a standard deviation or a 95% confidence interval."
        )
    if sd is not None and ninety_five_pct_confidence_interval is not None:
        raise ValueError(
            "Cannot provide both a standard deviation and a 95% confidence interval."
        )
    if ninety_five_pct_confidence_interval is not None:
        lower = ninety_five_pct_confidence_interval[0]
        upper = ninety_five_pct_confidence_interval[1]
        if not (lower <= mean <= upper):
            raise ValueError(
                f"The mean ({mean}) must be between the lower ({lower}) and upper ({upper}) "
                "quantile values."
            )

        stdnorm_quantiles = stats.norm.ppf((0.025, 0.975))
        sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return sd


def get_lognorm_from_quantiles(
    median: float, lower: float, upper: float, quantiles: Tuple[float, float] = (0.025, 0.975)
) -> stats.lognorm:
    """Returns a frozen lognormal distribution with the specified median, such that
    (lower, upper) are approximately equal to the quantiles with ranks
    (quantile_ranks[0], quantile_ranks[1]).
    """
    # Let Y ~ norm(mu, sigma^2) and X = exp(Y), where mu = log(median)
    # so X ~ lognorm(s=sigma, scale=exp(mu)) in scipy's notation.
    # We will determine sigma from the two specified quantiles lower and upper.
    if not (lower <= median <= upper):
        raise ValueError(
            f"The median ({median}) must be between the lower ({lower}) and upper ({upper}) "
            "quantile values."
        )

    # mean (and median) of the normal random variable Y = log(X)
    mu = np.log(median)
    # quantiles of the standard normal distribution corresponding to quantile_ranks
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    # quantiles of Y = log(X) corresponding to the quantiles (lower, upper) for X
    norm_quantiles = np.log([lower, upper])
    # standard deviation of Y = log(X) computed from the above quantiles for Y
    # and the corresponding standard normal quantiles
    sigma = (norm_quantiles[1] - norm_quantiles[0]) / (
        stdnorm_quantiles[1] - stdnorm_quantiles[0]
    )
    # Frozen lognormal distribution for X = exp(Y)
    # (s=sigma is the shape parameter; the scale parameter is exp(mu), which equals the median)
    return stats.lognorm(s=sigma, scale=median)


def get_random_variable_draws(columns: pd.Index, seed: str, distribution) -> pd.Series:
    return pd.Series(
        [get_random_variable(x, seed, distribution) for x in range(0, columns.size)],
        index=columns,
    )


def get_random_variable(draw: int, seed: str, distribution) -> pd.Series:
    np.random.seed(get_hash(f"{seed}_draw_{draw}"))
    return distribution.rvs()


def get_random_variable_draws_for_location(
    columns: pd.Index, location: str, seed: str, distribution
) -> np.array:
    return get_random_variable_draws(columns, f"{seed}_{location}", distribution)


def get_norm_from_quantiles(
    mean: float, lower: float, upper: float, quantiles: Tuple[float, float] = (0.025, 0.975)
) -> stats.norm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return stats.norm(loc=mean, scale=sd)


def vectorized_choice(
    options: Union[pd.Series, List],
    n_to_choose: int,
    randomness_stream: RandomnessStream = None,
    weights: Optional[Union[pd.Series, List]] = None,
    additional_key: Any = None,
    random_seed: int = None,
):
    if not randomness_stream and (additional_key == None and random_seed == None):
        raise RuntimeError(
            "An additional_key and a random_seed are required in 'vectorized_choice'"
            + "if no RandomnessStream is passed in"
        )
    if weights is None:
        n = len(options)
        weights = np.ones(n) / n
    # for each of n_to_choose, sample uniformly between 0 and 1
    index = pd.Index(np.arange(n_to_choose))
    if randomness_stream is None:
        # Generate an additional_key on-the-fly and use that in randomness.random
        random_state = np.random.RandomState(seed=get_hash(f"{additional_key}_{random_seed}"))
        raw_draws = random_state.random_sample(len(index))
        probs = pd.Series(raw_draws, index=index)
    else:
        probs = randomness_stream.get_draw(index, additional_key=additional_key)

    # build cdf based on weights
    pmf = weights / weights.sum()
    cdf = np.cumsum(pmf)

    # for each p_i in probs, count how many elements of cdf for which p_i >= cdf_i
    chosen_indices = np.searchsorted(cdf, probs, side="right")
    return np.take(options, chosen_indices)


def random_integers(
    min_val: int,
    max_val: int,
    index: pd.Index,
    randomness: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    Parameters
    ----------
    min_val
        inclusive
    max_val
        inclusive
    index
        an index whose length is the number of random draws made
        and which indexes the returned `pandas.Series`.
    randomness:
        RandomnessStream
    additional_key:
        A identifying key for the randomness stream (usually a string)

    Returns
    -------
    pandas.Series
        An indexed set of integers in the interval [a,b]
    """
    return np.floor(
        randomness.get_draw(index=index, additional_key=additional_key)
        * (max_val + 1 - min_val)
        + min_val
    ).astype(int)


def filter_by_rate(
    entity_to_filter: Union[pd.Index, pd.Series],
    randomness: RandomnessStream,
    rate_producer: Union[LookupTable, Pipeline],
    additional_key: Any = None,
) -> pd.Index:
    """
    Parameters
    ----------
    entity_to_filter: a series of every entity that might move. not necessarily unique.  Can be a list of ids or an
        index (household_ids, business_ids, or pandas index or simulants who may move).
    rate_producer: rate_producer for move rates
    randomness: RandomnessStream for component this is being run in
    additional_key: descriptive key to make sure randomness stream produces unique results

    Returns
    -------
    a pd.Index, subset from simulants, with those selected to be filtered.
    """
    entity_to_filter = entity_to_filter.drop_duplicates()
    if type(entity_to_filter) is pd.Series:
        idx = entity_to_filter.index
    else:
        idx = entity_to_filter

    filtered_sims = randomness.filter_for_rate(
        entity_to_filter, rate_producer(idx), additional_key
    )
    return filtered_sims


def build_output_dir(output_dir: Path, subdir: Optional[Union[str, Path]] = None) -> Path:
    if subdir:
        output_dir = output_dir / subdir

    old_umask = os.umask(0o002)
    try:
        output_dir.mkdir(exist_ok=True, parents=True)
    finally:
        os.umask(old_umask)

    return output_dir


def build_final_results_directory(
    results_dir: str,
    version: Optional[str] = None,
) -> Tuple[Path, Path]:
    tmp = paths.FINAL_RESULTS_DIR_NAME / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if version is None:
        subdir = tmp / "pseudopeople_input_data_usa"
    else:
        subdir = tmp / f"pseudopeople_input_data_usa_{version}"
    final_output_dir = build_output_dir(
        Path(results_dir),
        subdir=subdir,
    )
    raw_output_dir = Path(results_dir) / paths.RAW_RESULTS_DIR_NAME

    return raw_output_dir, final_output_dir


def add_guardian_address_ids(pop: pd.DataFrame) -> pd.DataFrame:
    """Map the address ids of guardians to each simulant's guardian address columns"""
    for i in [1, 2]:
        s_guardian_id = pop[f"guardian_{i}"]
        # is it faster to remove the negative values below?
        s_guardian_id = s_guardian_id[s_guardian_id != -1]
        pop[f"guardian_{i}_address_id"] = s_guardian_id.map(pop["address_id"])
    return pop


def convert_middle_name_to_initial(pop: pd.DataFrame) -> pd.DataFrame:
    """Converts middle names to middle initials. Note that this drops
    the 'middle_name' column altogether and replaces it with 'middle_initial'
    """
    pop["middle_initial"] = pop["middle_name"].str[0]
    pop = pop.drop(columns="middle_name")
    return pop


def sample_acs_standard_households(
    target_number_sims: Optional[int],
    acs_households: pd.DataFrame,
    acs_persons: pd.DataFrame,
    randomness: RandomnessStream,
    num_households: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples households from ACS using household weights and gets the associated persons.
    The chosen people include households-file columns (pre-merged).
    The acs_sample_household_id in both returned dataframes (chosen households and persons)
    represents the unique household sampling event, even if multiple instances of
    the same ACS household were sampled.

    Requires either target_number_sims or num_households to be provided.
    In the former case, aims for the target_number_sims and slightly undershoots it if it doesn't come out to
    a round number of households.
    In the latter case, samples the given number of households, regardless of how many people are in them.
    """
    if (num_households is None) == (target_number_sims is None):
        raise ValueError(
            "Exactly one of num_households and target_number_sims should be provided."
        )

    if num_households is None:
        # oversample households -- each household has at least one person,
        # so if we get as many households as we need people, we will always
        # have enough people (and probably far too many)
        num_households = target_number_sims

    chosen_households_index = vectorized_choice(
        options=acs_households.index,
        n_to_choose=num_households,
        randomness_stream=randomness,
        weights=acs_households["household_weight"],
        additional_key="choose_standard_households",
    )
    chosen_households = acs_households.loc[chosen_households_index]

    # create unique id for resampled households -- each census_household_id
    # can be sampled multiple times.
    chosen_households["acs_sample_household_id"] = np.arange(len(chosen_households))

    # get all simulants per household
    chosen_persons = pd.merge(
        chosen_households,
        acs_persons[metadata.PERSONS_COLUMNS_TO_INITIALIZE],
        on="census_household_id",
        how="left",
    )

    if target_number_sims is not None:
        # get rid of simulants and households in excess of desired pop size
        households_to_discard = chosen_persons.loc[
            target_number_sims:, "acs_sample_household_id"
        ].unique()
        chosen_persons = chosen_persons.loc[
            ~chosen_persons["acs_sample_household_id"].isin(households_to_discard)
        ]
        chosen_households = chosen_households.loc[
            ~chosen_households["acs_sample_household_id"].isin(households_to_discard)
        ]

    return chosen_households, chosen_persons


def sample_acs_persons(
    target_number_sims: int,
    acs_persons: pd.DataFrame,
    randomness: RandomnessStream,
    additional_key: str,
) -> pd.DataFrame:
    """
    Samples persons from ACS individually with person weights.
    """

    chosen_persons_index = vectorized_choice(
        options=acs_persons.index,
        n_to_choose=target_number_sims,
        randomness_stream=randomness,
        weights=acs_persons["person_weight"],
        additional_key=additional_key,
    )

    return acs_persons.loc[chosen_persons_index].reset_index(drop=True)


def randomly_sample_states_pumas(
    unit_ids: pd.Series,
    state_puma_options: pd.DataFrame,
    additional_key: str,
    randomness_stream: RandomnessStream = None,
    random_seed: int = None,
) -> pd.DataFrame:
    """Randomly sample new states and pumas from the raw data file.

    Args:
        unit_ids (pd.Series): household_ids or business_ids to sample for
        state_puma_options (pd.DataFrame): dataframe with state and puma columns
        additional_key (str): standard RandomnessStream additional_key
        randomness_stream (RandomnessStream, optional): Defaults to None.
        random_seed (int, optional): Only used when trying to call
            vectorized_choice when no RandomnessStream is available
            (eg during setup). Defaults to None.

    Returns:
        pd.DataFrame: Index are the unit_ids and has columns [['state_id', 'puma']]
    """
    if random_seed is None and randomness_stream is None:
        raise RuntimeError("A randomness_stream or random_seed must be provided")
    if random_seed is not None and randomness_stream is not None:
        raise RuntimeError("Only one of randomness_stream or random_seed should be provided")

    states_pumas_idx = vectorized_choice(
        options=state_puma_options.index,
        n_to_choose=len(unit_ids),
        randomness_stream=randomness_stream,
        additional_key=additional_key,
        random_seed=random_seed,
    )
    states_pumas = state_puma_options.loc[states_pumas_idx]
    states_pumas.index = unit_ids
    states_pumas = states_pumas.rename(columns={"state": "state_id"})

    return states_pumas


def get_state_puma_options(builder: Builder) -> pd.DataFrame:
    states_in_artifact = list(
        builder.data.load(data_keys.POPULATION.HOUSEHOLDS)["state"].drop_duplicates()
    )
    state_puma_options = pd.read_csv(paths.PUMA_TO_ZIP_DATA_PATH)[
        ["state", "puma"]
    ].drop_duplicates()
    # Subset to only states that exist in the artifact
    state_puma_options = state_puma_options[
        state_puma_options["state"].isin(states_in_artifact)
    ]

    return state_puma_options


def get_all_simulation_seeds(raw_output_dir: Path) -> List[str]:
    raw_results_files = list(
        chain(*[raw_output_dir.rglob(f"*.{ext}") for ext in metadata.SUPPORTED_EXTENSIONS])
    )

    return sorted(list(set([x.name.split(".")[0].split("_")[-1] for x in raw_results_files])))


def write_to_disk(data: pd.DataFrame, path: Path):
    """
    Converts all object dtypes to categorical and then writes to file to output
    path. If writing to an hdf file, bzip2 compression is used. Alternately can
    write to a parquet file.
    """
    for column in data.columns:
        if data[column].dtype.name == "object":
            data[column] = data[column].astype("category")
    if ".hdf" == path.suffix:
        data.to_hdf(
            path,
            "data",
            format="table",
            complib="bzip2",
            complevel=9,
            data_columns=data_values.DATA_COLUMNS,
        )
    elif ".parquet" == path.suffix:
        data.to_parquet(path)
    else:
        raise ValueError(
            f"Supported extensions are {metadata.SUPPORTED_EXTENSIONS}. "
            f"{path.suffix[1:]} was provided."
        )


def copy_from_household_member(
    pop: pd.DataFrame, randomness_stream: RandomnessStream
) -> pd.DataFrame:
    # Creates copy_age, copy_date_of_birth, and copy_ssn from household members
    # Note: copy value can be original value but copies from another household member

    copy_cols = metadata.COPY_HOUSEHOLD_MEMBER_COLS
    for col in [column for column in copy_cols.keys() if column in pop.columns]:
        pop[copy_cols[col]] = np.nan
        if col == "has_ssn":
            # Subset to rows that have SSN so we do not try and map nans.
            households = (
                pop[pop[col]].groupby("household_id").apply(lambda df_g: list(df_g.index))
            )
        else:
            # This makes the assumption age and date of birth will not have nans at this poitn
            households = pop.groupby("household_id").apply(lambda df_g: list(df_g.index))
        # Drop GQ and households with single member
        households = households[households.apply(len) > 1]
        if not households.empty:
            households = households.loc[
                households.index.difference(data_values.GQ_HOUSING_TYPE_MAP.keys())
            ]
            simulants_and_household_members = pop["household_id"].map(households).dropna()
            # Note the following call results in a dataframe if 'simulants_and_household_members' is empty
            # and is handled above
            simulant_ids_to_copy = simulants_and_household_members.reset_index().apply(
                lambda row: [
                    household
                    for household in row.loc["household_id"]
                    if household != row.loc["index"]
                ],
                axis=1,
            )
            simulant_ids_to_copy.index = simulants_and_household_members.index
            seed = get_hash(randomness_stream._key(additional_key=col))
            copy_ids = simulant_ids_to_copy.map(np.random.default_rng(seed).choice)
            if col == "has_ssn":
                pop.loc[copy_ids.index, copy_cols[col]] = copy_ids
            else:
                pop.loc[copy_ids.index, copy_cols[col]] = copy_ids.map(
                    pop.loc[copy_ids.index, col]
                )
        else:
            # Save as object type - current pandas defaults to dtype float with future warning
            pop[copy_cols[col]] = pd.Series(np.nan, index=pop.index, dtype=object)

    return pop
