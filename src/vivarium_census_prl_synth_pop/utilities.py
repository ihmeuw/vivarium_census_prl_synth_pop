import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.randomness import Array, RandomnessStream, get_hash
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks.data_transformations import pivot_categorical

from vivarium_census_prl_synth_pop.constants import metadata

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


def read_data_by_draw(artifact_path: str, key: str, draw: int) -> pd.DataFrame:
    """Reads data from the artifact on a per-draw basis. This
    is necessary for Low Birthweight Short Gestation (LBWSG) data.

    Parameters
    ----------
    artifact_path
        The artifact to read from.
    key
        The entity key associated with the data to read.
    draw
        The data to retrieve.

    """
    key = key.replace(".", "/")
    with pd.HDFStore(artifact_path, mode="r") as store:
        index = store.get(f"{key}/index")
        draw = store.get(f"{key}/draw_{draw}")
    draw = draw.rename("value")
    data = pd.concat([index, draw], axis=1)
    data = data.drop(columns="location")
    data = pivot_categorical(data)
    data[
        project_globals.LBWSG_MISSING_CATEGORY.CAT
    ] = project_globals.LBWSG_MISSING_CATEGORY.EXPOSURE
    return data


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
    options: Array,
    n_to_choose: int,
    randomness_stream: RandomnessStream,
    weights: Array = None,
    additional_key: Any = None,
):
    if weights is None:
        n = len(options)
        weights = np.ones(n) / n
    # for each of n_to_choose, sample uniformly between 0 and 1
    probs = randomness_stream.get_draw(np.arange(n_to_choose), additional_key=additional_key)

    # build cdf based on weights
    pmf = weights / weights.sum()
    cdf = np.cumsum(pmf)

    # for each p_i in probs, count how many elements of cdf for which p_i >= cdf_i
    vect_find_index = np.vectorize(lambda p_i: (p_i >= cdf).sum())
    chosen_indices = vect_find_index(probs)
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


def build_output_dir(builder: Builder, subdir: Optional[Union[str, Path]] = None) -> Path:
    output_dir = Path(builder.configuration.output_data.results_directory)
    if subdir:
        output_dir = output_dir / subdir

    old_umask = os.umask(0o002)
    try:
        output_dir.mkdir(exist_ok=True)
    finally:
        os.umask(old_umask)

    return output_dir


def update_address_id(
    df: pd.DataFrame,
    rows_to_update: pd.Index,
    starting_address_id: int,
    address_id_col_name: str = "address_id",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df: the pd.DataFrame to update, business table
    rows_to_update: a pd.Index of the rows of df to update
    starting_address_id: int that is tracking value for business or household_ids max value.:
    address_id_col_name: a string. the name of the column in df to hold addresses.
    Returns
    -------
    df with appropriately updated address_ids
    """
    df.loc[rows_to_update, address_id_col_name] = starting_address_id + np.arange(
        len(rows_to_update)
    )
    return df


def update_address_id_for_unit_and_sims(
    pop: pd.DataFrame,
    moving_units: pd.DataFrame,
    units_that_move_ids: pd.Index,
    starting_address_id: int,
    unit_id_col_name: str,
    address_id_col_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Units are multiperson groups tracked in the simulation.  Examples being households and employers.

    Parameters
    ----------
    pop: population table
    moving_units: Dataframe with column address_id_col_name.
    units_that_move_ids: IDs of moving units.  This is a subset of moving_units.index
    starting_address_id: Integer at which to start generating new address_ids, to prevent collisions.
    unit_id_col_name: Column name in state table where ids for the unit are stored.
    address_id_col_name: Column name in state table where address_id for the unit is stored.

    Returns
    -------
    pop: Updated version of the state table.
    moving_units: Updated version of units dataframe.  This is done for the purpose of the businesses table.
    starting_address_id: Updated integer at which to start when generating more address_ids.
    """

    if len(units_that_move_ids) > 0:
        # update the employer address_id in self.businesses
        # this will update address_id in a households dataframe we are not tracking like we are businesses
        # todo:  Should add income/salary to mapping function here
        moving_units = update_address_id(
            df=moving_units,
            rows_to_update=units_that_move_ids,
            starting_address_id=starting_address_id,
            address_id_col_name=address_id_col_name,
        )
        starting_address_id += len(units_that_move_ids)

        # update address_id column in the pop table
        rows_changing_address_id_idx = pop.loc[
            pop[unit_id_col_name].isin(units_that_move_ids)
        ].index
        # Preserve pop index
        pop = pop.reset_index().rename(columns={"index": "simulant_id"})
        # todo:  Should add income/salary to mapping function here
        updated_address_ids = (
            pop[["simulant_id", unit_id_col_name]]
            .merge(
                moving_units[[address_id_col_name]],
                how="left",
                left_on=unit_id_col_name,
                right_on=moving_units.index,
            )
            .set_index("simulant_id")[address_id_col_name]
        )
        pop = pop.set_index("simulant_id")
        pop.loc[rows_changing_address_id_idx, address_id_col_name] = updated_address_ids

    return pop, moving_units, starting_address_id
