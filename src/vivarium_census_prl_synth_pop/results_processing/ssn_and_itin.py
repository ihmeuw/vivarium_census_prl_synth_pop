from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import metadata
from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)
from vivarium_census_prl_synth_pop.utilities import random_integers


def get_simulant_id_maps(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    _: Any,
    seed: str,
    all_seeds: List[str],
) -> Dict[str, pd.Series]:
    """
    Get all maps that are indexed by `simulant_id`.

    Parameters
    ----------
    column_name
        Name of the column to use as an index
    obs_data
        Observer DataFrame with key for the observer name
    artifact
        A vivarium Artifact object needed by mapper
    seed
        The random seed of the simulation of interest
    all_seeds
        List of all seeds found in raw results

    Returns
    -------
    A dictionary of pd.Series suitable for pd.Series.map, indexed by `address_id`

    """
    logger.info(f"Generating {column_name} maps")
    if column_name != "simulant_id":
        raise ValueError(f"Expected `simulant_id`, got `{column_name}`")
    maps = dict()
    maps.update(get_ssn_map(obs_data, column_name, artifact, seed, all_seeds))
    maps.update(
        get_ssn_itin_map(obs_data, column_name, artifact, seed, all_seeds, maps["ssn"])
    )

    return maps


def generate_ssns(
    size: int,
    additional_key: Any,
    randomness: RandomnessStream,
) -> np.ndarray:
    """
    Generate a np.ndarray of length `size` of unique SSN values.

        first three digits 1 and 899 (inclusive), 666 disallowed, followed by dash
        next two digits between 1-99 (inclusive), followed by a dash
        last four digits between 1-9999 (inclusive)

    :param size: The number of itins to generate
    :param additional_key: Additional key to be used by randomness
    :param randomness: RandomnessStream stream to use
    """
    ssn = pd.DataFrame(index=pd.Index(range(size)))

    area = random_integers(
        min_val=1,
        max_val=899,
        index=ssn.index,
        randomness=randomness,
        additional_key=f"{additional_key}_ssn_area",
    )
    area = np.where(area == 666, 667, area)
    ssn["ssn_area"] = area

    group = random_integers(
        min_val=1,
        max_val=99,
        index=ssn.index,
        randomness=randomness,
        additional_key=f"{additional_key}_ssn_group",
    )
    ssn["ssn_group"] = group

    serial = random_integers(
        min_val=1,
        max_val=9999,
        index=ssn.index,
        randomness=randomness,
        additional_key=f"{additional_key}_ssn_serial",
    )
    ssn["ssn_serial"] = serial

    ssns = (
        ssn["ssn_area"].astype(str).str.zfill(3)
        + "-"
        + ssn["ssn_group"].astype(str).str.zfill(2)
        + "-"
        + ssn["ssn_serial"].astype(str).str.zfill(4)
    ).to_numpy()

    return ssns


def get_ssn_map(
    obs_data: Dict[str, pd.DataFrame],
    column_name: str,
    artifact: Artifact,
    seed: str,
    all_seeds: List[str],
) -> Dict[str, pd.Series]:
    # Anyone in the SSN observer has SSNs by definition
    need_ssns_ssn = obs_data[metadata.DatasetNames.SSA][column_name]

    # Simulants in W2 observer who `has_ssn` need SSNs
    w2_data = obs_data[metadata.DatasetNames.TAXES_W2_1099][
        [column_name, "has_ssn", "ssn_id", "copy_ssn"]
    ]
    need_ssns_has_ssn = w2_data.loc[w2_data["has_ssn"], column_name]

    # Simulants in W2 observer who are assigned as an `ssn_id` need to have SSNs
    # to be copied later
    need_ssns_ssn_id = w2_data.loc[w2_data["ssn_id"] != "-1", "ssn_id"]
    need_ssns = pd.concat(
        [need_ssns_ssn, need_ssns_has_ssn, need_ssns_ssn_id], axis=0
    ).drop_duplicates()
    ssns = _load_ids(artifact, "/synthetic_data/ssns", len(need_ssns), seed, all_seeds)

    return {"ssn": pd.Series(ssns, index=need_ssns, name="ssn")}


def get_ssn_itin_map(
    obs_data: Dict[str, pd.DataFrame],
    column_name: str,
    artifact: Artifact,
    seed: str,
    all_seeds: List[str],
    ssns: pd.Series,
):
    formatted_obs_data = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=[column_name, "has_ssn"],
    )
    simulant_data = formatted_obs_data.reset_index().drop_duplicates().set_index(column_name)
    itin_mask = ~simulant_data["has_ssn"]
    itins = _load_ids(artifact, "/synthetic_data/itins", itin_mask.sum(), seed, all_seeds)
    itins = pd.Series(itins, index=simulant_data.index[itin_mask], name="itin")
    ssn_itins = pd.concat([ssns, itins])
    ssn_itins.name = "ssn_itin"
    return {"ssn_itin": ssn_itins}


def _load_ids(
    artifact: Artifact, hdf_key: str, num_need_ids: int, seed: str, all_seeds: List[str]
) -> np.array:
    """Load (already-shuffled) IDs from unique-by-seed chunks of the full
    artifact data and convert to hyphenated strings
    """
    # NOTE: `artifact.load` does not currently have the option to select specific rows
    # to load and so it's much quicker to use `pd.HDFStore.select`
    with pd.HDFStore(artifact.path, mode="r") as store:
        num_available_ids = store.get_storer(hdf_key).nrows
        if seed == "":  # all seeds are already concatenated together so just pick off the top
            start_idx = 0
            chunksize = num_available_ids
        else:  # we are dealing with a single seed
            chunksize = int(num_available_ids / len(all_seeds))
            seed_position = all_seeds.index(seed)
            start_idx = seed_position * chunksize
        if num_need_ids > chunksize:
            raise IndexError(
                f"You are requesting {num_need_ids} IDs from seed {seed}'s unique "
                f"chunk, but this chunk only has {chunksize} available (hdf_key "
                f"{hdf_key} and {len(all_seeds)} simulation seeds)."
            )
        ids = store.select(
            hdf_key,
            start=start_idx,
            stop=start_idx + num_need_ids,
        )
    # Convert id section columns to hyphenated string IDs
    ids = (
        ids["area"].astype(str).str.zfill(3)
        + "-"
        + ids["group"].astype(str).str.zfill(2)
        + "-"
        + ids["serial"].astype(str).str.zfill(4)
    ).to_numpy()

    return ids


def do_collide_ssns(
    obs_data: pd.DataFrame, ssn_map: pd.Series, randomness: RandomnessStream
) -> pd.DataFrame:
    """Apply SSN collision to simulants with no SSN but provided an SSN to an employer."""
    # If a household member's SSN will be used, use ssn_map to map
    use_ssn_id_mask = (
        (obs_data["ssn_id"] != "-1")
        & (obs_data["employer_id"] != -1)
        & (~obs_data["has_ssn"])
    )
    obs_data.loc[use_ssn_id_mask, "ssn"] = obs_data[use_ssn_id_mask]["ssn_id"].map(ssn_map)

    # If a household member's SSN is unavailable, generate a new SSN, duplicates don't matter.
    create_ssn_mask = (
        (obs_data["ssn_id"] == "-1")
        & (obs_data["employer_id"] != -1)
        & (~obs_data["has_ssn"])
    )
    obs_data.loc[create_ssn_mask, "ssn"] = generate_ssns(
        size=create_ssn_mask.sum(),
        additional_key="collide",
        randomness=randomness,
    )
    return obs_data


def copy_ssn_from_household_member(
    ssn_ids_to_copy: pd.Series, ssn_map: pd.Series
) -> pd.Series:
    # Copy ssns for copy from household member
    copied_ssns = ssn_ids_to_copy.map(ssn_map)
    return copied_ssns
