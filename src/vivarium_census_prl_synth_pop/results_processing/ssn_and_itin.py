from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)
from vivarium_census_prl_synth_pop.utilities import random_integers


def get_simulant_id_maps(
    column_name: str,
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
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
    randomness
        RandomnessStream to use in choosing zipcodes proportionally

    Returns
    -------
    A dictionary of pd.Series suitable for pd.Series.map, indexed by `address_id`

    """
    if column_name != "simulant_id":
        raise ValueError(f"Expected `simulant_id`, got `{column_name}`")
    maps = dict()
    # ssn
    maps.update(get_ssn_map(obs_data, column_name, artifact))
    # itin
    output_cols_superset = [column_name, "has_ssn"]
    formatted_obs_data = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols_superset,
    )
    maps.update(get_itin_map(column_name, formatted_obs_data, artifact))

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
    obs_data: pd.DataFrame,
    column_name: str,
    artifact: Artifact,
) -> Dict[str, pd.Series]:
    # Anyone in the SSN observer has SSNs by definition
    need_ssns_ssn = obs_data["social_security_observer"][column_name]

    # Simulants in W2 observer who `has_ssn` need SSNs
    w2_data = obs_data["tax_w2_observer"][[column_name, "has_ssn", "ssn_id"]]
    need_ssns_has_ssn = w2_data.loc[w2_data["has_ssn"], column_name]

    # Simulants in W2 observer who are assigned as an `ssn_id` need to have SSNs
    # to be copied later
    need_ssns_ssn_id = w2_data.loc[w2_data["ssn_id"] != "-1", "ssn_id"]

    need_ssns = pd.concat(
        [need_ssns_ssn, need_ssns_has_ssn, need_ssns_ssn_id], axis=0
    ).drop_duplicates()

    # Load (already-shuffled) SSNs and choose the appropriate number off the top
    # NOTE: artifact.load does not currently have the option to select specific rows
    # to load and so that was much slower than using pandas.
    ssns = pd.read_hdf(
        artifact.path,
        key="/synthetic_data/ssns",
        start=0,
        stop=len(need_ssns),
    )

    # Convert to hyphenated string IDs
    ssns = (
        ssns["area"].astype(str).str.zfill(3)
        + "-"
        + ssns["group"].astype(str).str.zfill(2)
        + "-"
        + ssns["serial"].astype(str).str.zfill(4)
    ).to_numpy()

    return {"ssn": pd.Series(ssns, index=need_ssns, name="ssn")}


def get_itin_map(
    column_name: str,
    obs_data: pd.DataFrame,
    artifact: Artifact,
):
    itin_map_dict = {}
    output_cols = [column_name, "has_ssn"]  # columns in the output we use to map
    simulant_data = (
        obs_data.reset_index()[output_cols].drop_duplicates().set_index(column_name)
    )
    # Load (already-shuffled) ITINs and choose the appropriate number off the top
    # NOTE: artifact.load does not currently have the option to select specific rows
    # to load and so that was much slower than using pandas.
    itin_mask = ~simulant_data["has_ssn"]
    itins = pd.read_hdf(
        artifact.path,
        key="/synthetic_data/itins",
        start=0,
        stop=itin_mask.sum(),
    )

    itins = _convert_to_hyphenated_strings(itins)

    # Assign ITINs and create dictionary item
    itin_map = pd.Series("", index=simulant_data.index)
    itin_map[itin_mask] = itins
    itin_map_dict["itin"] = itin_map

    return itin_map_dict


def _convert_to_hyphenated_strings(ids: pd.DataFrame) -> np.array:
    return (
        ids["area"].astype(str).str.zfill(3)
        + "-"
        + ids["group"].astype(str).str.zfill(2)
        + "-"
        + ids["serial"].astype(str).str.zfill(4)
    ).to_numpy()


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
    # TODO: Replace with sampling from the artifact data if we figure out a
    # more performant way of doing so (eg querying the hdf with randomized idx)
    obs_data.loc[create_ssn_mask, "ssn"] = generate_ssns(
        size=create_ssn_mask.sum(),
        additional_key="collide",
        randomness=randomness,
    )
    return obs_data
