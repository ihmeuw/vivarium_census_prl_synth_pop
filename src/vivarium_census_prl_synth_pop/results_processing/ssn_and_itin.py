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
) -> Dict:
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
    output_cols_superset = [column_name, "has_ssn"]
    formatted_obs_data = format_data_for_mapping(
        index_name=column_name,
        obs_results=obs_data,
        output_columns=output_cols_superset,
    )
    maps.update(get_ssn_map(column_name, formatted_obs_data, randomness))
    return maps


def generate_itins(
    size: int,
    additional_key: Any,
    randomness: RandomnessStream,
) -> np.ndarray:
    """
    Generate a np.ndarray of length `size` of unique ITIN values.

        first three digits 900-999, followed by dash
        next two digits between 50-65, 70-88, 90-92, or 94-99, followed by a dash
        last four digits between 1-9999

    :param size: The number of itins to generate
    :param additional_key: Additional key to be used by randomness
    :param randomness: RandomnessStream stream to use

    """

    def generate_itin(
        count: int,
        extra_additional_key: Any,
    ):
        itin = pd.DataFrame(index=pd.Index(range(count)))

        # Area numbers have range 900-999, inclusive
        itin["area"] = random_integers(
            min_val=900,
            max_val=999,
            index=itin.index,
            randomness=randomness,
            additional_key=f"{extra_additional_key}_itin_area",
        )

        # Group numbers have range 50-65, 70-88, 90-92, or 94-99, inclusive
        itin["group"] = random_integers(
            min_val=1,
            max_val=44,  # (65-50+1)+(88-70+1)+(92-90+1)+(99-94+1)
            index=itin.index,
            randomness=randomness,
            additional_key=f"{extra_additional_key}_itin_group",
        )

        # Rescale the group digits to match expected ranges
        pre_shift_groups = itin["group"].copy()
        itin.loc[pre_shift_groups <= 16, "group"] += 49
        itin.loc[((pre_shift_groups <= 35) & (pre_shift_groups >= 17)), "group"] += 53
        itin.loc[((pre_shift_groups <= 38) & (pre_shift_groups >= 36)), "group"] += 54
        itin.loc[pre_shift_groups >= 39, "group"] += 55

        # Serial numbers 1-9999 inclusive
        itin["serial"] = random_integers(
            min_val=1,
            max_val=9999,
            index=itin.index,
            randomness=randomness,
            additional_key=f"{extra_additional_key}_itin_serial",
        )

        return (
            itin["area"].astype(str)  # no zfill needed, values should be 900-999
            + "-"
            + itin["group"].astype(str)  # no zfill needed likewise
            + "-"
            + itin["serial"].astype(str).str.zfill(4)
        ).to_numpy()

    if additional_key is None:
        additional_key = 1
    itins = pd.Series(generate_itin(size, additional_key))
    duplicate_mask = itins.duplicated()
    counter = 0
    while duplicate_mask.sum() > 0:
        if counter >= 10:
            logger.info(
                f"Resampled ITIN {counter} times and data still contains {duplicate_mask.sum()}"
                f"duplicates remaining.  Check data or usage of randomness stream."
            )
            break
        new_additional_key = f"{additional_key}_{counter}"
        itins.loc[duplicate_mask] = generate_itin(duplicate_mask.sum(), new_additional_key)
        duplicate_mask = itins.duplicated()
        counter += 1
    return itins.to_numpy()


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

    def generate_ssn(
        count: int,
        extra_additional_key: Any,
    ) -> pd.DataFrame:
        ssn = pd.DataFrame(index=pd.Index(range(count)))

        area = random_integers(
            min_val=1,
            max_val=899,
            index=ssn.index,
            randomness=randomness,
            additional_key=f"{extra_additional_key}_ssn_area",
        )
        area = np.where(area == 666, 667, area)
        ssn["ssn_area"] = area

        group = random_integers(
            min_val=1,
            max_val=99,
            index=ssn.index,
            randomness=randomness,
            additional_key=f"{extra_additional_key}_ssn_group",
        )
        ssn["ssn_group"] = group

        serial = random_integers(
            min_val=1,
            max_val=9999,
            index=ssn.index,
            randomness=randomness,
            additional_key=f"{extra_additional_key}_ssn_serial",
        )
        ssn["ssn_serial"] = serial

        return (
            ssn["ssn_area"].astype(str).str.zfill(3)
            + "-"
            + ssn["ssn_group"].astype(str).str.zfill(2)
            + "-"
            + ssn["ssn_serial"].astype(str).str.zfill(4)
        ).to_numpy()

    if additional_key is None:
        additional_key = 1
    ssns = pd.Series(generate_ssn(size, additional_key))
    duplicate_mask = ssns.duplicated()
    counter = 0
    while duplicate_mask.sum() > 0:
        if counter >= 10:
            logger.info(
                f"Resampled SSN {counter} times and data still contains {duplicate_mask.sum()}"
                f"duplicates remaining.  Check data or usage of randomness stream."
            )
            break
        new_additional_key = f"{additional_key}_{counter}"
        ssns.loc[duplicate_mask] = generate_ssn(duplicate_mask.sum(), new_additional_key)
        duplicate_mask = ssns.duplicated()
        counter += 1
    return ssns.to_numpy()


def get_ssn_map(
    column_name: str,
    obs_data: pd.DataFrame,
    randomness: RandomnessStream,
):
    ssn_map_dict = {}
    output_cols = [column_name, "has_ssn"]  # columns in the output we use to map
    simulant_data = (
        obs_data.reset_index()[output_cols].drop_duplicates().set_index(column_name)
    )
    ssn_map = pd.Series("", index=simulant_data.index)
    ssn_map[simulant_data["has_ssn"]] = generate_ssns(
        simulant_data["has_ssn"].sum(), "get_ssn_map", randomness
    )

    # Map against obs_data
    ssn_map_dict["ssn"] = ssn_map.fillna("").astype(str)
    return ssn_map_dict
