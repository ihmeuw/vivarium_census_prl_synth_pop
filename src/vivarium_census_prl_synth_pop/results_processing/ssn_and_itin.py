from typing import Any

import numpy as np
import pandas as pd
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.utilities import random_integers


def generate_itin(
    size: int,
    additional_key: Any,
    randomness: RandomnessStream,
) -> np.ndarray:
    """
    Generate a np.ndarray of length `size` of hopefully unique ITIN values.

        first three digits 900-999, followed by dash
        next two digits between 50-65, 70-88, 90-92, or 94-99, followed by a dash
        last four digits between 1-9999

    :param size: The number of itins to generate
    :param additional_key: Additional key to be used by randomness
    :param randomness: RandomnessStream stream to use

    """
    itin = pd.DataFrame(index=pd.Index(range(size)))

    # Area numbers have range 900-999, inclusive
    itin["area"] = random_integers(
        min_val=900,
        max_val=999,
        index=itin.index,
        randomness=randomness,
        additional_key=f"{additional_key}_itin_area",
    )

    # Group numbers have range 50-65, 70-88, 90-92, or 94-99, inclusive
    itin["group"] = random_integers(
        min_val=1,
        max_val=44,  # (65-50+1)+(88-70+1)+(92-90+1)+(99-94+1)
        index=itin.index,
        randomness=randomness,
        additional_key=f"{additional_key}_itin_group",
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
        additional_key=f"{additional_key}_itin_serial",
    )
    itin["itin"] = ""
    itin["itin"] += (
        itin["area"].astype(str) + "-"
    )  # no zfill needed, values should be 900-999
    itin["itin"] += itin["group"].astype(str) + "-"  # no zfill needed likewise
    itin["itin"] += itin["serial"].astype(str).str.zfill(4)

    return itin["itin"].to_numpy()
