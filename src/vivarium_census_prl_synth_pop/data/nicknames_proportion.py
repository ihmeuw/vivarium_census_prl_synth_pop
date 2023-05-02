import numpy as np
import pandas as pd
from vivarium import Artifact

from vivarium_census_prl_synth_pop.constants import data_keys, paths


def _load_nicknames_data():
    # Load and format nicknames dataset
    nicknames = pd.read_csv(paths.NICKNAMES_DATA_PATH)
    nicknames = nicknames.apply(lambda x: x.astype(str).str.title()).set_index("name")
    nicknames = nicknames.replace("Nan", np.nan)
    return nicknames


def get_nicknames_proportion():
    # This function calculates the proportion of names that have potential nicknames. This will return a float that
    # that will be used as a constant to scale the nicknames noise function in the pseudopeople packages.

    art = Artifact(paths.DEFAULT_ARTIFACT)
    names = art.load(data_keys.SYNTHETIC_DATA.FIRST_NAMES).reset_index()["name"]
    nicknames = _load_nicknames_data()

    proportion_of_nicknames = len(nicknames.index) / len(names)
    return proportion_of_nicknames
