from itertools import chain
from pathlib import Path

import pandas as pd
from vivarium import Artifact

from vivarium_census_prl_synth_pop.constants import data_keys, paths


def get_nicknames_proportion(data_dir: Path) -> float:
    # This function calculates the proportion of names that have potential nicknames. This will return a float that
    # that will be used as a constant to scale the nicknames noise function in the pseudopeople packages.

    data_files = []
    for file in data_dir.glob("*.parquet"):
        data_files.append(file)
    names = []
    for file in data_files:
        df = pd.read_parquet(file)
        df = df.loc[df["year"] == 2030, "first_name"]
        names.append(df)
    names = pd.concat(names)

    # Load and format nicknames dataset
    names_with_nicknames = pd.read_csv(paths.NICKNAMES_DATA_PATH)
    names_with_nicknames = names_with_nicknames["name"].apply(lambda x: x.title())

    # Proportion
    nickname_eligible = names.isin(names_with_nicknames)
    proportion_of_nicknames = nickname_eligible.sum() / len(names)
    return proportion_of_nicknames
