import glob
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger

from vivarium_census_prl_synth_pop.components.synthetic_pii import NameGenerator

COLUMNS_FOR_NAME_PROCESSING = [
    "first_name_id",
    "middle_name_id",
    "last_name_id",
    "date_of_birth",
    "sex",
    "race_ethnicity",
]


def load_data(raw_results_dir: Path):
    data = pd.DataFrame()

    observer_directories = [f.path for f in os.scandir(raw_results_dir) if f.is_dir()]
    for obs_dir in observer_directories:
        logger.info(f'Processing data for {obs_dir.split("/")[-1]}...')
        results = glob.glob(os.path.join(obs_dir, "*.csv.bz2"))
        for file in results:
            tmp = pd.read_csv(file)
            data = pd.concat([data, tmp])

    # Drop unnecessary columns (Social Security Observer does not have sex  and race ethnicity)
    data = data[COLUMNS_FOR_NAME_PROCESSING].drop_duplicates()

    return data


def fromat_data_for_name_generation(
    data: pd.DataFrame,
) -> Tuple:
    # todo: update to handle with seed column once we are doing parallel runs
    seed = 0

    # Clean data
    data["first_name_id"] = f"{seed}_" + data["first_name_id"].astype(str)
    data["middle_name_id"] = f"{seed}_" + data["middle_name_id"].astype(str)
    data["last_name_id"] = f"{seed}_" + data["last_name_id"].astype(str)
    # Get year of birth - string column and not timestamp
    data["year_of_birth"] = data["date_of_birth"].str[:4]

    first_name_data = data[["first_name_id", "year_of_birth", "sex"]].set_index(
        "first_name_id"
    )
    middle_name_data = data[["middle_name_id", "year_of_birth", "sex"]].set_index(
        "middle_name_id"
    )
    return first_name_data, middle_name_data


def generate_names(raw_results_dir: Path, final_output_dir: Path) -> None:
    # This will update the 3 name_id columns in the observer outputs and to be name columns in the processed outputs
    pop = load_data(raw_results_dir)
    first_name_data, middle_name_data = fromat_data_for_name_generation(pop)
