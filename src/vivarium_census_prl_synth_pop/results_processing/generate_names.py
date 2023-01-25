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
    "random_seed",
]


def load_data(raw_results_dir: Path) -> pd.DataFrame:
    obs_data = []

    observer_directories = [
        obs_dir for obs_dir in raw_results_dir.glob("*") if obs_dir.is_dir()
    ]
    for obs_dir in observer_directories:
        logger.info(f"Processing data for {obs_dir.name}...")
        obs_files = obs_dir.glob("*.csv.bz2")
        for file in obs_files:
            df = pd.read_csv(file)
            df["random_seed"] = file.name.split(".")[0].split("_")[-1]
            obs_data.append(df)

    data = pd.concat(obs_data)

    # Drop unnecessary columns (Social Security Observer does not have sex  and race ethnicity)
    data = data[COLUMNS_FOR_NAME_PROCESSING].drop_duplicates()

    return data


def format_data_for_name_generation(
    data: pd.DataFrame,
) -> Tuple:

    data["first_name_id"] = (
        data["random_seed"].astype(str) + "_" + data["first_name_id"].astype(str)
    )
    data["middle_name_id"] = (
        data["random_seed"].astype(str) + "_" + data["middle_name_id"].astype(str)
    )
    data["last_name_id"] = (
        data["random_seed"].astype(str) + "_" + data["last_name_id"].astype(str)
    )
    # Get year of birth - string column and not timestamp
    data["year_of_birth"] = pd.to_datetime(data["date_of_birth"]).dt.year

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
    first_name_data, middle_name_data = format_data_for_name_generation(pop)
