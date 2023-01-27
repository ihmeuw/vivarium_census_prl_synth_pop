from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from loguru import logger
from vivarium import Artifact

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import paths
from vivarium_census_prl_synth_pop.results_processing.names import (
    get_first_name_map,
    get_middle_name_map,
)

FINAL_OBSERVERS = [
    "decennial_census",
    "household_survey_acs",
    "household_survey_cps",
    "wic",
    "social_security",
    "tax_w2_1099",
]
COLUMNS_TO_NOT_MAP = {
    "decennial_census": [],
    "household_survey_acs": [],
    "household_survey_cps": [],
    "wic": [],
    "social_security": ["sex", "race_ethnicity"],
    "tax_w2_1099": ["race_ethnicity"],
}


def build_results(
    results_dir: str, mark_best: bool, test_run: bool, artifact_path: str
) -> None:
    if mark_best and test_run:
        logger.error(
            "A test run can't be marked best. "
            "Please remove either the mark best or the test run flag."
        )
        return
    logger.info("Creating final results directory.")
    raw_output_dir, final_output_dir = build_final_results_directory(results_dir)

    artifact_path = Path(artifact_path)
    logger.info("Performing post-processing")
    perform_post_processing(raw_output_dir, final_output_dir, artifact_path)

    if test_run:
        logger.info("Test run - not marking results as latest.")
    else:
        create_results_link(final_output_dir, paths.LATEST_DIR_NAME)

    if mark_best:
        create_results_link(final_output_dir, paths.BEST_DIR_NAME)


def create_results_link(output_dir: Path, link_name: Path) -> None:
    logger.info(f"Marking results as {link_name}.")
    output_root_dir = output_dir.parent
    link_dir = output_root_dir / link_name
    link_dir.unlink(missing_ok=True)
    link_dir.symlink_to(output_dir, target_is_directory=True)


def perform_post_processing(
    raw_output_dir: Path, final_output_dir: Path, artifact_path: Path
) -> None:
    raw_results = load_data(raw_output_dir)
    # Generate all post-processing maps to apply to raw results
    artifact = Artifact(artifact_path)
    maps = generate_maps(raw_results, artifact)

    # Iterate through expected forms and generate them. Generate columns each of these forms need to have.
    for observer in FINAL_OBSERVERS:
        logger.info(f"Processing data for {observer}...")
        # Fixme: This code assumes a 1 to 1 relationship of raw to final observers
        obs_data = raw_results[observer]

        obs_data = obs_data.drop(columns=COLUMNS_TO_NOT_MAP[observer])
        for column in obs_data.columns:
            if column in maps:
                for target_column_name, column_map in maps.items():
                    obs_data[target_column_name] = obs_data.map(column_map)
                # This assumes the column we are mapping will be dropped
                obs_data.drop(columns=column)

        logger.info(f"Writing final results for {observer}.")
        obs_data.to_csv(final_output_dir / f"{observer}.csv.bz2")


def load_data(raw_results_dir: Path) -> Dict[str, pd.DataFrame]:
    # Loads in all observer outputs and stores them in a dict with observer name as keys.
    observers_results = {}

    observer_directories = [
        obs_dir for obs_dir in raw_results_dir.glob("*") if obs_dir.is_dir()
    ]
    for obs_dir in observer_directories:
        logger.info(f"Processing data for {obs_dir.name}...")
        obs_files = obs_dir.glob("*.csv.bz2")
        obs_data = []
        for file in obs_files:
            df = pd.read_csv(file)
            df["random_seed"] = file.name.split(".")[0].split("_")[-1]
            obs_data.append(df)

        observers_results[obs_dir.name] = pd.concat(obs_data)

    return observers_results


def generate_maps(
    raw_data: Dict[str, pd.DataFrame], artifact: Artifact
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Parameters:
        raw_data: Dictionary of raw observer outputs with key being the observer name and the value being a dataframe.
        artifact: Artifact that contains data needed to generate values.

    Returns:
        maps: Dictionary with key being string of the name of the column to be mapped.  Example first_name_id or
          address_id.  Values for each key will be a dictionary named with the column to be mapped to as the key with a
          corresponding series containing the mapped values.
    """
    first_name_map = get_first_name_map(raw_data, artifact)
    middle_name_map = get_middle_name_map(raw_data, artifact)

    return {}


def build_final_results_directory(results_dir: str) -> Tuple[Path, Path]:
    final_output_root_dir = utilities.build_output_dir(
        Path(results_dir), subdir=paths.FINAL_RESULTS_DIR_NAME
    )
    raw_output_dir = Path(results_dir) / paths.RAW_RESULTS_DIR_NAME
    final_output_dir = final_output_root_dir / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    return raw_output_dir, final_output_dir
