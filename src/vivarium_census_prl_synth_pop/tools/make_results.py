import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from loguru import logger
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import paths
from vivarium_census_prl_synth_pop.results_processing import formatter
from vivarium_census_prl_synth_pop.results_processing.addresses import (
    get_address_id_maps,
)
from vivarium_census_prl_synth_pop.results_processing.names import (
    get_employer_name_map,
    get_given_name_map,
    get_last_name_map,
    get_middle_initial_map,
)
from vivarium_census_prl_synth_pop.results_processing.ssn_and_itin import (
    get_simulant_id_maps,
)

FINAL_OBSERVERS = [
    "decennial_census_observer",
    "household_survey_observer_acs",
    "household_survey_observer_cps",
    "wic_observer",
    "social_security_observer",
    "tax_w2_observer",
]
COLUMNS_TO_NOT_MAP = {
    "decennial_census_observer": [],
    "household_survey_observer_acs": [],
    "household_survey_observer_cps": [],
    "wic_observer": [],
    "social_security_observer": ["sex", "race_ethnicity"],
    "tax_w2_observer": ["race_ethnicity"],
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
    processed_results = load_data(raw_output_dir)
    # Generate all post-processing maps to apply to raw results
    artifact = Artifact(artifact_path)
    maps = generate_maps(processed_results, artifact)

    # Iterate through expected forms and generate them. Generate columns each of these forms need to have.
    for observer in FINAL_OBSERVERS:
        logger.info(f"Processing data for {observer}...")
        # Fixme: This code assumes a 1 to 1 relationship of raw to final observers
        obs_data = processed_results[observer]

        obs_data = obs_data.drop(columns=COLUMNS_TO_NOT_MAP[observer])
        for column in obs_data.columns:
            if column in maps:
                for target_column_name, column_map in maps[column].items():
                    obs_data[target_column_name] = obs_data[column].map(column_map)

        logger.info(f"Writing final results for {observer}.")
        # fixme: Process columns for final outputs - columns being mapped are no longer dropped
        obs_data.to_csv(
            final_output_dir / f"{observer}.csv.bz2",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )


def load_data(raw_results_dir: Path) -> Dict[str, pd.DataFrame]:
    # Loads in all observer outputs and stores them in a dict with observer name as keys.
    observers_results = {}

    observer_directories = [
        obs_dir for obs_dir in raw_results_dir.glob("*") if obs_dir.is_dir()
    ]
    for obs_dir in observer_directories:
        logger.info(f"Loading data for {obs_dir.name}...")
        obs_files = obs_dir.glob("*.csv.bz2")
        obs_data = []
        for file in obs_files:
            df = pd.read_csv(file)
            df["random_seed"] = file.name.split(".")[0].split("_")[-1]
            # Process columns to map for observers
            for output_column, (
                column_formatter,
                required_columns,
            ) in formatter.COLUMN_FORMATTERS.items():
                if set(required_columns).issubset(set(df.columns)):
                    df[output_column] = column_formatter(df[required_columns])
            obs_data.append(df)

        observers_results[obs_dir.name] = pd.concat(obs_data)

    return observers_results


def generate_maps(
    obs_data: Dict[str, pd.DataFrame], artifact: Artifact
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Parameters:
        obs_data: Dictionary of raw observer outputs with key being the observer name and the value being a dataframe.
        artifact: Artifact that contains data needed to generate values.

    Returns:
        maps: Dictionary with key being string of the name of the column to be mapped.  Example first_name_id or
          address_id.  Values for each key will be a dictionary named with the column to be mapped to as the key with a
          corresponding series containing the mapped values.
    """

    # Create RandomnessStream for post-processing
    key = "post_processing_maps"
    clock = lambda: pd.Timestamp("2020-04-01")
    seed = 0
    randomness = RandomnessStream(key=key, clock=clock, seed=seed)

    # Add column maps to mapper here
    # The key should be the index of the map and the mapping function the value
    mappers = {
        "first_name_id": get_given_name_map,
        "middle_name_id": get_middle_initial_map,
        "last_name_id": get_last_name_map,
        "address_id": get_address_id_maps,
        "employer_id": get_employer_name_map,
        # "simulant_id": get_simulant_id_maps,
        # fixme: MIC-3841 issue with W2 observer
    }
    maps = {
        column: mapper(column, obs_data, artifact, randomness)
        for column, mapper in mappers.items()
    }

    return maps


def build_final_results_directory(results_dir: str) -> Tuple[Path, Path]:
    final_output_dir = utilities.build_output_dir(
        Path(results_dir),
        subdir=paths.FINAL_RESULTS_DIR_NAME / datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    raw_output_dir = Path(results_dir) / paths.RAW_RESULTS_DIR_NAME

    return raw_output_dir, final_output_dir
