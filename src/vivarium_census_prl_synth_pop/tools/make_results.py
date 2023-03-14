import csv
from itertools import chain
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from loguru import logger
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import paths
from vivarium_census_prl_synth_pop.constants.metadata import SUPPORTED_EXTENSIONS
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
    do_collide_ssns,
    get_simulant_id_maps,
)

FINAL_OBSERVERS = {
    "decennial_census_observer": {
        "simulant_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "date_of_birth",
        "street_number",
        "street_name",
        "unit_number",
        "city",
        "zipcode",
        "state",
        "relation_to_household_head",
        "sex",
        "race_ethnicity",
        "guardian_1",
        "guardian_2",
        # todo: Update with additional address columns in MIC-3846
        # "guardian_1_addrress_id",
        # "guardian_2_address_id",
        "housing_type",
    },
    "household_survey_observer_acs": {
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "date_of_birth",
        "sex",
        "street_number",
        "street_name",
        "unit_number",
        "city",
        "zipcode",
        "state",
        "guardian_1",
        "guardian_2",
        # todo: Update with additional address columns in MIC-3846
        # "guardian_1_addrress_id",
        # "guardian_2_address_id",
        "housing_type",
    },
    "household_survey_observer_cps": {
        "household_id",
        "simulant_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "date_of_birth",
        "sex",
        "street_number",
        "street_name",
        "unit_number",
        "city",
        "zipcode",
        "state",
        "guardian_1",
        "guardian_2",
        # todo: Update with additional address columns in MIC-3846
        # "guardian_1_addrress_id",
        # "guardian_2_address_id",
        "housing_type",
    },
    "wic_observer": {
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "date_of_birth",
        "street_number",
        "street_name",
        "unit_number",
        "city",
        "zipcode",
        "state",
        "relation_to_household_head",
        "sex",
        "race_ethnicity",
        "guardian_1",
        "guardian_2",
        # todo: Update with additional address columns in MIC-3846
        # "guardian_1_addrress_id",
        # "guardian_2_address_id",
        "housing_type",
    },
    "social_security_observer": {
        "simulant_id",
        "first_name",
        "middle_initial",
        "last_name",
        "date_of_birth",
        "ssn",
        "event_type",
        "event_date",
    },
    "tax_w2_observer": {
        "simulant_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "date_of_birth",
        "mailing_address_street_number",
        "mailing_address_street_name",
        "mailing_address_unit_number",
        "mailing_address_city",
        "mailing_address_zipcode",
        "mailing_address_state",
        "mailing_address_po_box",
        "income",
        "employer_id",
        "employer_name",
        "employer_street_number",
        "employer_street_name",
        "employer_unit_number",
        "employer_city",
        "employer_zipcode",
        "employer_state",
        "ssn",
        "is_w2",
    },
    "tax_1040_observer": {
        "simulant_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "date_of_birth",
        "mailing_address_street_number",
        "mailing_address_street_name",
        "mailing_address_unit_number",
        "mailing_address_city",
        "mailing_address_zipcode",
        "mailing_address_state",
        "mailing_address_po_box",
        "housing_type",
        "joint_filer",
        "ssn",
        "itin",
    },
    "tax_dependents_observer": {
        # Metadata is for a dependent.  This should capture each dependent/guardian pair.  Meaning that if a dependent
        # has 2 guardians, there should be a duplicate row but the guardian_id column would contain the 2 simulant_ids
        # for that dependent's guardians.
        "simulant_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "date_of_birth",
        "mailing_address_street_number",
        "mailing_address_street_name",
        "mailing_address_unit_number",
        "mailing_address_city",
        "mailing_address_zipcode",
        "mailing_address_state",
        "mailing_address_po_box",
        "sex",
        "ssn",
        "tax_year",
        "guardian_id",
        "housing_type",
    },
}


def build_results(
    raw_output_dir: Union[str, Path],
    final_output_dir: Union[str, Path],
    mark_best: bool,
    test_run: bool,
    seed: str,
    artifact_path: Union[str, Path],
):
    if mark_best and test_run:
        logger.error(
            "A test run can't be marked best. "
            "Please remove either the mark best or the test run flag."
        )
        return
    raw_output_dir = Path(raw_output_dir)
    final_output_dir = Path(final_output_dir)
    artifact_path = Path(artifact_path)
    logger.info("Performing post-processing")
    perform_post_processing(raw_output_dir, final_output_dir, seed, artifact_path)

    if test_run:
        logger.info("Test run - not marking results as latest.")
    else:
        create_results_link(final_output_dir, paths.LATEST_DIR_NAME)

    if mark_best:
        create_results_link(final_output_dir, paths.BEST_DIR_NAME)


def create_results_link(output_dir: Path, link_name: Path) -> None:
    logger.info(f"Marking results as {link_name}: {str(output_dir)}.")
    output_root_dir = output_dir.parent
    link_dir = output_root_dir / link_name
    link_dir.unlink(missing_ok=True)
    link_dir.symlink_to(output_dir, target_is_directory=True)


def perform_post_processing(
    raw_output_dir: Path, final_output_dir: Path, seed: str, artifact_path: Path
) -> None:
    # Create RandomnessStream for post-processing
    randomness = RandomnessStream(
        key="post_processing_maps",
        clock=lambda: pd.Timestamp("2020-04-01"),
        seed=0,
    )

    processed_results = load_data(raw_output_dir, seed)
    # Generate all post-processing maps to apply to raw results
    artifact = Artifact(artifact_path)
    maps = generate_maps(processed_results, artifact, randomness, seed)

    # Iterate through expected forms and generate them. Generate columns each of these forms need to have.
    for observer in FINAL_OBSERVERS:
        logger.info(f"Processing data for {observer}.")
        obs_data = processed_results[observer]

        for column in obs_data.columns:
            if column not in maps:
                continue
            for target_column_name, column_map in maps[column].items():
                if target_column_name not in FINAL_OBSERVERS[observer]:
                    continue
                obs_data[target_column_name] = obs_data[column].map(column_map)

        if observer == "tax_w2_observer":
            # For w2, we need to post-process to allow for SSN collisions in the data in cases where
            #  the simulant has no SSN but is employed (they'd need to have supplied an SSN to their employer)
            obs_data = do_collide_ssns(obs_data, maps["simulant_id"]["ssn"], randomness)

        obs_data = obs_data[list(FINAL_OBSERVERS[observer])]
        logger.info(f"Writing final results for {observer}.")
        obs_dir = final_output_dir / observer
        obs_dir.mkdir(parents=True, exist_ok=True)
        seed_ext = f"_{seed}" if seed != "" else ""
        obs_data.to_csv(
            obs_dir / f"{observer}{seed_ext}.csv.bz2",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )


def load_data(raw_results_dir: Path, seed: str) -> Dict[str, pd.DataFrame]:
    # Loads in all observer outputs and stores them in a dict with observer name as keys.
    observers_results = {}
    for observer in FINAL_OBSERVERS:
        obs_dir = raw_results_dir / observer
        if seed == "":
            logger.info(f"Loading data for {obs_dir.name}")
            obs_files = sorted(
                list(chain(*[obs_dir.glob(f"*.{ext}") for ext in SUPPORTED_EXTENSIONS]))
            )
            obs_data = []
            for file in obs_files:
                if ".hdf" == file.suffix:
                    df = pd.read_hdf(file).reset_index()
                else:
                    df = pd.read_csv(file)
                df["random_seed"] = file.name.split(".")[0].split("_")[-1]
                df = formatter.format_columns(df)
                obs_data.append(df)
            obs_data = pd.concat(obs_data)
        else:
            obs_files = list(
                chain(*[obs_dir.glob(f"*_{seed}.{ext}") for ext in SUPPORTED_EXTENSIONS])
            )
            if len(obs_files) > 1:
                raise FileExistsError(
                    f"Too many files found with the given seed {seed} for observer {observer}"
                    f" - {obs_files}."
                )
            elif len(obs_files) == 0:
                raise FileNotFoundError(
                    f"No file found with the seed {seed} for observer {observer}."
                )
            obs_file = obs_files[0]
            logger.info(f"Loading data for {obs_file.name}")
            if ".hdf" == obs_file.suffix:
                obs_data = pd.read_hdf(obs_file).reset_index()
            else:
                obs_data = pd.read_csv(obs_file)
            obs_data["random_seed"] = seed
            obs_data = formatter.format_columns(obs_data)

        observers_results[obs_dir.name] = obs_data

    return observers_results


def generate_maps(
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
    seed: str,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Parameters:
        obs_data: Dictionary of raw observer outputs with key being the observer name and the value being a dataframe.
        artifact: Artifact that contains data needed to generate values.
        randomness: RandomnessStream to use in creating maps.
        seed: vivarium random seed this is being run for

    Returns:
        maps: Dictionary with key being string of the name of the column to be mapped.  Example first_name_id or
          address_id.  Values for each key will be a dictionary named with the column to be mapped to as the key with a
          corresponding series containing the mapped values.
    """

    # Add column maps to mapper here
    # The key should be the index of the map and the mapping function the value
    mappers = {
        "first_name_id": get_given_name_map,
        "middle_name_id": get_middle_initial_map,
        "last_name_id": get_last_name_map,
        "address_id": get_address_id_maps,
        "employer_id": get_employer_name_map,
        "simulant_id": get_simulant_id_maps,
        "employer_address_id": get_address_id_maps,
    }
    maps = {
        column: mapper(column, obs_data, artifact, randomness, seed)
        for column, mapper in mappers.items()
    }
    # todo: Include with MIC-3846
    # Add guardian address details and change dict keys for address_ids map to be clear.
    # maps["guardian_1_address_id"] = {
    #   outer_k: {"uardian_1_"+ inner_k: inner_v for inner_k, inner_v in outer_v.items()}
    #   for outer_k, outer_v in maps["address_id.items()
    #   }
    # maps["guardian_2_address_id"] = {
    #   outer_k: {"uardian_2_"+ inner_k: inner_v for inner_k, inner_v in outer_v.items()}
    #   for outer_k, outer_v in maps["address_id.items()
    #   }

    return maps
