import shutil
from itertools import chain
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from loguru import logger
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.constants import data_keys, metadata, paths
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
from vivarium_census_prl_synth_pop.utilities import (
    build_output_dir,
    get_all_simulation_seeds,
    write_to_disk, sanitize_location,
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
        "year",
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
        "survey_date",
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
        "survey_date",
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
        "year",
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
        "tax_form",
        "tax_year",
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
        "tax_year",
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
        "tax_year",
    },
}

PUBLIC_SAMPLE_PUMA_PROPORTION = 0.5

PUBLIC_SAMPLE_ADDRESS_PARTS = {
    "city": "Anytown",
    "state": "US",
    "zipcode": "00000",
}


def build_results(
    raw_output_dir: Union[str, Path],
    final_output_dir: Union[str, Path],
    mark_best: bool,
    test_run: bool,
    extension: str,
    public_sample: bool,
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
    perform_post_processing(
        raw_output_dir, final_output_dir, extension, seed, artifact_path, public_sample
    )

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
    raw_output_dir: Path,
    final_output_dir: Path,
    extension: str,
    seed: str,
    artifact_path: Path,
    public_sample: bool,
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
    all_seeds = get_all_simulation_seeds(raw_output_dir)
    maps = generate_maps(processed_results, artifact, randomness, seed, all_seeds)

    if public_sample:
        pumas_to_keep = (
            artifact.load(data_keys.POPULATION.HOUSEHOLDS)
            .reset_index()[["state", "puma"]]
            # Note: The value in the ACS data is the FIPS code, not the abbreviation
            .rename(columns={"state": "state_id"})
            .drop_duplicates()
            .sample(frac=PUBLIC_SAMPLE_PUMA_PROPORTION, random_state=0)
        )

        # For SSA data, we keep the simulants ever observed in that geographic area.
        simulants_to_keep = []
        for observer in FINAL_OBSERVERS:
            obs_data = processed_results[observer]
            if {"state_id", "puma"} <= set(obs_data.columns):
                obs_data = obs_data.merge(pumas_to_keep, on=["state_id", "puma"], how="inner")
                simulants_to_keep.append(obs_data["simulant_id"])

        simulants_to_keep = pd.concat(simulants_to_keep, ignore_index=True).unique()

    # Iterate through expected forms and generate them. Generate columns each of these forms need to have.
    for observer in FINAL_OBSERVERS:
        logger.info(f"Processing data for {observer}.")
        obs_data = processed_results[observer]

        if public_sample:
            if {"state_id", "puma"} <= set(obs_data.columns):
                obs_data = obs_data.merge(pumas_to_keep, on=["state_id", "puma"], how="inner")
            else:
                obs_data = obs_data[obs_data["simulant_id"].isin(simulants_to_keep)].copy()
                logger.warning(
                    f"Cannot geographic subset {observer}; using simulants ever in geographic subset."
                )

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

        if public_sample:
            for address_prefix in ["", "mailing_address_", "employer_"]:
                for address_part, address_part_value in PUBLIC_SAMPLE_ADDRESS_PARTS.items():
                    if f"{address_prefix}{address_part}" in obs_data.columns:
                        obs_data[f"{address_prefix}{address_part}"] = address_part_value

        logger.info(f"Writing final results for {observer}.")
        obs_dir = build_output_dir(final_output_dir, subdir=observer)
        seed_ext = f"_{seed}" if seed != "" else ""
        write_to_disk(obs_data.copy(), obs_dir / f"{observer}{seed_ext}.{extension}")


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
                df = read_datafile(file)
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
            obs_data = read_datafile(obs_file)
            obs_data["random_seed"] = seed
            obs_data = formatter.format_columns(obs_data)

        observers_results[obs_dir.name] = obs_data

    return observers_results


def read_datafile(file: Path) -> pd.DataFrame:
    if ".hdf" == file.suffix:
        df = pd.read_hdf(file).reset_index()
    elif ".parquet" == file.suffix:
        df = pd.read_parquet(file).reset_index()
    else:
        raise ValueError(
            f"Supported extensions are {metadata.SUPPORTED_EXTENSIONS}. "
            f"{file.suffix[1:]} was provided."
        )
    return df


def generate_maps(
    obs_data: Dict[str, pd.DataFrame],
    artifact: Artifact,
    randomness: RandomnessStream,
    seed: str,
    all_seeds: List[str],
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Parameters:
        obs_data: Dictionary of raw observer outputs with key being the observer name and the value being a dataframe.
        artifact: Artifact that contains data needed to generate values.
        randomness: RandomnessStream to use in creating maps.
        seed: vivarium random seed this is being run for
        all_seeds: list of all vivarium random seeds found in raw results folder

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
        column: mapper(column, obs_data, artifact, randomness, seed, all_seeds)
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


def subset_results_by_state(processed_results_dir: str, state: str) -> None:
    # Loads final results and subsets those files to a provide state excluding the Social Security Observer

    from vivarium_census_prl_synth_pop.constants.metadata import US_STATE_ABBRV_MAP

    abbrev_name_dict = {v: k for k, v in US_STATE_ABBRV_MAP.items()}
    state_name = sanitize_location(abbrev_name_dict[state.upper()])
    processed_results_dir = Path(processed_results_dir)
    all_results = processed_results_dir / "usa"
    state_dir = processed_results_dir / "states" / state_name
    state_dir.mkdir(exist_ok=True, parents=True)
    # copy files from final results to state directory for further processing
    shutil.copytree(all_results, state_dir, dirs_exist_ok=True)

    for observer in FINAL_OBSERVERS:
        if observer == "social_security_observer":
            continue
        logger.info(f"Processing data for {observer}...")
        obs_dir = state_dir / observer
        obs_files = sorted(
            list(chain(*[obs_dir.glob(f"*.{ext}") for ext in SUPPORTED_EXTENSIONS]))
        )
        for file in obs_files:
            df = read_datafile(file)
            df["random_seed"] = file.name.split(".")[0].split("_")[-1]
            df = formatter.format_columns(df)
            obs_data = df.loc[df["state"] == state]
            write_to_disk(obs_data.copy(), file)
        logger.info(f"Finished writing data for {state_name} subset of {observer} files.")

    return None
