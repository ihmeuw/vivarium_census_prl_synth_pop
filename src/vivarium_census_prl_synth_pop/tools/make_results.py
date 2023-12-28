from itertools import chain
from pathlib import Path
from shutil import copyfile
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from pseudopeople.constants.metadata import COPY_HOUSEHOLD_MEMBER_COLS
from pseudopeople.schema_entities import COLUMNS
from vivarium import Artifact
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap

from vivarium_census_prl_synth_pop.constants import data_keys, metadata
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
    copy_ssn_from_household_member,
    do_collide_ssns,
    get_simulant_id_maps,
)
from vivarium_census_prl_synth_pop.utilities import (
    build_output_dir,
    calculate_guardian_duplication_metadata_proportions,
    get_all_simulation_seeds,
    load_nicknames_data,
    merge_dependents_and_guardians,
    sanitize_location,
    write_to_disk,
)

FINAL_OBSERVERS = {
    metadata.DatasetNames.CENSUS: {
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "copy_age",
        "date_of_birth",
        "copy_date_of_birth",
        "street_number",
        "street_name",
        "unit_number",
        "city",
        "zipcode",
        "state",
        "relationship_to_reference_person",
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
    metadata.DatasetNames.ACS: {
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "copy_age",
        "date_of_birth",
        "copy_date_of_birth",
        "relationship_to_reference_person",
        "sex",
        "race_ethnicity",
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
    metadata.DatasetNames.CPS: {
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "copy_age",
        "date_of_birth",
        "copy_date_of_birth",
        "sex",
        "race_ethnicity",
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
    metadata.DatasetNames.WIC: {
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "date_of_birth",
        "copy_date_of_birth",
        "street_number",
        "street_name",
        "unit_number",
        "city",
        "zipcode",
        "state",
        "relationship_to_reference_person",
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
    metadata.DatasetNames.SSA: {
        "simulant_id",
        "first_name",
        "middle_name",
        "last_name",
        "sex",
        "date_of_birth",
        "copy_date_of_birth",
        "ssn",
        "copy_ssn",
        "event_type",
        "event_date",
    },
    metadata.DatasetNames.TAXES_W2_1099: {
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "copy_age",
        "date_of_birth",
        "copy_date_of_birth",
        "mailing_address_street_number",
        "mailing_address_street_name",
        "mailing_address_unit_number",
        "mailing_address_city",
        "mailing_address_zipcode",
        "mailing_address_state",
        "mailing_address_po_box",
        "wages",
        "employer_id",
        "employer_name",
        "employer_street_number",
        "employer_street_name",
        "employer_unit_number",
        "employer_city",
        "employer_zipcode",
        "employer_state",
        "ssn",
        "copy_ssn",
        "tax_form",
        "tax_year",
    },
    metadata.DatasetNames.TAXES_1040: {
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "copy_age",
        "date_of_birth",
        "copy_date_of_birth",
        "mailing_address_street_number",
        "mailing_address_street_name",
        "mailing_address_unit_number",
        "mailing_address_city",
        "mailing_address_zipcode",
        "mailing_address_state",
        "mailing_address_po_box",
        "housing_type",
        "joint_filer",
        "ssn_itin",
        "copy_ssn",
        "tax_year",
        "relationship_to_reference_person",
    },
    metadata.DatasetNames.TAXES_DEPENDENTS: {
        # Metadata is for a dependent.  This should capture each dependent/guardian pair.  Meaning that if a dependent
        # has 2 guardians, there should be a duplicate row but the guardian_id column would contain the 2 simulant_ids
        # for that dependent's guardians.
        "simulant_id",
        "household_id",
        "first_name",
        "middle_initial",
        "last_name",
        "age",
        "copy_age",
        "date_of_birth",
        "copy_date_of_birth",
        "mailing_address_street_number",
        "mailing_address_street_name",
        "mailing_address_unit_number",
        "mailing_address_city",
        "mailing_address_zipcode",
        "mailing_address_state",
        "mailing_address_po_box",
        "sex",
        "ssn_itin",
        "copy_ssn",
        "guardian_id",
        "housing_type",
        "tax_year",
    },
}

OUTPUT_DATASETS = [
    metadata.DatasetNames.ACS,
    metadata.DatasetNames.CENSUS,
    metadata.DatasetNames.CPS,
    metadata.DatasetNames.SSA,
    metadata.DatasetNames.WIC,
    metadata.DatasetNames.TAXES_W2_1099,
    metadata.DatasetNames.TAXES_1040,
]

PUBLIC_SAMPLE_PUMA_PROPORTION = 0.5

PUBLIC_SAMPLE_ADDRESS_PARTS = {
    "city": "Anytown",
    "state": "WA",
    "zipcode": "00000",
}


def build_results(
    raw_output_dir: Path,
    final_output_dir: Path,
    extension: str,
    public_sample: bool,
    seed: str,
    artifact_path: str,
) -> None:
    logger.info("Performing post-processing")
    perform_post_processing(
        raw_output_dir, final_output_dir, extension, seed, artifact_path, public_sample
    )


def perform_post_processing(
    raw_output_dir: Path,
    final_output_dir: Path,
    extension: str,
    seed: str,
    artifact_path: str,
    public_sample: bool,
) -> None:
    # Create RandomnessStream for post-processing
    # NOTE: We use an IndexMap size of 15 million because that is a bit more than
    # the length of all business_ids in the simulation which is expected to be
    # the largest set of things to be mapped.
    randomness = RandomnessStream(
        key="post_processing_maps",
        clock=lambda: pd.Timestamp("2020-04-01"),
        seed=0,
        index_map=IndexMap(size=15_000_000),
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

        # In order to have all housing types in the sample data, we additionally include
        # the (up to 6) PUMAs with group quarters in them.
        gq_pumas = (
            processed_results[metadata.DatasetNames.CENSUS]
            .pipe(lambda df: df[df["housing_type"] != "Household"])[["state_id", "puma"]]
            .drop_duplicates()
        )
        pumas_to_keep = pd.concat(
            [pumas_to_keep, gq_pumas], ignore_index=True
        ).drop_duplicates()

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
        logger.info(f"Processing {observer} data.")
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

        if observer == metadata.DatasetNames.TAXES_W2_1099:
            # For w2, we need to post-process to allow for SSN collisions in the data in cases where
            #  the simulant has no SSN but is employed (they'd need to have supplied an SSN to their employer)
            obs_data = do_collide_ssns(obs_data, maps["simulant_id"]["ssn"], randomness)
        if observer in [
            metadata.DatasetNames.SSA,
            metadata.DatasetNames.TAXES_W2_1099,
            metadata.DatasetNames.TAXES_DEPENDENTS,
            metadata.DatasetNames.TAXES_1040,
        ]:
            # Copy SSN from household members
            obs_data["copy_ssn"] = copy_ssn_from_household_member(
                obs_data["copy_ssn"], maps["simulant_id"]["ssn"]
            )

        obs_data = obs_data[list(FINAL_OBSERVERS[observer])]

        if public_sample:
            for address_prefix in ["", "mailing_address_", "employer_"]:
                for address_part, address_part_value in PUBLIC_SAMPLE_ADDRESS_PARTS.items():
                    if f"{address_prefix}{address_part}" in obs_data.columns:
                        obs_data[f"{address_prefix}{address_part}"] = address_part_value
            # Fix the state column dtypes
            state_categories = sorted(list(metadata.US_STATE_ABBRV_MAP.values()))
            state_cols = [c for c in obs_data.columns if "state" in c]
            for col in state_cols:
                obs_data[col] = obs_data[col].astype(
                    pd.CategoricalDtype(categories=state_categories)
                )
            processed_results[observer] = obs_data
    # Format 1040 by combining with tax dependents to match guardians with dependents
    processed_results[metadata.DatasetNames.TAXES_1040] = formatter.format_1040_dataset(
        processed_results
    )

    # Write results for each dataset - we do not need to write out tax dependents now that we format
    # the 1040 dataset above
    for observer in FINAL_OBSERVERS:
        if observer in OUTPUT_DATASETS:
            obs_data = processed_results[observer]
            logger.info(f"Writing final {observer} results.")
            obs_dir = build_output_dir(final_output_dir, subdir=observer)
            seed_ext = f"_{seed}" if seed != "" else ""
            write_shard_metadata(observer, obs_data, obs_dir, seed_ext)
            write_to_disk(obs_data.copy(), obs_dir / f"{observer}{seed_ext}.{extension}")


def load_data(raw_results_dir: Path, seed: str) -> Dict[str, pd.DataFrame]:
    # Loads in all observer outputs and stores them in a dict with observer name as keys.
    observers_results = {}
    for observer in FINAL_OBSERVERS:
        obs_dir = raw_results_dir / observer
        if seed == "":
            logger.info(f"Loading {obs_dir.name} data ")
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


def read_datafile(file: Path, reset_index: bool = True, state: str = None) -> pd.DataFrame:
    state_column = "mailing_address_state" if "tax" in file.parent.name else "state"
    if ".hdf" == file.suffix:
        state_filter = [f"{state_column} == {state}."] if state else None
        with pd.HDFStore(str(file), mode="r") as hdf_store:
            df = hdf_store.select("data", where=state_filter)
    elif ".parquet" == file.suffix:
        state_filter = [(state_column, "==", state)] if state else None
        df = pq.read_table(file, filters=state_filter).to_pandas()
    else:
        raise ValueError(
            f"Supported extensions are {metadata.SUPPORTED_EXTENSIONS}. "
            f"{file.suffix[1:]} was provided."
        )
    if reset_index:
        df = df.reset_index()
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

    abbrev_name_dict = {v: k for k, v in metadata.US_STATE_ABBRV_MAP.items()}
    state_name = sanitize_location(abbrev_name_dict[state.upper()])
    processed_results_dir = Path(processed_results_dir)
    directories = [x for x in processed_results_dir.iterdir() if x.is_dir()]
    # Get USA results directory
    for directory in directories:
        if "pseudopeople_input_data_usa" in directory.name:
            usa_results_dir = directory
    # Parse version number if necessary
    version = usa_results_dir.name.split("_")[-1]
    if version != "usa":
        state_dir = (
            processed_results_dir
            / "states"
            / f"pseudopeople_input_data_{state_name}_{version}"
        )
    else:
        state_dir = processed_results_dir / "states" / f"pseudopeople_input_data_{state_name}"

    for observer in FINAL_OBSERVERS:
        logger.info(f"Processing {observer} data")
        usa_obs_dir = usa_results_dir / observer
        usa_obs_files = sorted(
            list(chain(*[usa_obs_dir.glob(f"*.{ext}") for ext in SUPPORTED_EXTENSIONS]))
        )
        state_observer_dir = state_dir / observer
        build_output_dir(state_observer_dir)
        for usa_obs_file in usa_obs_files:
            output_file_path = state_observer_dir / usa_obs_file.name
            if output_file_path.exists():
                continue
            if observer == metadata.DatasetNames.SSA:
                state_filter_val = None
            else:
                state_filter_val = state
            state_data = read_datafile(
                usa_obs_file, reset_index=False, state=state_filter_val
            )
            write_to_disk(state_data, output_file_path)
        logger.info(f"Finished writing {observer} files for {state_name}.")

    # Update metadata and proportions files for state
    metadata_proportions = pd.read_csv(usa_results_dir / "metadata_proportions.csv")
    state_proportions = metadata_proportions.loc[metadata_proportions["state"] == state]
    state_proportions.to_csv(state_dir / "metadata_proportions.csv", index=False)
    # Copy over metadata file
    metadata_path = usa_results_dir / "metadata.yaml"
    state_metadata_path = state_dir / "metadata.yaml"
    copyfile(metadata_path, state_metadata_path)


def write_shard_metadata(
    observer: str, obs_data: pd.DataFrame, obs_dir: Path, seed_ext: str
) -> None:
    # Writes metadata for each shard of data. This will be proportions available to noise
    # for specific columns.

    def _get_metadata_values(
        observer: str, df: pd.DataFrame, location: str, year: int
    ) -> pd.DataFrame:
        metadata_df = pd.DataFrame(index=[location])
        metadata_df["number_of_rows"] = len(df)
        metadata_df["year"] = year

        if observer in [
            metadata.DatasetNames.CENSUS,
            metadata.DatasetNames.ACS,
            metadata.DatasetNames.CPS,
        ]:
            df = calculate_guardian_duplication_metadata_proportions(metadata_df, df)

        for column_name in df.columns:
            columns = [col.name for col in COLUMNS]
            if column_name in columns:
                column = COLUMNS.get_column(column_name)
                noise_type_names = [noise_type.name for noise_type in column.noise_types]
                if "copy_from_household_member" in noise_type_names:
                    # Get number of rows that could potentially copy a household member
                    metadata_df[f"{column_name}.copy_from_household_member"] = (
                        df[COPY_HOUSEHOLD_MEMBER_COLS[column_name]].notna().sum()
                    )
                if "use_nickname" in noise_type_names:
                    # Get number of rows eligible to be noised to a nickname
                    nicknames = load_nicknames_data()
                    metadata_df[f"{column_name}.use_nickname"] = (
                        df[column_name].isin(nicknames.index).sum()
                    )
        return metadata_df

    # Get year column to group by
    obs_data = obs_data.copy()
    date_columns = ["year", "tax_year", "event_date", "survey_date"]
    year_col = [col for col in obs_data.columns if col in date_columns]
    state_col = "state" if "state" in obs_data.columns else "mailing_address_state"
    # For ACS, CPS, and SSA, we need to extract year from the year column because they are
    # currently dates
    if observer in [
        metadata.DatasetNames.ACS,
        metadata.DatasetNames.CPS,
        metadata.DatasetNames.SSA,
    ]:
        # Note: In these three datasets the year column is survey date or event date.
        obs_data["year"] = obs_data[year_col].squeeze().dt.year
        year_col = ["year"]
    # Special case SSA dataset since that does not have a state column
    if observer == metadata.DatasetNames.SSA:
        state_col = "state"
        obs_data[state_col] = "USA"

    # Merge dependents with their guardians. This is only implemented for census
    # but we have data to implement this for ACS and CPS as well.
    if observer in [
        metadata.DatasetNames.CENSUS,
        metadata.DatasetNames.ACS,
        metadata.DatasetNames.CPS,
    ]:
        obs_data = merge_dependents_and_guardians(observer, obs_data)
    groupby_cols = year_col + [state_col]
    metadata_dfs = []
    for (year, location), group_data in obs_data.groupby(groupby_cols):
        state_metadata = _get_metadata_values(observer, group_data, location, year)
        metadata_dfs.append(state_metadata)

    # We need to also calculate guardian duplication at the national level here since
    # each shard should have all guardian dependent pair.
    if observer == metadata.DatasetNames.CENSUS:
        for year in obs_data["year"].unique():
            national_metadata = pd.DataFrame(index=["USA"])
            national_metadata["year"] = year
            national_metadata = calculate_guardian_duplication_metadata_proportions(
                national_metadata, obs_data[obs_data["year"] == year]
            )
            metadata_dfs.append(national_metadata)

    shard_metadata = pd.concat(metadata_dfs)
    shard_metadata["dataset"] = observer

    # Write shard metadata
    shard_metadata_path = obs_dir / f"shard_metadata{seed_ext}.csv"
    shard_metadata.to_csv(shard_metadata_path, index_label="state")
