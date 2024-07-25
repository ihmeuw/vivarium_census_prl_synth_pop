from pathlib import Path

import vivarium_census_prl_synth_pop
from vivarium_census_prl_synth_pop.constants import metadata

REPO_DIR = Path(vivarium_census_prl_synth_pop.__file__).resolve().parent.parent.parent
BASE_DIR = Path(vivarium_census_prl_synth_pop.__file__).resolve().parent

PROJECT_ROOT = Path(f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RAW_DATA_ROOT = BASE_DIR / "data" / "raw_data"

RAW_RESULTS_DIR_NAME = "results"
FINAL_RESULTS_DIR_NAME = "final_results"

PROCESSED_RESULTS_DIR_NAME_BASE = "pseudopeople_input_data_usa"
BEST_SYMLINK_NAME = "best"
LATEST_SYMLINK_NAME = "latest"

HOUSEHOLDS_DATA_DIR = PROJECT_ROOT / "data/raw_data/current/United_States"
PERSONS_DATA_DIR = PROJECT_ROOT / "data/raw_data/current/United_States"

INDIVIDUAL_DOMESTIC_MIGRATION_RATES_PATH = (
    RAW_DATA_ROOT / "individual_domestic_migration_rates.csv"
)
HOUSEHOLD_DOMESTIC_MIGRATION_RATES_PATH = (
    RAW_DATA_ROOT / "household_domestic_migration_rates.csv"
)

NON_REFERENCE_PERSON_EMIGRATION_RATES_PATH = (
    RAW_DATA_ROOT / "non_reference_person_emigration_rates.csv"
)
GQ_PERSON_EMIGRATION_RATES_PATH = RAW_DATA_ROOT / "group_quarters_person_emigration_rates.csv"
HOUSEHOLD_EMIGRATION_RATES_PATH = RAW_DATA_ROOT / "household_emigration_rates.csv"

BUSINESS_NAMES_DATA = PROJECT_ROOT / "data/raw_data/business_names.csv.bz2"
BUSINESS_NAMES_DATA_ARTIFACT_INPUT_PATH = (
    PROJECT_ROOT / "data/raw_data/business_names_generation"
)

HOUSEHOLDS_FILENAMES = [f"psam_hus{x}.csv" for x in ["a", "b", "c", "d"]]
PERSONS_FILENAMES = [f"psam_pus{x}.csv" for x in ["a", "b", "c", "d"]]

DEFAULT_ARTIFACT = ARTIFACT_ROOT / "united_states_of_america.hdf"
SYNTHETIC_DATA_INPUTS_ROOT = Path(
    f"/mnt/team/simulation_science/priv/engineering/{metadata.PROJECT_NAME}/data/raw_data/synthetic_pii"
)
LAST_NAME_DATA_PATH = SYNTHETIC_DATA_INPUTS_ROOT / "Names_2010Census.csv"
ADDRESS_DATA_PATH = SYNTHETIC_DATA_INPUTS_ROOT / "deepparse_address_data_usa.csv.bz2"

INCOME_DISTRIBUTIONS_DATA_PATH = (
    RAW_DATA_ROOT / "income_scipy_lognorm_distribution_parameters.csv"
)
REFERENCE_PERSON_UPDATE_RELATIONSHIP_DATA_PATH = (
    RAW_DATA_ROOT / "reference_person_update_relationship_mapping.csv"
)

PUMA_TO_ZIP_DATA_PATH = RAW_DATA_ROOT / "puma_to_zip.csv"
GENERATED_BUSINESS_NAMES_DATA_PATH = (
    PROJECT_ROOT
    / "data/raw_data"
    / "business_names_generation/v0.1/generated_business_names.csv.bz2"
)

NICKNAMES_DATA_PATH = RAW_DATA_ROOT / "nicknames.csv"

CHANGELOG_PATH = REPO_DIR / "CHANGELOG.rst"
