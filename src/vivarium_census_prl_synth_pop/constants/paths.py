from pathlib import Path

import vivarium_census_prl_synth_pop
from vivarium_census_prl_synth_pop.constants import metadata

REPO_DIR = Path(vivarium_census_prl_synth_pop.__file__).resolve().parent.parent.parent
BASE_DIR = Path(vivarium_census_prl_synth_pop.__file__).resolve().parent

PROJECT_ROOT = Path(f"/mnt/team/simulation_science/priv/engineering/{metadata.PROJECT_NAME}")
ARTIFACT_ROOT = PROJECT_ROOT / "data/artifacts/"
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"

RAW_RESULTS_DIR_NAME = Path("raw_results")
FINAL_RESULTS_DIR_NAME = Path("final_results")

HOUSEHOLDS_DATA_DIR = PROJECT_ROOT / "data/raw_data/current/United_States/"
PERSONS_DATA_DIR = PROJECT_ROOT / "data/raw_data/current/United_States/"

INDIVIDUAL_DOMESTIC_MIGRATION_RATES_PATH = (
    BASE_DIR / "data/raw_data/individual_domestic_migration_rates.csv"
)
HOUSEHOLD_DOMESTIC_MIGRATION_RATES_PATH = (
    BASE_DIR / "data/raw_data/household_domestic_migration_rates.csv"
)

NON_REFERENCE_PERSON_EMIGRATION_RATES_PATH = (
    BASE_DIR / "data/raw_data/non_reference_person_emigration_rates.csv"
)
GQ_PERSON_EMIGRATION_RATES_PATH = (
    BASE_DIR / "data/raw_data/group_quarters_person_emigration_rates.csv"
)
HOUSEHOLD_EMIGRATION_RATES_PATH = BASE_DIR / "data/raw_data/household_emigration_rates.csv"

BUSINESS_NAMES_DATA = PROJECT_ROOT / "data/raw_data/business_names.csv.bz2"
BUSINESS_NAMES_DATA_ARTIFACT_INPUT_PATH = (
    PROJECT_ROOT / "data/raw_data/business_names_generation/"
)

HOUSEHOLDS_FILENAMES = [f"psam_hus{x}.csv" for x in ["a", "b", "c", "d"]]
PERSONS_FILENAMES = [f"psam_pus{x}.csv" for x in ["a", "b", "c", "d"]]

SYNTHETIC_DATA_INPUTS_ROOT = Path(
    f"/mnt/team/simulation_science/priv/engineering/{metadata.PROJECT_NAME}/data/raw_data/synthetic_pii"
)
LAST_NAME_DATA_PATH = SYNTHETIC_DATA_INPUTS_ROOT / "Names_2010Census.csv"
ADDRESS_DATA_PATH = SYNTHETIC_DATA_INPUTS_ROOT / "deepparse_address_data_usa.csv.bz2"

INCOME_DISTRIBUTIONS_DATA_PATH = (
    BASE_DIR / "data/raw_data/income_scipy_lognorm_distribution_parameters.csv"
)
