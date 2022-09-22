from pathlib import Path

import vivarium_census_prl_synth_pop
from vivarium_census_prl_synth_pop.constants import metadata

REPO_DIR = Path(vivarium_census_prl_synth_pop.__file__).resolve().parent.parent.parent
BASE_DIR = Path(vivarium_census_prl_synth_pop.__file__).resolve().parent

PROJECT_ROOT = Path(f"/mnt/team/simulation_science/priv/engineering/{metadata.PROJECT_NAME}")
ARTIFACT_ROOT = PROJECT_ROOT / "data/artifacts/"
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RESULTS_ROOT = PROJECT_ROOT / "results/"

HOUSEHOLDS_DATA_DIR = PROJECT_ROOT / "data/raw_data/current/United_States/"
PERSONS_DATA_DIR = PROJECT_ROOT / "data/raw_data/current/United_States/"

HOUSEHOLD_MOVE_RATE_PATH = BASE_DIR / "data/raw_data/move_rates_including_gq.csv"

HOUSEHOLDS_FILENAMES = [f"psam_hus{x}.csv" for x in ["a", "b", "c", "d"]]
PERSONS_FILENAMES = [f"psam_pus{x}.csv" for x in ["a", "b", "c", "d"]]

SYNTHETIC_DATA_INPUTS_ROOT = Path(
    f"/mnt/team/simulation_science/priv/engineering/{metadata.PROJECT_NAME}/data/raw_data/synthetic_pii"
)
LAST_NAME_DATA_PATH = SYNTHETIC_DATA_INPUTS_ROOT / "Names_2010Census.csv"
ADDRESS_DATA_PATH = SYNTHETIC_DATA_INPUTS_ROOT / "deepparse_address_data_usa.csv.bz2"
