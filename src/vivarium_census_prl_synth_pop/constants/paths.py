from pathlib import Path

import vivarium_census_prl_synth_pop
from vivarium_census_prl_synth_pop.constants import metadata

BASE_DIR = Path(vivarium_census_prl_synth_pop.__file__).resolve().parent

PROJECT_ROOT = Path(f'/mnt/team/simulation_science/priv/engineering/{metadata.PROJECT_NAME}')
ARTIFACT_ROOT = PROJECT_ROOT / 'data/artifacts/'
MODEL_SPEC_DIR = BASE_DIR / 'model_specifications'
RESULTS_ROOT = PROJECT_ROOT / 'results/'

HOUSEHOLDS_DATA_DIR = PROJECT_ROOT / 'data/raw_data/current/United_States/'
PERSONS_DATA_DIR = PROJECT_ROOT / 'data/raw_data/current/'

HOUSEHOLDS_FNAMES = [f'psam_hus{x}.csv' for x in ['a', 'b', 'c', 'd']]  # TODO: update this once abie gives us new data
PERSONS_FNAME = 'psam_p12.csv' # TODO: this is the name of the florida persons file we downloaded. # will need to be updated / made more general once we have more/new data form abie
