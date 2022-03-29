from pathlib import Path

import vivarium_census_prl_synth_pop
from vivarium_census_prl_synth_pop.constants import metadata

BASE_DIR = Path(vivarium_census_prl_synth_pop.__file__).resolve().parent

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{metadata.PROJECT_NAME}/")
MODEL_SPEC_DIR = BASE_DIR / 'model_specifications'
RESULTS_ROOT = Path(f'/share/costeffectiveness/results/{metadata.PROJECT_NAME}/')
