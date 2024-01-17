from pathlib import Path
from vivarium import Artifact

from vivarium_census_prl_synth_pop.constants.paths import DEFAULT_ARTIFACT


def read_artifact():
    assert DEFAULT_ARTIFACT.exists()
