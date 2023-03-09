import pytest
from vivarium import Artifact


@pytest.fixture(scope="function")
def artifact() -> Artifact:
    """Loads the artifact"""
    artifact_path = "/mnt/team/simulation_science/priv/engineering/vivarium_census_prl_synth_pop/data/artifacts/united_states_of_america.hdf"
    return Artifact(artifact_path)
