import pytest

from vivarium import Artifact


@pytest.fixture(scope="session")
def artifact() -> Artifact:
    """Loads the artifact"""
    # artifact_path = "/mnt/team/simulation_science/priv/engineering/vivarium_census_prl_synth_pop/data/artifacts/united_states_of_america.hdf"
    artifact_path = "/mnt/share/homes/sbachmei/scratch/vivarium/prl/artifacts/ids-dataframe/united_states_of_america.hdf"
    return Artifact(artifact_path)