from pathlib import Path
import pytest
from vivarium import Artifact


@pytest.fixture(scope="function")
def artifact(mocker):
    artifact_path = Path("/mnt/team/simulation_science/priv/engineering/vivarium_census_prl_synth_pop/data/artifacts/united_states_of_america.hdf")
    if artifact_path.exists():
        return Artifact(artifact_path)
    else:
        # Artifact will generate a new one at the path if it doesn't exist
        # We do not want this when testing so instead mock the path.
        artifact = mocker.MagicMock()
        artifact.path = artifact_path
        return artifact
