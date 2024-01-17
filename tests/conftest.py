from pathlib import Path

import pytest
from vivarium import Artifact

from vivarium_census_prl_synth_pop.constants.paths import DEFAULT_ARTIFACT


@pytest.fixture()
def artifact(mocker):
    artifact_path = Path(DEFAULT_ARTIFACT)
    if artifact_path.exists():
        return Artifact(artifact_path)
    else:
        # Artifact will generate a new one at the path if it doesn't exist
        # We do not want this when testing so instead mock the path.
        artifact = mocker.MagicMock()
        artifact.path = artifact_path
        return artifact
