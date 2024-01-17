from vivarium_census_prl_synth_pop.constants.paths import DEFAULT_ARTIFACT


def test_read_artifact():
    assert DEFAULT_ARTIFACT.exists()
