import tempfile

import numpy as np
import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.components.observers import HouseholdSurveyObserver

# TODO: Broader test coverage


@pytest.fixture
def observer(mocker):
    """Generate post-setup observer with mocked methods to patch as necessary"""
    observer = HouseholdSurveyObserver("ACS")
    builder = mocker.MagicMock()

    # create a temp directory so setup can generate an output directory
    tmpdir = tempfile.TemporaryDirectory()
    builder.configuration.output_data.results_directory = tmpdir.name
    observer.setup(builder)
    tmpdir.cleanup()
    return observer


@pytest.fixture
def mocked_pop_view():
    """Generate a state table view"""
    cols = HouseholdSurveyObserver.INPUT_COLS
    df = pd.DataFrame(np.random.randint(0, 100, size=(2, len(cols))), columns=cols)
    df["alive"] = "alive"
    df[["first_name", "middle_name", "last_name"]] = (
        ["Alex", "J", "Honnold"],
        ["Carolynn", "Marie", "Hill"],
    )

    return df


def test_instantiate(observer):
    assert str(observer) == "HouseholdSurveyObserver(ACS)"


def test_responses_schema(observer):
    """Is the initial self.responses (after setup) the expected schema
    (pd.DataFrame with correct columns)?
    """
    expected = pd.DataFrame(columns=HouseholdSurveyObserver.OUTPUT_COLS)
    pd.testing.assert_frame_equal(expected, observer.responses)


def test_do_observation(observer, mocked_pop_view, mocker):
    """Are responses recorded and with the the required columns?"""
    event = mocker.MagicMock()
    event.time = pd.to_datetime("2021-01-01")
    event.index = pd.Index([0, 1])
    observer.population_view.get.return_value = mocked_pop_view  # FIXME: This returns mocked_pop_view regardless of event.index or alive status
    mocker.patch(
        "vivarium_census_prl_synth_pop.components.observers.utilities.vectorized_choice",
        return_value=list(mocked_pop_view["household_id"]),
    )
    observer.do_observation(event)
    assert set(observer.responses.columns) == set(HouseholdSurveyObserver.OUTPUT_COLS)
