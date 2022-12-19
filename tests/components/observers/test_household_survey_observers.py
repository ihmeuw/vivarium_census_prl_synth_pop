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
    return observer


@pytest.fixture
def mocked_pop_view(observer):
    """Generate a state table view"""
    cols = observer.input_columns
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
    pd.testing.assert_frame_equal(
        pd.DataFrame(columns=observer.output_columns), observer.responses
    )


def test_do_observation(observer, mocked_pop_view, mocker):
    """Are responses recorded correctly (including concatenation after time steps?"""
    sim_start_date = "2021-01-01"
    # Simulate first time step
    event = mocker.MagicMock()
    event.time = pd.to_datetime(sim_start_date)
    event.index = pd.Index([0, 1])
    observer.population_view.get.return_value = mocked_pop_view  # FIXME: This returns mocked_pop_view regardless of event.index or alive status
    mocker.patch(
        "vivarium_census_prl_synth_pop.components.observers.utilities.vectorized_choice",
        return_value=list(mocked_pop_view["household_id"]),
    )
    observer.do_observation(event)
    # Simulate second time step
    event.time = event.time + pd.Timedelta(28, "days")
    observer.do_observation(event)

    expected = pd.concat([mocked_pop_view] * 2)
    expected["middle_initial"] = ["J", "M"] * 2
    expected[["guardian_1_address_id", "guardian_2_address_id"]] = np.nan
    expected["survey_date"] = [pd.to_datetime(sim_start_date).date()] * 2 + [
        event.time.date()
    ] * 2

    pd.testing.assert_frame_equal(
        expected[observer.output_columns], observer.responses, check_dtype=False
    )
