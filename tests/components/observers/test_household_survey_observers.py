import numpy as np
import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.components.observers import HouseholdSurveyObserver

# TODO: Broader test coverage


@pytest.fixture
def observer(mocker, tmp_path):
    """Generate post-setup observer with mocked methods to patch as necessary"""
    obs = HouseholdSurveyObserver("acs")
    builder = mocker.MagicMock()
    builder.configuration.output_data.results_directory = tmp_path
    obs.setup_component(builder)
    return obs


@pytest.fixture
def mocked_pop_view(observer):
    """Generate a state table view"""
    cols = observer.columns_required
    df = pd.DataFrame(np.random.randint(0, 100, size=(2, len(cols))), columns=cols)
    df["alive"] = "alive"
    df[["first_name", "middle_name", "last_name"]] = (
        ["Alex", "J", "Honnold"],
        ["Carolynn", "Marie", "Hill"],
    )
    # Ensure there are no guardians in this dataset
    df[["guardian_1", "guardian_2"]] = np.random.randint(len(df), 100, size=(len(df), 2))
    df["po_box"] = [-1, -2]
    # Add copy from household member columns with dummy values instead of nans
    df["copy_age"] = [-1, -2]
    df["copy_date_of_birth"] = [-1, -2]
    # state table includes event time
    df["event_time"] = pd.to_datetime("2021-01-01")

    return df


@pytest.fixture
def mocked_household_details_pipeline(mocked_pop_view):
    def _mocked_pipeline(_):
        df = mocked_pop_view[["household_id"]]
        df["housing_type"] = ["Van", "Household"]
        df["address_id"] = [100, 200]
        df["po_box"] = [-1, -2]
        df["state_id"] = [-1, -2]
        df["puma"] = [-1, -2]
        return df

    return _mocked_pipeline


def test_instantiate(observer):
    assert observer.name == "household_survey_observer_acs"


def test_get_observation(
    observer, mocked_pop_view, mocked_household_details_pipeline, mocker
):
    """Are responses recorded correctly (including concatenation after time steps?"""
    sim_start_date = str(mocked_pop_view.event_time.iat[0].date())
    # Simulate first time step
    # FIXME: This returns mocked_pop_view regardless of event.index or alive status
    observer.population_view.get.return_value = mocked_pop_view
    mocker.patch(
        "vivarium_census_prl_synth_pop.components.observers.utilities.vectorized_choice",
        return_value=list(mocked_pop_view["household_id"]),
    )
    mocker.patch(
        "vivarium_census_prl_synth_pop.components.observers.utilities.copy_from_household_member",
        return_value=mocked_pop_view,
    )
    observer.pipelines["household_details"] = mocked_household_details_pipeline
    observation = observer.get_observation(mocked_pop_view)

    expected = mocked_pop_view
    expected["simulant_id"] = mocked_pop_view.index
    expected["middle_initial"] = ["J", "M"]
    expected[["guardian_1_address_id", "guardian_2_address_id"]] = np.nan
    expected["survey_date"] = [pd.to_datetime(sim_start_date)] * 2
    # FIXME: Having dtype with with datime64[s] and [ns] causes pd.testing.assert_frame_equal to fail
    expected["survey_date"] = expected["survey_date"].astype(observation["survey_date"].dtype)
    expected[
        ["housing_type", "address_id", "state_id", "puma"]
    ] = mocked_household_details_pipeline("dummy")[
        ["housing_type", "address_id", "state_id", "puma"]
    ]
    expected["copy_age"] = [-1, -2]
    expected["copy_date_of_birth"] = [-1, -2]
    pd.testing.assert_frame_equal(expected[observer.output_columns], observation)


def test_multiple_observation(
    observer, mocked_pop_view, mocked_household_details_pipeline, mocker
):
    """Are responses recorded correctly (including concatenation after time steps)?"""
    sim_start_date = str(mocked_pop_view.event_time.iat[0].date())
    # Simulate first time step
    # event = mocker.MagicMock()
    # event.time = pd.to_datetime(sim_start_date)
    # event.index = pd.Index([0, 1])
    # FIXME: This returns mocked_pop_view regardless of event.index or alive status
    observer.population_view.get.return_value = mocked_pop_view
    mocker.patch(
        "vivarium_census_prl_synth_pop.components.observers.utilities.vectorized_choice",
        return_value=list(mocked_pop_view["household_id"]),
    )
    mocker.patch(
        "vivarium_census_prl_synth_pop.components.observers.utilities.copy_from_household_member",
        return_value=mocked_pop_view,
    )
    observer.pipelines["household_details"] = mocked_household_details_pipeline
    results_1 = observer.get_observation(mocked_pop_view)
    # Simulate second time step
    event_time = pd.to_datetime(sim_start_date) + pd.Timedelta(28, "days")
    mocked_pop_view["event_time"] = event_time
    results_2 = observer.get_observation(mocked_pop_view)
    results = pd.concat([results_1, results_2])

    expected = pd.concat([mocked_pop_view] * 2)
    expected["simulant_id"] = mocked_pop_view.index.tolist() * 2
    expected["middle_initial"] = ["J", "M"] * 2
    expected[["guardian_1_address_id", "guardian_2_address_id"]] = np.nan
    expected["survey_date"] = [pd.to_datetime(sim_start_date)] * 2 + [event_time] * 2
    # FIXME: Having dtype with with datime64[s] and [ns] causes pd.testing.assert_frame_equal to fail
    expected["survey_date"] = expected["survey_date"].astype(results["survey_date"].dtype)
    expected[["housing_type", "address_id", "state_id", "puma"]] = pd.concat(
        [
            mocked_household_details_pipeline("dummy")[
                ["housing_type", "address_id", "state_id", "puma"]
            ]
        ]
        * 2,
        axis=0,
    )
    expected["copy_age"] = [-1, -2] * 2
    expected["copy_date_of_birth"] = [-1, -2] * 2
    pd.testing.assert_frame_equal(expected[observer.output_columns], results)
