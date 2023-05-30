import numpy as np
import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.components.observers import HouseholdSurveyObserver
from vivarium_census_prl_synth_pop.constants import paths

# TODO: Broader test coverage


@pytest.fixture(params=["hdf", "parquet"])
def observer(mocker, tmp_path, request):
    """Generate post-setup observer with mocked methods to patch as necessary"""
    observer = HouseholdSurveyObserver("acs")
    builder = mocker.MagicMock()
    builder.configuration.output_data.results_directory = tmp_path
    builder.configuration["household_survey_observer_acs"].file_extension = request.param
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
    # Ensure there are no guardians in this dataset
    df[["guardian_1", "guardian_2"]] = np.random.randint(len(df), 100, size=(len(df), 2))
    df["po_box"] = [-1, -2]
    # Add copy from household member columns with dummy values instead of nans
    df["copy_age"] = [-1, -2]
    df["copy_date_of_birth"] = [-1, -2]

    return df


@pytest.fixture
def mocked_household_details_pipeline(mocked_pop_view):
    def _mocked_pipeline(_):
        df = mocked_pop_view[["household_id"]]
        df["housing_type"] = ["Van", "Standard"]
        df["address_id"] = [100, 200]
        df["po_box"] = [-1, -2]
        df["state_id"] = [-1, -2]
        df["puma"] = [-1, -2]
        return df

    return _mocked_pipeline


def test_instantiate(observer):
    assert str(observer) == "HouseholdSurveyObserver(acs)"


def test_on_simulation_end(observer, mocker):
    """Are the final results written out to the expected directory?"""
    event = mocker.MagicMock()
    observer.responses = pd.DataFrame()
    observer.on_simulation_end(event)
    assert (
        observer.output_dir
        / paths.RAW_RESULTS_DIR_NAME
        / observer.output_name
        / f"{observer.output_name}_{observer.seed}.{observer.file_extension}"
    ).is_file()


def test_get_observation(
    observer, mocked_pop_view, mocked_household_details_pipeline, mocker
):
    """Are responses recorded correctly (including concatenation after time steps?"""
    sim_start_date = "2021-01-01"
    # Simulate first time step
    event = mocker.MagicMock()
    event.time = pd.to_datetime(sim_start_date)
    event.index = pd.Index([0, 1])
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
    observation = observer.get_observation(event)

    expected = mocked_pop_view
    expected["middle_initial"] = ["J", "M"]
    expected[["guardian_1_address_id", "guardian_2_address_id"]] = np.nan
    expected["survey_date"] = [pd.to_datetime(sim_start_date)] * 2
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
    sim_start_date = "2021-01-01"
    # Simulate first time step
    event = mocker.MagicMock()
    event.time = pd.to_datetime(sim_start_date)
    event.index = pd.Index([0, 1])
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
    observer.on_collect_metrics(event)
    # Simulate second time step
    event.time = event.time + pd.Timedelta(28, "days")
    observer.on_collect_metrics(event)

    expected = pd.concat([mocked_pop_view] * 2)
    expected["middle_initial"] = ["J", "M"] * 2
    expected[["guardian_1_address_id", "guardian_2_address_id"]] = np.nan
    expected["survey_date"] = [pd.to_datetime(sim_start_date)] * 2 + [event.time] * 2
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
    pd.testing.assert_frame_equal(expected[observer.output_columns], observer.responses)
