import pytest

from integration_tests.conftest import TIME_STEPS_TO_TEST
from vivarium_census_prl_synth_pop.constants import data_values

# TODO: Broader test coverage

# TODO stop skipping once MIC-3703 has been implemented
@pytest.mark.parametrize("time_step", TIME_STEPS_TO_TEST)
@pytest.mark.skip
def test_military_gq_employment(tracked_live_populations, time_step):
    pop = tracked_live_populations[time_step]
    # All people in military group quarters are
    military_gq = pop[
        pop["household_id"] == data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS["Military"]
    ]
    assert (
        military_gq["employer_id"].isin(
            # todo they are not allowed to be unemployed.
            #  Once this bug is fixed, change the test
            [data_values.MilitaryEmployer.EMPLOYER_ID, data_values.UNEMPLOYED_ID]
        )
    ).all()


@pytest.mark.parametrize("time_step", TIME_STEPS_TO_TEST)
def test_underage_are_unemployed(tracked_live_populations, time_step):
    pop = tracked_live_populations[time_step]
    # All people in military group quarters are
    under_18 = pop[pop["age"] < 18]
    assert (under_18["employer_id"] == data_values.UNEMPLOYED_ID).all()


@pytest.mark.parametrize("time_step", TIME_STEPS_TO_TEST)
def test_unemployed_have_no_income(tracked_live_populations, time_step):
    pop = tracked_live_populations[time_step]
    # All people in military group quarters are
    unemployed = pop[pop["employer_id"] == data_values.UNEMPLOYED_ID]
    assert (unemployed["income"] == 0).all()


@pytest.mark.parametrize("time_step", TIME_STEPS_TO_TEST)
def test_employed_have_income(tracked_live_populations, time_step):
    pop = tracked_live_populations[time_step]
    # All people in military group quarters are
    employed = pop[pop["employer_id"] != data_values.UNEMPLOYED_ID]
    assert (employed["income"] > 0).all()