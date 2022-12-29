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


def test_only_living_change_employment(populations):
    for before, after in zip(populations, populations[1:]):
        common_simulants = before.index.intersection(after.index)
        before = before.loc[common_simulants]
        after = after.loc[common_simulants]

        employment_changers = before["employer_id"] != after["employer_id"]
        assert after[employment_changers]["tracked"].all()
        assert (after[employment_changers]["alive"] == "alive").all()


def test_movers_change_employment(populations):
    for before, after in zip(populations, populations[1:]):
        common_working_age_simulants = before.index[before.age >= 18].intersection(
            after.index[after.age >= 18]
        )
        before = before.loc[common_working_age_simulants]
        after = after.loc[common_working_age_simulants]

        moved = before["address_id"] != after["address_id"]
        moved_to_military_gq = moved & (after["housing_type"] == "Military")
        previously_military_employed = before["employer_id"] == 1
        should_change = moved & (~moved_to_military_gq | ~previously_military_employed)
        assert (
            before[should_change]["employer_id"] != after[should_change]["employer_id"]
        ).all()
