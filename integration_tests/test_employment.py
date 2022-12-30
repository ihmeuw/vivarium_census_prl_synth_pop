import numpy as np
import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.constants import data_values

# TODO: Broader test coverage


# TODO stop skipping once MIC-3703 has been implemented
@pytest.mark.skip
def test_military_gq_employment(tracked_live_populations):
    for pop in tracked_live_populations:
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


def test_underage_are_unemployed(tracked_live_populations):
    for pop in tracked_live_populations:
        # All people in military group quarters are
        under_18 = pop[pop["age"] < 18]
        assert (under_18["employer_id"] == data_values.UNEMPLOYED_ID).all()


def test_unemployed_have_no_income(tracked_live_populations):
    for pop in tracked_live_populations:
        # All people in military group quarters are
        unemployed = pop[pop["employer_id"] == data_values.UNEMPLOYED_ID]
        assert (unemployed["income"] == 0).all()


def test_employed_have_income(tracked_live_populations):
    for pop in tracked_live_populations:
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


def test_employment_income_propensity_updates(tracked_live_populations):
    for time_step in range(max(TIME_STEPS_TO_TEST)):
        (
            employer_ids_before,
            employer_propensity_before,
            employer_ids_after,
            employer_propensity_after,
        ) = get_consecutive_time_step_employer_ids_and_test_column_for_comparable_sims(
            tracked_live_populations, time_step, "employer_income_propensity"
        )

        # Find those that change employers
        changed_jobs_idx = employer_ids_before.index[
            employer_ids_before != employer_ids_after
        ]

        assert (
            employer_propensity_before.loc[changed_jobs_idx]
            != employer_propensity_after.loc[changed_jobs_idx]
        ).all()


def test_no_employment_income_propensity_updates(tracked_live_populations):
    for time_step in range(max(TIME_STEPS_TO_TEST)):
        (
            employer_ids_before,
            employer_propensity_before,
            employer_ids_after,
            employer_propensity_after,
        ) = get_consecutive_time_step_employer_ids_and_test_column_for_comparable_sims(
            tracked_live_populations, time_step, "employer_income_propensity"
        )

        # Find those that change employers
        static_jobs_idx = employer_ids_before.index[employer_ids_before == employer_ids_after]

        assert (
            employer_propensity_before.loc[static_jobs_idx]
            == employer_propensity_after.loc[static_jobs_idx]
        ).all()


def test_personal_income_propensity_is_constant(tracked_live_populations):
    starting_personal_income_propensity = tracked_live_populations[min(TIME_STEPS_TO_TEST)][
        "personal_income_propensity"
    ]
    ending_personal_income_propensity = tracked_live_populations[max(TIME_STEPS_TO_TEST)][
        "personal_income_propensity"
    ]
    # Get index for sims that exist in both state tables
    present_sims_idx = starting_personal_income_propensity.index.intersection(
        ending_personal_income_propensity.index
    )

    assert (
        starting_personal_income_propensity.loc[present_sims_idx]
        == ending_personal_income_propensity.loc[present_sims_idx]
    ).all()


def test_income_updates(tracked_live_populations):
    for time_step in range(max(TIME_STEPS_TO_TEST)):
        (
            employer_ids_before,
            income_before,
            employer_ids_after,
            income_after,
        ) = get_consecutive_time_step_employer_ids_and_test_column_for_comparable_sims(
            tracked_live_populations, time_step, "income"
        )

        # Find those that change employers
        changed_jobs_idx = employer_ids_before.index[
            employer_ids_before != employer_ids_after
        ]

        assert (
            income_before.loc[changed_jobs_idx] != income_after.loc[changed_jobs_idx]
        ).all()


def test_no_income_uupdate(tracked_live_populations):
    for time_step in range(max(TIME_STEPS_TO_TEST)):
        (
            employer_ids_before,
            income_before,
            employer_ids_after,
            income_after,
        ) = get_consecutive_time_step_employer_ids_and_test_column_for_comparable_sims(
            tracked_live_populations, time_step, "income"
        )

        # Find those that change employers
        static_jobs_idx = employer_ids_before.index[employer_ids_before == employer_ids_after]

        assert (income_before.loc[static_jobs_idx] == income_after.loc[static_jobs_idx]).all()


def test_income_updates_when_age_changes(tracked_live_populations):
    # This test checks whether simulants who have a birthdays and therefore will have a new income distribution
    # Note: The other tests for income implementation are contigent on comparing simulants who are the same age and
    #  therefore ignore this edge case.
    for time_step in range(max(TIME_STEPS_TO_TEST)):
        pop_before = tracked_live_populations[time_step]
        pop_after = tracked_live_populations[time_step]
        common_sims_idx = pop_before.index.intersection(pop_after.index)
        age_before = np.floor(pop_before.loc[common_sims_idx, "age"])
        age_after = np.floor(pop_after.loc[common_sims_idx, "age"])
        birthday_sims_idx = age_before.index[age_before != age_after]
        common_sims_idx = common_sims_idx.intersection(birthday_sims_idx)

        # Get sims with same jobs but exclude unemployed
        employer_ids_before = pop_before.loc[common_sims_idx, "employer_id"]
        employer_ids_after = pop_after.loc[common_sims_idx, "employer_id"]
        same_jobs_idx = employer_ids_before.index[employer_ids_before == employer_ids_after]

        assert (
            pop_before.loc[same_jobs_idx, "income"] != pop_after.loc[same_jobs_idx, "income"]
        ).all()


def get_consecutive_time_step_employer_ids_and_test_column_for_comparable_sims(
    tracked_live_populations,
    time_step,
    test_column,
) -> tuple:
    pop_before = tracked_live_populations[time_step]
    pop_after = tracked_live_populations[time_step + 1]
    common_sims_idx = pop_before.index.intersection(pop_after.index)

    # Deal with sims who change age bins on time step
    # This is handling the cases where simulants enter the next age pin so their distribution for income changes
    age_before = np.floor(pop_before.loc[common_sims_idx, "age"])
    age_after = np.floor(pop_after.loc[common_sims_idx, "age"])
    same_age_idx = age_before.index[age_before == age_after]
    common_sims_idx = common_sims_idx.intersection(same_age_idx)

    employer_ids_before = pop_before.loc[common_sims_idx, "employer_id"]
    test_column_before = pop_before.loc[common_sims_idx, test_column]
    employer_ids_after = pop_after.loc[common_sims_idx, "employer_id"]
    test_column_after = pop_after.loc[common_sims_idx, test_column]

    return (
        employer_ids_before,
        test_column_before,
        employer_ids_after,
        test_column_after,
    )
