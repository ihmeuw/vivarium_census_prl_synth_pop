import numpy as np
import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.constants import data_values, paths

# TODO: Broader test coverage


@pytest.mark.skip(reason="waiting for MIC-3703 to be implemented")
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
                [data_values.MilitaryEmployer.EMPLOYER_ID, data_values.Unemployed.EMPLOYER_ID]
            )
        ).all()


def test_underage_are_unemployed(tracked_live_populations):
    for pop in tracked_live_populations:
        # All people in military group quarters are
        under_18 = pop[pop["age"] < 18]
        assert (under_18["employer_id"] == data_values.Unemployed.EMPLOYER_ID).all()


def test_unemployed_have_no_income(tracked_live_populations):
    for pop in tracked_live_populations:
        # All people in military group quarters are
        unemployed = pop[pop["employer_id"] == data_values.Unemployed.EMPLOYER_ID]
        assert (unemployed["income"] == 0).all()


def test_employed_have_income(tracked_live_populations):
    for pop in tracked_live_populations:
        # All people in military group quarters are
        employed = pop[pop["employer_id"] != data_values.Unemployed.EMPLOYER_ID]
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
        common_alive_simulants = before[before["alive"] == "alive"].index.intersection(
            after[after["alive"] == "alive"].index
        )
        common_working_age_simulants = before.index[before["age"] >= 18].intersection(
            after.index[after["age"] >= 18]
        )
        before = before.loc[common_working_age_simulants.intersection(common_alive_simulants)]
        after = after.loc[common_working_age_simulants.intersection(common_alive_simulants)]

        moved = (
            before["household_details.address_id"] != after["household_details.address_id"]
        )
        moved_to_military_gq = moved & (after["household_details.housing_type"] == "Military")
        previously_military_employed = before["employer_id"] == 1
        should_change = moved & (~moved_to_military_gq | ~previously_military_employed)
        assert (
            before[should_change]["employer_id"] != after[should_change]["employer_id"]
        ).all()


def test_employment_income_propensity_updates(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        changed_jobs = (before["employer_id"] != after["employer_id"]) & (
            before["age"] > data_values.WORKING_AGE
        )
        changed_employer_propensity = (
            before["employer_income_propensity"] != after["employer_income_propensity"]
        ) & (before["age"] > data_values.WORKING_AGE)

        assert (changed_jobs == changed_employer_propensity).all()


def test_personal_income_propensity_is_constant(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        assert (
            before["personal_income_propensity"] == after["personal_income_propensity"]
        ).all()


def test_income_updates_for_same_age_simulants(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        changed_jobs = (before["employer_id"] != after["employer_id"]) & (
            np.floor(before["age"]) == np.floor(after["age"])
        )
        changed_income = (before["income"] != after["income"]) & (
            np.floor(before["age"]) == np.floor(after["age"])
        )

        assert (changed_jobs == changed_income).all()


def test_income_updates_when_age_changes(simulants_on_adjacent_timesteps):
    # Checks whether simulants who age into the next income age bin and therefore will have a new income distribution
    # Note: The other tests for income implementation are contigent on comparing simulants who are the same age and
    #  therefore ignore this edge case.
    # Uses age bins from income distribution CSV
    income_distributions = pd.read_csv(paths.INCOME_DISTRIBUTIONS_DATA_PATH)
    ages = income_distributions["age_end"]

    for before, after in simulants_on_adjacent_timesteps:
        for age in ages:
            birthdays_idx = before.index[
                (before["employer_id"] == after["employer_id"])
                & (before["employer_id"] != data_values.Unemployed.EMPLOYER_ID)
                & (before["age"] < age)
                & (after["age"] > age)
            ]

            assert (
                before.loc[birthdays_idx, "income"] != after.loc[birthdays_idx, "income"]
            ).all()


@pytest.mark.skip(reason="TODO when employer_name_id is implemented")
def test_employer_name_uniqueness(simulants_on_adjacent_timesteps):
    """Employers should have unique names that never change"""
    for before, after in simulants_on_adjacent_timesteps:
        ...
