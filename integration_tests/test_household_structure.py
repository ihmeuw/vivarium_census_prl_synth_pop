import numpy as np
import pandas as pd

from integration_tests.conftest import TIME_STEPS_TO_TEST
from vivarium_census_prl_synth_pop.constants import data_values, metadata

# TODO: Broader test coverage


def test_housing_type_is_categorical(tracked_live_populations):
    for pop in tracked_live_populations:
        housing_type = pop["housing_type"]

        # Assert the dtype is correct and that there are no NaNs
        assert housing_type.dtype == pd.CategoricalDtype(categories=data_values.HOUSING_TYPES)
        assert not housing_type.isnull().any()


def test_relationship_is_categorical(tracked_live_populations):
    for pop in tracked_live_populations:
        relationship = pop["relation_to_household_head"]

        # Assert the dtype is correct and that there are no NaNs
        assert relationship.dtype == pd.CategoricalDtype(categories=metadata.RELATIONSHIPS)
        assert not relationship.isnull().any()


def test_all_households_have_reference_person(tracked_live_populations):
    for pop in tracked_live_populations:
        non_gq_household_ids = pop[
            ~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)
        ]["household_id"].unique()
        reference_person_household_ids = pop.loc[
            pop["relation_to_household_head"] == "Reference person", "household_id"
        ].values

        # Assert these two sets are identical
        assert non_gq_household_ids.size == reference_person_household_ids.size
        assert np.setxor1d(non_gq_household_ids, reference_person_household_ids).size == 0


def test_household_id_and_address_id_correspond(tracked_live_populations):
    for pop in tracked_live_populations:
        assert pop["household_id"].notnull().all()
        assert pop["address_id"].notnull().all()
        # 1-to-1 at any given point in time
        assert (pop.groupby("household_id")["address_id"].nunique() == 1).all()
        assert (pop.groupby("address_id")["household_id"].nunique() == 1).all()

    # Even over time, there is only 1 household_id for each address_id -- address_ids are not reused.
    all_time_pop = pd.concat(tracked_live_populations, ignore_index=True)
    assert (all_time_pop.groupby("address_id")["household_id"].nunique() == 1).all()
    # Note, however, that the reverse is not true: a household_id can span multiple address_ids
    # (over multiple time steps) when the whole house moved as a unit between those time steps.


def test_new_reference_person_is_oldest_household_member(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        before_reference_person_idx = before.index[
            before["relation_to_household_head"] == "Reference person"
        ]
        after_reference_person_idx = after.index[
            (after["relation_to_household_head"] == "Reference person")
            & (after["household_id"].isin(before["household_id"]))
        ]
        new_reference_person_idx = np.setdiff1d(
            after_reference_person_idx, before_reference_person_idx
        )

        # Get households with new reference persons
        household_ids_with_new_reference_person = after.loc[
            new_reference_person_idx, "household_id"
        ]
        households_with_new_reference_person_idx = after.index[
            after["household_id"].isin(household_ids_with_new_reference_person)
        ]
        oldest_members_of_affected_households = (
            after.loc[households_with_new_reference_person_idx]
            .groupby(["household_id"])["age"]
            .idxmax()
            .values
        )

        assert new_reference_person_idx.sort() == oldest_members_of_affected_households.sort()


def test_households_only_have_one_reference_person(tracked_live_populations):
    for pop in tracked_live_populations:
        household_ids = pop.loc[
            pop["relation_to_household_head"] == "Reference person", "household_id"
        ]

        assert len(household_ids) == len(household_ids.unique())

