import numpy as np
import pandas as pd

from vivarium_census_prl_synth_pop.components.population import Population
from vivarium_census_prl_synth_pop.constants import data_values, metadata, paths

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


fake_population = pd.DataFrame(
    data={
        "household_id": np.sort(np.array([1, 2, 3, 4, 5] * 5)),
        "date_of_birth": [
            pd.Timestamp("1987-03-01 00:00:00"),
            pd.Timestamp("2023-05-11 00:00:00"),
            pd.Timestamp("2026-04-01 00:00:00"),
            pd.Timestamp("2025-11-21 00:00:00"),
            pd.Timestamp("1992-07-25 00:00:00"),
            pd.Timestamp("1959-12-03 00:00:00"),
            pd.Timestamp("1983-01-31 00:00:00"),
            pd.Timestamp("1983-01-31 00:00:00"),
            pd.Timestamp("2025-04-30 00:00:00"),
            pd.Timestamp("1990-01-11 00:00:00"),
            pd.Timestamp("1965-11-08 00:00:00"),
            pd.Timestamp("1992-05-18 00:00:00"),
            pd.Timestamp("1967-02-05 00:00:00"),
            pd.Timestamp("1995-04-28 00:00:00"),
            pd.Timestamp("2028-08-13 00:00:00"),
            pd.Timestamp("1995-04-22 00:00:00"),
            pd.Timestamp("1996-07-02 00:00:00"),
            pd.Timestamp("1996-04-22 00:00:00"),
            pd.Timestamp("1996-09-16 00:00:00"),
            pd.Timestamp("1997-10-30 00:00:00"),
            pd.Timestamp("1988-12-22 00:00:00"),
            pd.Timestamp("2025-04-11 00:00:00"),
            pd.Timestamp("1990-02-14 00:00:00"),
            pd.Timestamp("1992-08-15 00:00:00"),
            pd.Timestamp("1990-04-01 00:00:00"),
        ],
        "age": [
            39,
            8,
            5,
            7,
            35,
            68,
            44,
            7,
            9,
            38,
            63,
            27,
            60,
            23,
            2,
            23,
            22,
            22,
            21,
            20,
            38,
            5,
            36,
            34,
            36,
        ],
        "guardian_1": [
            -1,
            0,
            300,
            -1,
            -1,
            -1,
            -1,
            100,
            100,
            -1,
            -1,
            -1,
            -1,
            -1,
            35,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            20,
            -1,
            -1,
            -1,
        ],
        "relation_to_household_head": [
            "Opp-sex spouse",
            "Stepchild",
            "Adopted child",
            "Adopted child",
            "Other relative",
            "Parent-in-law",
            "Same-sex spouse",
            "Biological child",
            "Biological child",
            "Sibling",
            "Parent",
            "Sibling",
            "Parent",
            "Opp-sex spouse",
            "Biological child",
            "Roommate",
            "Roommate",
            "Roommate",
            "Roommate",
            "Sibling",
            "Opp-sex spouse",
            "Biological child",
            "Roommate",
            "Other relative",
            "Other nonrelative",
        ],
        "housing_type": ["Standard"] * 25,
    }
)


def test_update_to_reference_person_and_relationships():

    expected_relationships = pd.Series(
        data=[
            "Reference person",
            "Biological child",
            "Adopted child",
            "Adopted child",
            "Other relative",
            "Reference person",
            "Biological child",
            "Grandchild",
            "Grandchild",
            "Other relative",
            "Reference person",
            "Biological child",
            "Other relative",
            "Child-in-law",
            "Grandchild",
            "Reference person",
            "Roommate",
            "Roommate",
            "Roommate",
            "Roommate",
            "Reference person",
            "Biological child",
            "Roommate",
            "Other relative",
            "Other nonrelative",
        ]
    )

    pop = Population()

    # Setup class methods we need to update fake state table
    pop.start_time = pd.Timestamp("2020-04-01 00:00:00")
    pop.reference_person_update_relationships_map = pd.read_csv(
        paths.REFERENCE_PERSON_UPDATE_RELATIONSHIP_DATA_PATH,
    )
    # This is a series with household_id as the index and the new reference person as the value
    expected_reference_person = (
        fake_population.loc[fake_population.index].groupby(["household_id"])["age"].idxmax()
    )
    updated_population = pop.update_reference_person_and_relationships(fake_population)

    assert not updated_population["relation_to_household_head"].isnull().any()
    assert (updated_population["relation_to_household_head"] == expected_relationships).all()

    new_reference_person_idx = updated_population.index[
        updated_population["relation_to_household_head"] == "Reference person"
    ]

    assert (expected_reference_person == new_reference_person_idx).all()
