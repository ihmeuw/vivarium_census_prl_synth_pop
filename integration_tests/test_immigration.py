from typing import List, Tuple

import pandas as pd
import pytest


@pytest.fixture(scope="class")
def immigrants_by_timestep(populations) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    timestep_values = []

    for before, after in zip(populations, populations[1:]):
        new_simulants = after.index.difference(before.index)
        # NOTE: We assume that immigrants never have tracked parents.
        # If this changes in the future, we'll need to identify them differently.
        immigrants = new_simulants.intersection(after.index[after["parent_id"] == -1])
        timestep_values.append((before, after, after.loc[immigrants]))

    return timestep_values


def test_there_is_immigration(immigrants_by_timestep, sim):
    # This is a very rare event, so we can't assert that it happens
    # on every timestep; instead, we aggregate across all timesteps.
    all_time_immigrants = pd.concat(
        [immigrants for _, _, immigrants in immigrants_by_timestep]
    )

    # Super conservative bounds on how much immigration should occur
    population_size = sim.configuration.population.population_size
    assert (population_size * 0.0005) < len(all_time_immigrants) < (population_size * 0.10)


def test_immigration_into_gq(immigrants_by_timestep, sim):
    all_time_gq_immigrants = pd.concat(
        [
            immigrants[immigrants["household_details.housing_type"] != "Standard"]
            for _, _, immigrants in immigrants_by_timestep
        ]
    )

    # GQ immigration is so rare we can't put a lower bound on it even at 20k population
    population_size = sim.configuration.population.population_size
    assert 0 < len(all_time_gq_immigrants) < (population_size * 0.05)


def test_immigration_into_existing_households(immigrants_by_timestep, sim):
    all_time_existing_household_immigrants = []

    for before, _, immigrants in immigrants_by_timestep:
        existing_households = before[before["household_details.housing_type"] == "Standard"][
            "household_id"
        ].unique()
        existing_household_immigrants = immigrants[
            immigrants["household_id"].isin(existing_households)
        ]

        # These will all be non-reference-person immigrants, who cannot have certain relationships
        expected_relationship = ~existing_household_immigrants[
            "relation_to_household_head"
        ].isin(
            [
                "Reference person",
                "Institutionalized GQ pop",
                "Noninstitutionalized GQ pop",
                "Parent",
                "Opp-sex spouse",
                "Opp-sex partner",
                "Same-sex spouse",
                "Same-sex partner",
            ]
        )

        assert expected_relationship.all()

        all_time_existing_household_immigrants.append(existing_household_immigrants)

    all_time_existing_household_immigrants = pd.concat(all_time_existing_household_immigrants)

    # Super conservative bounds on how much immigration should occur into existing households
    population_size = sim.configuration.population.population_size
    assert (
        (population_size * 0.00025)
        < len(all_time_existing_household_immigrants)
        < (population_size * 0.05)
    )


def test_immigration_into_new_households(immigrants_by_timestep):
    all_time_household_immigrants = []

    for before, after, immigrants in immigrants_by_timestep:
        existing_households = before[before["household_details.housing_type"] == "Standard"][
            "household_id"
        ].unique()
        new_household_immigrants = immigrants[
            (immigrants["household_details.housing_type"] == "Standard")
            & ~immigrants["household_id"].isin(existing_households)
        ]

        households_established_by_domestic_migration = after[
            (after["relation_to_household_head"] == "Reference person")
            & (~after.index.isin(immigrants.index))
            & (~after["household_id"].isin(existing_households))
        ]["household_id"].unique()

        is_non_reference_person_immigrant = new_household_immigrants["household_id"].isin(
            households_established_by_domestic_migration
        )
        non_reference_person_immigrants = new_household_immigrants[
            is_non_reference_person_immigrant
        ]

        expected_relationship = ~non_reference_person_immigrants[
            "relation_to_household_head"
        ].isin(
            [
                "Reference person",
                "Institutionalized GQ pop",
                "Noninstitutionalized GQ pop",
                "Parent",
                "Opp-sex spouse",
                "Opp-sex partner",
                "Same-sex spouse",
                "Same-sex partner",
            ]
        )

        assert expected_relationship.all()

        household_immigrants = new_household_immigrants[~is_non_reference_person_immigrant]
        all_time_household_immigrants.append(household_immigrants)

    all_time_household_immigrants = pd.concat(all_time_household_immigrants)

    assert len(all_time_household_immigrants) > 0
