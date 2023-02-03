from typing import List, Tuple

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def immigrants_by_timestep(populations) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    timestep_values = []

    for before, after in zip(populations, populations[1:]):
        new_simulants = after.index.difference(before.index)
        immigrants = after.index[after["parent_id"] == -1].intersection(new_simulants)
        timestep_values.append((before, after, after.loc[immigrants]))

    return timestep_values


def test_there_is_immigration(immigrants_by_timestep):
    # This is a very rare event, so we can't assert that it happens
    # on every timestep; instead, we aggregate across all timesteps.
    all_time_immigrants = pd.concat(
        [immigrants for _, _, immigrants in immigrants_by_timestep]
    )

    assert len(all_time_immigrants) > 0


def test_immigration_into_gq(immigrants_by_timestep):
    all_time_gq_immigrants = pd.concat(
        [
            immigrants[immigrants["household_details.housing_type"] != "Standard"]
            for _, _, immigrants in immigrants_by_timestep
        ]
    )

    assert len(all_time_gq_immigrants) > 0


def test_immigration_into_existing_households(immigrants_by_timestep):
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

    assert len(all_time_existing_household_immigrants) > 0


def test_immigration_into_new_households(immigrants_by_timestep):
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

        # TODO: There should be some household immigrants
        # household_immigrants = new_household_immigrants[~is_non_reference_person_immigrant]
