from typing import List, Tuple

import pandas as pd
import pytest

from .conftest import FuzzyTest


@pytest.fixture(scope="class")
def immigrants_by_timestep(populations) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    timestep_values = []

    for before, after in zip(populations, populations[1:]):
        new_simulants = after.index.difference(before.index)
        # NOTE: We assume that immigrants never have tracked parents.
        # If this changes in the future, we'll need to identify them differently.
        immigrant_idx = new_simulants.intersection(after.index[after["parent_id"] == -1])
        timestep_values.append(
            (
                before,
                after,
                pd.Series(after.index.isin(immigrant_idx), index=after.index),
                after.loc[immigrant_idx],
            )
        )

    return timestep_values


def test_there_is_immigration(immigrants_by_timestep, fuzzy_tester: FuzzyTest):
    all_time_immigrant_status = pd.concat(
        [immigrant_status for _, _, immigrant_status, _ in immigrants_by_timestep]
    )

    # How much immigration should occur, as a proportion of the population?
    # There are roughly 1.85 million immigrants per year 2016-2020.
    approx_immigration_rate = (1_850_000 / 330_000_000) / 12
    fuzzy_tester.fuzzy_assert_proportion(
        all_time_immigrant_status,
        # A bit of leeway: the above number should be pretty faithfully
        # replicated, but the population of the sim could change a bit over time
        # and immigration wouldn't scale with it
        true_value_min=approx_immigration_rate * 0.9,
        true_value_max=approx_immigration_rate * 1.1,
    )


def test_immigration_into_gq(immigrants_by_timestep, fuzzy_tester: FuzzyTest):
    all_time_gq_immigrant_status = pd.concat(
        [
            immigrant_status & (after["household_details.housing_type"] != "Standard")
            for _, after, immigrant_status, _ in immigrants_by_timestep
        ]
    )

    # There are roughly 1.85 million immigrants per year 2016-2020.
    # GQ is about a tenth.
    approx_gq_immigration_rate = ((1_850_000 / 10) / 330_000_000) / 12
    fuzzy_tester.fuzzy_assert_proportion(
        all_time_gq_immigrant_status,
        # Some leeway, due to inaccuracy of above numbers, and the fact
        # that immigration does not actually scale with population size
        true_value_min=approx_gq_immigration_rate * 0.8,
        true_value_max=approx_gq_immigration_rate * 1.2,
    )


def test_immigration_into_existing_households(
    immigrants_by_timestep, fuzzy_tester: FuzzyTest
):
    all_time_existing_household_immigrant_status = []

    for before, after, immigrant_status, immigrants in immigrants_by_timestep:
        existing_households = before[before["household_details.housing_type"] == "Standard"][
            "household_id"
        ].unique()
        existing_household_immigrant_status = immigrant_status & after["household_id"].isin(
            existing_households
        )
        existing_household_immigrants = after[existing_household_immigrant_status]

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

        all_time_existing_household_immigrant_status.append(
            existing_household_immigrant_status
        )

    all_time_existing_household_immigrant_status = pd.concat(
        all_time_existing_household_immigrant_status
    )

    # There are roughly 1.85 million immigrants per year 2016-2020.
    # Non-reference-person is about half.
    approx_nrp_immigration_rate = ((1_850_000 / 2) / 330_000_000) / 12
    fuzzy_tester.fuzzy_assert_proportion(
        all_time_existing_household_immigrant_status,
        # Some leeway, due to inaccuracy of above numbers, and the fact
        # that immigration does not actually scale with population size
        true_value_min=approx_nrp_immigration_rate * 0.8,
        true_value_max=approx_nrp_immigration_rate * 1.2,
    )


def test_immigration_into_new_households(immigrants_by_timestep, fuzzy_tester: FuzzyTest):
    all_time_household_immigrant_status = []

    for before, after, immigrant_status, immigrants in immigrants_by_timestep:
        existing_households = before[before["household_details.housing_type"] == "Standard"][
            "household_id"
        ].unique()
        new_household_immigrant_status = (
            immigrant_status
            & (after["household_details.housing_type"] == "Standard")
            & ~after["household_id"].isin(existing_households)
        )

        households_established_by_domestic_migration = after[
            (after["relation_to_household_head"] == "Reference person")
            & (~after.index.isin(immigrants.index))
            & (~after["household_id"].isin(existing_households))
        ]["household_id"].unique()

        is_non_reference_person_immigrant = new_household_immigrant_status & after[
            "household_id"
        ].isin(households_established_by_domestic_migration)
        non_reference_person_immigrants = after[is_non_reference_person_immigrant]

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

        household_immigrant_status = (
            new_household_immigrant_status & ~is_non_reference_person_immigrant
        )
        all_time_household_immigrant_status.append(household_immigrant_status)

    all_time_household_immigrant_status = pd.concat(all_time_household_immigrant_status)

    # There are roughly 1.85 million immigrants per year 2016-2020.
    # Household is about half.
    approx_household_immigration_rate = ((1_850_000 / 2) / 330_000_000) / 12
    fuzzy_tester.fuzzy_assert_proportion(
        all_time_household_immigrant_status,
        # Some leeway, due to inaccuracy of above numbers, and the fact
        # that immigration does not actually scale with population size
        true_value_min=approx_household_immigration_rate * 0.8,
        true_value_max=approx_household_immigration_rate * 1.2,
    )
