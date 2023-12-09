import math
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest
import yaml
from vivarium.framework.utilities import from_yearly

from vivarium_census_prl_synth_pop.constants import metadata

from .conftest import FuzzyChecker


@pytest.fixture(scope="module")
def target_immigration_events(sim):
    assert (
        metadata.UNITED_STATES_LOCATIONS == []
    ), "Automated V&V does not support subsets by US state"

    with open(Path(os.path.dirname(__file__)) / "v_and_v_inputs/immigration.yaml") as f:
        targets = yaml.safe_load(f)

    # Rescale to configured sim size, and from yearly to per-timestep
    targets_rescaled = {}
    for k in targets.keys():
        targets_rescaled[
            k.replace("_immigration_events_per_10k_starting_pop", "")
        ] = from_yearly(
            targets[k] * (sim.configuration.population.population_size / 10_000),
            pd.Timedelta(days=sim.configuration.time.step_size),
        )

    return targets_rescaled


@pytest.fixture(scope="module")
def immigrants_by_timestep(populations) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    timestep_values = []

    for before, after in zip(populations, populations[1:]):
        new_simulants = after.index.difference(before.index)
        # NOTE: We assume that immigrants never have tracked parents.
        # If this changes in the future, we'll need to identify them differently.
        immigrants = new_simulants.intersection(after.index[after["parent_id"] == -1])
        timestep_values.append((before, after, after.loc[immigrants]))

    return timestep_values


def test_immigration_into_gq(
    immigrants_by_timestep, target_immigration_events, fuzzy_checker: FuzzyChecker
):
    target_per_timestep = target_immigration_events["gq_person"]
    rounded_up = []

    for _, _, immigrants in immigrants_by_timestep:
        gq_immigrants = immigrants[immigrants["household_details.housing_type"] != "Standard"]
        assert (
            math.floor(target_per_timestep)
            <= len(gq_immigrants)
            <= math.ceil(target_per_timestep)
        )
        rounded_up.append(len(gq_immigrants) == math.ceil(target_per_timestep))

    fuzzy_checker.fuzzy_assert_proportion(
        "GQ immigration stochastic rounding",
        pd.Series(rounded_up),
        target_value=target_per_timestep % 1,
    )


def test_immigration_into_gq(immigrants_by_timestep, sim):
    all_time_gq_immigrants = pd.concat(
        [
            immigrants[immigrants["household_details.housing_type"] != "Household"]
            for _, _, immigrants in immigrants_by_timestep
        ]
    )

    # GQ immigration is so rare we can't put a lower bound on it even at 20k population
    population_size = sim.configuration.population.population_size
    assert 0 < len(all_time_gq_immigrants) < (population_size * 0.05)


def test_immigration_into_existing_households(
    immigrants_by_timestep, target_immigration_events, fuzzy_checker: FuzzyChecker
):
    target_per_timestep = target_immigration_events["non_reference_person"]
    rounded_up = []

    for before, after, immigrants in immigrants_by_timestep:
        previous_timestep_households = before[
            before["household_details.housing_type"] == "Standard"
        ]["household_id"]

        households_established_by_domestic_migration = after[
            (after["relationship_to_reference_person"] == "Reference person")
            & (~after.index.isin(immigrants.index))
            & (~after["household_id"].isin(previous_timestep_households))
        ]["household_id"]

        # NOTE: Households can also be established by immigration itself, but because this
        # happens after individual immigration, we don't have to worry about a non-reference-person
        # immigrant joining a household that was created by immigration on the same timestep.
        existing_households = pd.concat(
            [previous_timestep_households, households_established_by_domestic_migration],
            ignore_index=True,
        ).unique()

        existing_household_immigrants = immigrants[
            immigrants["household_id"].isin(existing_households)
        ]

        # These will all be non-reference-person immigrants, who cannot have certain relationships
        expected_relationship = ~existing_household_immigrants[
            "relationship_to_reference_person"
        ].isin(
            [
                "Reference person",
                "Institutionalized group quarters population",
                "Noninstitutionalized group quarters population",
                "Parent",
                "Opposite-sex spouse",
                "Opposite-sex unmarried partner",
                "Same-sex spouse",
                "Same-sex unmarried partner",
            ]
        )

        assert expected_relationship.all()

        assert (
            math.floor(target_per_timestep)
            <= len(existing_household_immigrants)
            <= math.ceil(target_per_timestep)
        )
        rounded_up.append(
            len(existing_household_immigrants) == math.ceil(target_per_timestep)
        )

    fuzzy_checker.fuzzy_assert_proportion(
        "Non-reference-person immigration stochastic rounding",
        pd.Series(rounded_up),
        target_value=target_per_timestep % 1,
    )


def test_immigration_into_new_households(
    immigrants_by_timestep, target_immigration_events, fuzzy_checker: FuzzyChecker
):
    target_per_timestep = target_immigration_events["household"]
    rounded_up = []

    for before, after, immigrants in immigrants_by_timestep:
        existing_households = before[before["household_details.housing_type"] == "Household"][
            "household_id"
        ].unique()
        new_household_immigrants = immigrants[
            (immigrants["household_details.housing_type"] == "Household")
            & ~immigrants["household_id"].isin(existing_households)
        ]

        households_established_by_domestic_migration = after[
            (after["relationship_to_reference_person"] == "Reference person")
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
            "relationship_to_reference_person"
        ].isin(
            [
                "Reference person",
                "Institutionalized group quarters population",
                "Noninstitutionalized group quarters population",
                "Parent",
                "Opposite-sex spouse",
                "Opposite-sex unmarried partner",
                "Same-sex spouse",
                "Same-sex unmarried partner",
            ]
        )

        assert expected_relationship.all()

        household_immigration_events = len(
            new_household_immigrants[
                new_household_immigrants["relationship_to_reference_person"]
                == "Reference person"
            ]
        )

        assert (
            math.floor(target_per_timestep)
            <= household_immigration_events
            <= math.ceil(target_per_timestep)
        )
        rounded_up.append(household_immigration_events == math.ceil(target_per_timestep))

    fuzzy_checker.fuzzy_assert_proportion(
        "Household immigration stochastic rounding",
        pd.Series(rounded_up),
        target_value=target_per_timestep % 1,
    )
