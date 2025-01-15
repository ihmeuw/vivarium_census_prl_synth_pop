import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from vivarium.framework.utilities import from_yearly

from vivarium_census_prl_synth_pop.constants import data_values, metadata

from .conftest import (
    SIMULATION_STEP_SIZE,
    FuzzyChecker,
    from_yearly_multiplicative_drift,
    multiplicative_drifts_to_bounds_at_timestep,
)


@pytest.fixture(scope="module")
def target_state_proportions():
    assert (
        metadata.UNITED_STATES_LOCATIONS == []
    ), "Integration tests do not support subsets by US state"

    with open(
        Path(os.path.dirname(__file__)) / "v_and_v_inputs/household_structure.yaml"
    ) as f:
        targets = yaml.safe_load(f)["state_proportions"]

    result = {
        "states": targets["states"],
    }

    result["multiplicative_drift"] = {
        "lower_bound": from_yearly_multiplicative_drift(
            targets["multiplicative_drift_per_year"]["lower_bound"],
            time_step_days=SIMULATION_STEP_SIZE,
        ),
        "upper_bound": from_yearly_multiplicative_drift(
            targets["multiplicative_drift_per_year"]["upper_bound"],
            time_step_days=SIMULATION_STEP_SIZE,
        ),
    }

    return result


# TODO: Broader test coverage


def test_housing_type_is_categorical(tracked_live_populations):
    for pop in tracked_live_populations:
        housing_type = pop["household_details.housing_type"]

        # Assert the dtype is correct and that there are no NaNs
        assert housing_type.dtype == pd.CategoricalDtype(categories=data_values.HOUSING_TYPES)
        assert not housing_type.isnull().any()


def test_relationship_is_categorical(tracked_live_populations):
    for pop in tracked_live_populations:
        relationship = pop["relationship_to_reference_person"]

        # Assert the dtype is correct and that there are no NaNs
        assert relationship.dtype == pd.CategoricalDtype(categories=metadata.RELATIONSHIPS)
        assert not relationship.isnull().any()


def test_all_households_have_reference_person(tracked_live_populations):
    for pop in tracked_live_populations:
        non_gq_household_ids = pop[
            ~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)
        ]["household_id"].unique()
        reference_person_household_ids = pop.loc[
            pop["relationship_to_reference_person"] == "Reference person", "household_id"
        ].values

        # Assert these two sets are identical
        assert non_gq_household_ids.size == reference_person_household_ids.size
        assert np.setxor1d(non_gq_household_ids, reference_person_household_ids).size == 0


def test_household_id_and_address_id_correspond(tracked_live_populations):
    for pop in tracked_live_populations:
        assert pop["household_id"].notnull().all()
        assert pop["household_details.address_id"].notnull().all()
        # 1-to-1 at any given point in time
        assert (
            pop.groupby("household_id")["household_details.address_id"].nunique() == 1
        ).all()
        assert (
            pop.groupby("household_details.address_id")["household_id"].nunique() == 1
        ).all()

    # Even over time, there is only 1 household_id for each address_id -- address_ids are not reused.
    all_time_pop = pd.concat(tracked_live_populations, ignore_index=True)
    assert (
        all_time_pop.groupby("household_details.address_id")["household_id"].nunique() == 1
    ).all()
    # Note, however, that the reverse is not true: a household_id can span multiple address_ids
    # (over multiple time steps) when the whole house moved as a unit between those time steps.


def test_new_reference_person_is_oldest_household_member(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        before_reference_person_idx = before.index[
            before["relationship_to_reference_person"] == "Reference person"
        ]
        after_reference_person_idx = after.index[
            (after["relationship_to_reference_person"] == "Reference person")
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
            pop["relationship_to_reference_person"] == "Reference person", "household_id"
        ]

        assert len(household_ids) == len(household_ids.unique())


def test_households_only_have_one_parter_or_spouse(tracked_live_populations):
    for pop in tracked_live_populations:
        household_ids = pop.loc[
            pop["relationship_to_reference_person"].isin(
                [
                    "Opposite-sex spouse",
                    "Opposite-sex unmarried partner",
                    "Same-sex spouse",
                    "Same-sex unmarried partner",
                ]
            ),
            "household_id",
        ]

        assert household_ids.is_unique


def test_housing_type_does_not_change(simulants_on_adjacent_timesteps):
    """Household types should not change for a given household"""
    for before, after in simulants_on_adjacent_timesteps:
        common_households = set(before["household_id"]).intersection(
            set(after["household_id"])
        )
        before = (
            before.loc[
                before["household_id"].isin(common_households),
                ["household_id", "household_details.housing_type"],
            ]
            .drop_duplicates()
            .sort_values("household_id")
            .set_index("household_id")
        )
        after = (
            after.loc[
                after["household_id"].isin(common_households),
                ["household_id", "household_details.housing_type"],
            ]
            .drop_duplicates()
            .sort_values("household_id")
            .set_index("household_id")
        )

        pd.testing.assert_frame_equal(before, after)
        assert not after.index.duplicated().any()


def test_state_population_proportions(
    populations, fuzzy_checker: FuzzyChecker, target_state_proportions
):
    # NOTE: We check these proportions on each timestep, but not across timesteps.
    # The reason for this is that households generally stay in the same state from
    # timestep to timestep, so multiple observations of the same house are not independent.
    for time_steps, pop in enumerate(populations):
        # No states in sim that were not in PUMS
        assert set(target_state_proportions["states"].keys()) >= set(
            pop["household_details.state_id"]
        )

        household_states = (
            # We want the proportion of the *households* in each state.
            # That's because it's only the location of *households* that are independent
            # of each other.
            # The GQ population is a whole other issue (we know we are way off in the
            # state distribution) which is ignored here.
            pop[pop["household_details.housing_type"] == "Household"]
            .groupby("household_id")["household_details.state_id"]
            .first()
        )

        for state_id, proportion in target_state_proportions["states"].items():
            # NOTE: Prior to fuzzy checking, we checked that all states were at least present in the population table.
            # The exact analog to this would be some complicated hypothesis about a coupon collector's partition with
            # uneven probabilities of different "coupons" (since states are different sizes).
            # To make things easier, we do a fuzzy check of the *proportion* of each state.
            # One downside to this approach is that it generates a lot of hypotheses.
            # An upside is that it is a more stringent check -- we not only have one household from each state,
            # but about the right *number*.
            fuzzy_checker.fuzzy_assert_proportion(
                name=f"State proportion for {state_id}",
                observed_numerator=(household_states == state_id).sum(),
                observed_denominator=len(household_states),
                # Relative size of states can change over time in the sim due to differential immigration, emigration,
                # mortality and fertility, and domestic migration
                target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                    proportion,
                    target_state_proportions["multiplicative_drift"]["lower_bound"],
                    target_state_proportions["multiplicative_drift"]["upper_bound"],
                    time_steps,
                ),
                name_additional=f"Time step {time_steps}",
            )


def test_pumas_states(populations):
    """Each unique address_id should have identical puma/state"""
    for pop in populations:
        assert (
            pop.groupby("household_details.address_id")[
                ["household_details.state_id", "household_details.puma"]
            ].nunique()
            == 1
        ).values.all()
