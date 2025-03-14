import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from vivarium.framework.utilities import from_yearly

from vivarium_census_prl_synth_pop.constants import data_values, metadata, paths

from .conftest import (
    FuzzyChecker,
    from_yearly_multiplicative_drift,
    multiplicative_drifts_to_bounds_at_timestep,
    multiplicative_drifts_to_bounds_through_timestep,
)


@pytest.fixture(scope="module")
def target_migration_rates(sim):
    assert (
        metadata.UNITED_STATES_LOCATIONS == []
    ), "Integration tests do not support subsets by US state"

    with open(
        Path(os.path.dirname(__file__)) / "v_and_v_inputs/domestic_migration.yaml"
    ) as f:
        targets = yaml.safe_load(f)

    targets["individual_migration_rate_per_year"]["total"] = sum(
        [
            v
            for k, v in targets["individual_migration_rate_per_year"].items()
            if k != "non_reference_person_into_household_created_in_last_year"
        ]
    )

    targets["individual_migration_rate_per_year"][
        "non_reference_person_into_household_created_in_last_timestep"
    ] = from_yearly(
        targets["individual_migration_rate_per_year"][
            "non_reference_person_into_household_created_in_last_year"
        ],
        pd.Timedelta(days=sim.configuration.time.step_size),
    )

    targets["household"] = from_yearly(
        targets["household_migration_rate_per_year"],
        pd.Timedelta(days=sim.configuration.time.step_size),
    )

    targets["individual"] = {}
    for k in targets["individual_migration_rate_per_year"].keys():
        targets["individual"][k] = from_yearly(
            targets["individual_migration_rate_per_year"][k],
            pd.Timedelta(days=sim.configuration.time.step_size),
        )

    targets["multiplicative_drift"] = {
        "lower_bound": from_yearly_multiplicative_drift(
            targets["multiplicative_drift_per_year"]["lower_bound"],
            time_step_days=sim.configuration.time.step_size,
        ),
        "upper_bound": from_yearly_multiplicative_drift(
            targets["multiplicative_drift_per_year"]["upper_bound"],
            time_step_days=sim.configuration.time.step_size,
        ),
    }

    return targets


def test_individuals_move(
    simulants_on_adjacent_timesteps, fuzzy_checker: FuzzyChecker, target_migration_rates
):
    all_time_individual_movers = []

    for time_steps, (before, after) in enumerate(simulants_on_adjacent_timesteps):
        individual_movers = before["household_id"] != after["household_id"]
        fuzzy_checker.fuzzy_assert_proportion(
            name="Individual migration rate",
            observed_numerator=individual_movers.sum(),
            observed_denominator=len(individual_movers),
            target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                target_migration_rates["individual"]["total"],
                target_migration_rates["multiplicative_drift"]["lower_bound"],
                target_migration_rates["multiplicative_drift"]["upper_bound"],
                time_steps,
            ),
            name_additional=f"Time step {time_steps}",
        )
        assert (
            before[individual_movers]["household_details.address_id"]
            != after[individual_movers]["household_details.address_id"]
        ).all()
        check_po_box_collisions(after, before, individual_movers)
        all_time_individual_movers.append(individual_movers)

    all_time_individual_movers = pd.concat(all_time_individual_movers, ignore_index=True)
    fuzzy_checker.fuzzy_assert_proportion(
        name="Individual migration rate",
        observed_numerator=all_time_individual_movers.sum(),
        observed_denominator=len(all_time_individual_movers),
        target_proportion=multiplicative_drifts_to_bounds_through_timestep(
            target_migration_rates["individual"]["total"],
            target_migration_rates["multiplicative_drift"]["lower_bound"],
            target_migration_rates["multiplicative_drift"]["upper_bound"],
            len(simulants_on_adjacent_timesteps),
        ),
        name_additional="All time steps",
    )


def test_individuals_move_into_new_households(
    simulants_on_adjacent_timesteps, fuzzy_checker: FuzzyChecker, target_migration_rates
):
    all_time_joining_recently_created_households = []
    all_time_new_household_movers = []

    for time_step, (before, after) in enumerate(simulants_on_adjacent_timesteps):
        # NOTE: This set is not exactly the same as "new-household movers," as
        # implemented in the component, because it can in rare cases include
        # non-reference-person movers who join a household that was *just*
        # established by a new-household mover (within the same time step).
        movers_into_new_households = (
            (before["household_id"] != after["household_id"])
            & (after["household_details.housing_type"] == "Household")
            & (~after["household_id"].isin(before["household_id"]))
        )

        assert (
            after[movers_into_new_households]["relationship_to_reference_person"]
            # Handling the non-reference-person movers described above.
            .isin(["Reference person", "Other nonrelative"]).all()
        )

        # These are the true new-household movers, as that term is used in the
        # migration component: the movers who establish a new household.
        new_household_movers = movers_into_new_households & (
            after["relationship_to_reference_person"] == "Reference person"
        )
        joining_recently_created_households = movers_into_new_households & (
            ~new_household_movers
        )
        fuzzy_checker.fuzzy_assert_proportion(
            name="Domestic migration joining recently created households",
            observed_numerator=joining_recently_created_households.sum(),
            observed_denominator=len(joining_recently_created_households),
            target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                target_migration_rates["individual"][
                    "non_reference_person_into_household_created_in_last_timestep"
                ],
                target_migration_rates["multiplicative_drift"]["lower_bound"],
                target_migration_rates["multiplicative_drift"]["upper_bound"],
                time_step,
            ),
            name_additional=f"Time step {time_step}",
        )
        all_time_joining_recently_created_households.append(
            joining_recently_created_households
        )
        fuzzy_checker.fuzzy_assert_proportion(
            name="Domestic migration creating new households",
            observed_numerator=new_household_movers.sum(),
            observed_denominator=len(new_household_movers),
            target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                target_migration_rates["individual"]["new_household"],
                target_migration_rates["multiplicative_drift"]["lower_bound"],
                target_migration_rates["multiplicative_drift"]["upper_bound"],
                time_step,
            ),
            name_additional=f"Time step {time_step}",
        )
        all_time_new_household_movers.append(new_household_movers)
        # There is exactly one new-household mover for each new household.
        assert (
            new_household_movers[movers_into_new_households]
            .groupby(after[movers_into_new_households]["household_id"])
            .sum()
            == 1
        ).all()

        # Household IDs moved to are unique
        new_households = after[new_household_movers]["household_id"]
        assert new_households.nunique() == len(new_households)

        # Address IDs moved to are unique and new
        new_addresses = after[new_household_movers]["household_details.address_id"]
        assert new_addresses.nunique() == len(new_addresses)
        assert not new_addresses.isin(before["household_details.address_id"]).any()

    all_time_joining_recently_created_households = pd.concat(
        all_time_joining_recently_created_households, ignore_index=True
    )
    all_time_new_household_movers = pd.concat(
        all_time_new_household_movers, ignore_index=True
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="Domestic migration joining recently created households",
        observed_numerator=all_time_joining_recently_created_households.sum(),
        observed_denominator=len(all_time_joining_recently_created_households),
        target_proportion=multiplicative_drifts_to_bounds_through_timestep(
            target_migration_rates["individual"][
                "non_reference_person_into_household_created_in_last_timestep"
            ],
            target_migration_rates["multiplicative_drift"]["lower_bound"],
            target_migration_rates["multiplicative_drift"]["upper_bound"],
            len(simulants_on_adjacent_timesteps),
        ),
        name_additional="All time steps",
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="Domestic migration creating new households",
        observed_numerator=all_time_new_household_movers.sum(),
        observed_denominator=len(all_time_new_household_movers),
        target_proportion=multiplicative_drifts_to_bounds_through_timestep(
            target_migration_rates["individual"]["new_household"],
            target_migration_rates["multiplicative_drift"]["lower_bound"],
            target_migration_rates["multiplicative_drift"]["upper_bound"],
            len(simulants_on_adjacent_timesteps),
        ),
        name_additional="All time steps",
    )


def test_individuals_move_into_group_quarters(
    simulants_on_adjacent_timesteps, fuzzy_checker: FuzzyChecker, target_migration_rates
):
    all_time_gq_movers = []
    all_time_into_gq_from_gq = []

    for time_step, (before, after) in enumerate(simulants_on_adjacent_timesteps):
        gq_movers = (before["household_id"] != after["household_id"]) & (
            after["household_details.housing_type"] != "Household"
        )
        fuzzy_checker.fuzzy_assert_proportion(
            name="Domestic migration into GQ",
            observed_numerator=gq_movers.sum(),
            observed_denominator=len(gq_movers),
            target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                target_migration_rates["individual"]["gq_person"],
                target_migration_rates["multiplicative_drift"]["lower_bound"],
                target_migration_rates["multiplicative_drift"]["upper_bound"],
                time_step,
            ),
            name_additional=f"Time step {time_step}",
        )
        all_time_gq_movers.append(gq_movers)
        into_gq_from_gq = before[gq_movers]["household_details.housing_type"] != "Household"
        fuzzy_checker.fuzzy_assert_proportion(
            name="Domestic migration into GQ from GQ",
            observed_numerator=into_gq_from_gq.sum(),
            observed_denominator=len(into_gq_from_gq),
            target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                target_migration_rates["gq_into_gq_proportion_of_gq_movers"],
                target_migration_rates["multiplicative_drift"]["lower_bound"],
                target_migration_rates["multiplicative_drift"]["upper_bound"],
                time_step,
            ),
            name_additional=f"Time step {time_step}",
        )
        all_time_into_gq_from_gq.append(into_gq_from_gq)
        assert after[gq_movers]["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP).all()
        assert (
            after[gq_movers]["relationship_to_reference_person"]
            .isin(
                [
                    "Institutionalized group quarters population",
                    "Noninstitutionalized group quarters population",
                ]
            )
            .all()
        )

    all_time_gq_movers = pd.concat(all_time_gq_movers, ignore_index=True)
    all_time_into_gq_from_gq = pd.concat(all_time_into_gq_from_gq, ignore_index=True)

    fuzzy_checker.fuzzy_assert_proportion(
        name="Domestic migration into GQ",
        observed_numerator=all_time_gq_movers.sum(),
        observed_denominator=len(all_time_gq_movers),
        target_proportion=multiplicative_drifts_to_bounds_through_timestep(
            target_migration_rates["individual"]["gq_person"],
            target_migration_rates["multiplicative_drift"]["lower_bound"],
            target_migration_rates["multiplicative_drift"]["upper_bound"],
            len(simulants_on_adjacent_timesteps),
        ),
        name_additional="All time steps",
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="Domestic migration into GQ from GQ",
        observed_numerator=all_time_into_gq_from_gq.sum(),
        observed_denominator=len(all_time_into_gq_from_gq),
        target_proportion=multiplicative_drifts_to_bounds_through_timestep(
            target_migration_rates["gq_into_gq_proportion_of_gq_movers"],
            target_migration_rates["multiplicative_drift"]["lower_bound"],
            target_migration_rates["multiplicative_drift"]["upper_bound"],
            len(simulants_on_adjacent_timesteps),
        ),
        name_additional="All time steps",
    )


def get_households_with_stable_reference_person(after, before):
    reference_persons = before["relationship_to_reference_person"] == "Reference person"
    movers = before["household_id"] != after["household_id"]
    deaths = before["alive"] != after["alive"]
    emigrants = before["in_united_states"] & ~after["in_united_states"]
    newly_untracked = before["tracked"] & ~after["tracked"]
    changed_reference_people = reference_persons & (
        movers | deaths | emigrants | newly_untracked
    )
    in_household_with_reference_person = ~after.loc[movers, "household_id"].isin(
        before[changed_reference_people]["household_id"]
    )
    return in_household_with_reference_person


def test_individual_movers_have_correct_relationship(
    simulants_on_adjacent_timesteps, fuzzy_checker: FuzzyChecker, target_migration_rates
):
    all_time_non_reference_person_movers = []

    for time_step, (before, after) in enumerate(simulants_on_adjacent_timesteps):
        non_reference_person_movers = (
            (before["household_id"] != after["household_id"])
            & (after["household_details.housing_type"] == "Household")
            & (after["household_id"].isin(before["household_id"]))
        )
        fuzzy_checker.fuzzy_assert_proportion(
            name="Domestic migration into existing households",
            observed_numerator=non_reference_person_movers.sum(),
            observed_denominator=len(non_reference_person_movers),
            target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                target_migration_rates["individual"]["non_reference_person"],
                target_migration_rates["multiplicative_drift"]["lower_bound"],
                target_migration_rates["multiplicative_drift"]["upper_bound"],
                time_step,
            ),
            name_additional=f"Time step {time_step}",
        )
        all_time_non_reference_person_movers.append(non_reference_person_movers)

        # They move in as nonrelative, which doesn't change unless the reference person
        # of their new household also moved or died
        in_household_with_reference_person = get_households_with_stable_reference_person(
            after, before
        )

        mover_to_household_with_reference_person = (
            non_reference_person_movers & in_household_with_reference_person
        )
        assert (
            after.loc[
                mover_to_household_with_reference_person, "relationship_to_reference_person"
            ]
            == "Other nonrelative"
        ).all()

        mover_to_household_without_reference_person = (
            non_reference_person_movers & ~in_household_with_reference_person
        )
        assert (
            after.loc[
                mover_to_household_without_reference_person,
                "relationship_to_reference_person",
            ].isin(["Other nonrelative", "Reference person", "Roommate or housemate"])
        ).all()

    all_time_non_reference_person_movers = pd.concat(
        all_time_non_reference_person_movers, ignore_index=True
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="Domestic migration into existing households",
        observed_numerator=all_time_non_reference_person_movers.sum(),
        observed_denominator=len(all_time_non_reference_person_movers),
        target_proportion=multiplicative_drifts_to_bounds_through_timestep(
            target_migration_rates["individual"]["non_reference_person"],
            target_migration_rates["multiplicative_drift"]["lower_bound"],
            target_migration_rates["multiplicative_drift"]["upper_bound"],
            len(simulants_on_adjacent_timesteps),
        ),
        name_additional="All time steps",
    )


def test_households_move(
    simulants_on_adjacent_timesteps, fuzzy_checker: FuzzyChecker, target_migration_rates
):
    all_time_household_moves = []

    for time_step, (before, after) in enumerate(simulants_on_adjacent_timesteps):
        household_movers = (before["household_id"] == after["household_id"]) & (
            before["household_details.address_id"] != after["household_details.address_id"]
        )
        household_moves = household_movers.groupby(before["household_id"]).first()
        fuzzy_checker.fuzzy_assert_proportion(
            name="Domestic migration of households",
            observed_numerator=household_moves.sum(),
            observed_denominator=len(household_moves),
            target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                target_migration_rates["household"],
                target_migration_rates["multiplicative_drift"]["lower_bound"],
                target_migration_rates["multiplicative_drift"]["upper_bound"],
                time_step,
            ),
            name_additional=f"Time step {time_step}",
        )
        all_time_household_moves.append(household_moves)

        # Household moves don't change household structure unless the reference person left
        in_household_with_reference_person = get_households_with_stable_reference_person(
            after, before
        )
        movers_with_reference_person = household_movers & in_household_with_reference_person
        assert (
            before[movers_with_reference_person]["relationship_to_reference_person"]
            == after[movers_with_reference_person]["relationship_to_reference_person"]
        ).all()
        assert (
            before[movers_with_reference_person]["household_details.housing_type"]
            == after[movers_with_reference_person]["household_details.housing_type"]
        ).all()

        # Address IDs moved to are new
        moved_households = before[household_movers]["household_id"]
        new_addresses = after[household_movers]["household_details.address_id"]
        assert new_addresses.nunique() == moved_households.nunique()
        assert not before["household_details.address_id"].isin(new_addresses).any()

        # Never GQ households
        assert (
            before[household_movers]["household_details.housing_type"] == "Household"
        ).all()

        check_po_box_collisions(after, before, household_movers)

    all_time_household_moves = pd.concat(all_time_household_moves, ignore_index=True)
    fuzzy_checker.fuzzy_assert_proportion(
        name="Domestic migration of households",
        observed_numerator=all_time_household_moves.sum(),
        observed_denominator=len(all_time_household_moves),
        target_proportion=multiplicative_drifts_to_bounds_at_timestep(
            target_migration_rates["household"],
            target_migration_rates["multiplicative_drift"]["lower_bound"],
            target_migration_rates["multiplicative_drift"]["upper_bound"],
            time_step,
        ),
        name_additional="All time steps",
    )


def check_po_box_collisions(after, before, movers):
    """Assert that the number of PO Box collisions are miniscule. Movers
    assumed to have address_id changes."""
    po_box_movers = (
        (before["household_details.po_box"] != data_values.NO_PO_BOX)
        | (after["household_details.po_box"] != data_values.NO_PO_BOX)
    ) & movers
    assert (
        before[po_box_movers]["household_details.po_box"]
        == after[po_box_movers]["household_details.po_box"]
    ).sum() <= int(po_box_movers.sum() * 0.05)


def test_only_living_people_change_households(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        movers = before["household_id"] != after["household_id"]
        assert after[movers]["tracked"].all()
        assert (after[movers]["alive"] == "alive").all()


@pytest.mark.parametrize(
    "unit_id_col, address_id_col, state_id_col, puma_col",
    [
        (
            "employer_id",
            "business_details.employer_address_id",
            "business_details.employer_state_id",
            "business_details.employer_puma",
        ),
        (
            "household_id",
            "household_details.address_id",
            "household_details.state_id",
            "household_details.puma",
        ),
        (
            "household_id",
            "household_details.po_box",
            "household_details.state_id",
            "household_details.puma",
        ),
    ],
)
def test_unit_members_share_address(
    populations, unit_id_col, address_id_col, state_id_col, puma_col
):
    """Check that all simulants in the same unit have the same address"""
    address_cols = [unit_id_col, address_id_col, state_id_col, puma_col]
    for pop in populations:
        assert (pop.groupby(unit_id_col)[address_cols].nunique() == 1).all().all()


@pytest.mark.parametrize(
    "unit_id_col, address_id_col, state_id_col, puma_col",
    [
        (
            "employer_id",
            "business_details.employer_address_id",
            "business_details.employer_state_id",
            "business_details.employer_puma",
        ),
        (
            "household_id",
            "household_details.address_id",
            "household_details.state_id",
            "household_details.puma",
        ),
        (
            "household_id",
            "household_details.po_box",
            "household_details.state_id",
            "household_details.puma",
        ),
    ],
)
def test_unit_address_uniqueness(
    populations, unit_id_col, address_id_col, state_id_col, puma_col
):
    """Check that all units (households or businesses) have unique addresses
    (unless it's a PO Box, which has relaxed requirements)
    """
    address_cols = [unit_id_col, address_id_col, state_id_col, puma_col]
    for pop in populations:
        if address_id_col != "household_details.po_box":
            assert (pop.groupby(address_cols)[unit_id_col].nunique() == 1).all()


@pytest.mark.parametrize(
    "state_id_col, puma_col",
    [
        ("household_details.state_id", "household_details.puma"),
        ("business_details.employer_state_id", "business_details.employer_puma"),
    ],
)
def test_pumas_exist_in_states(populations, state_id_col, puma_col):
    """Check that PUMAs map to correct state.

    NOTE: This is an imperfect test because PUMAs are only unique within states
    and so one PUMA can exist in multiple states. Here we only ensure no
    impossible PUMAs exist in each state
    """
    state_puma_map = (
        pd.read_csv(paths.PUMA_TO_ZIP_DATA_PATH).groupby("state")["puma"].unique()
    )
    for pop in populations:
        for state_id, puma in pop[[state_id_col, puma_col]].drop_duplicates().values:
            assert puma in state_puma_map[state_id]


@pytest.mark.parametrize(
    "unit_id_col, address_id_col, state_id_col, puma_col",
    [
        (
            "employer_id",
            "business_details.employer_address_id",
            "business_details.employer_state_id",
            "business_details.employer_puma",
        ),
        (
            "household_id",
            "household_details.address_id",
            "household_details.state_id",
            "household_details.puma",
        ),
    ],
)
def test_addresses_during_moves(
    simulants_on_adjacent_timesteps,
    unit_id_col,
    address_id_col,
    state_id_col,
    puma_col,
    sim,
    target_migration_rates,
    fuzzy_checker: FuzzyChecker,
):
    """Check that unit (household and business) address details change after a move."""
    address_cols = [address_id_col, state_id_col, puma_col]

    state_puma_options = pd.read_csv(paths.PUMA_TO_ZIP_DATA_PATH)[
        ["state", "puma"]
    ].drop_duplicates()

    us_locs = metadata.UNITED_STATES_LOCATIONS
    if us_locs:
        # Subset to only states that exist in the sim
        state_puma_options = state_puma_options[state_puma_options["state"].isin(us_locs)]

    # With equal likelihood of moving from and to any given PUMA, this is the probability
    # that source and destination state are the same
    true_proportion_moving_within_state = (
        (state_puma_options.groupby("state").size() / len(state_puma_options)) ** 2
    ).sum()

    all_time_moved = []
    for time_step, (before, after) in enumerate(simulants_on_adjacent_timesteps):
        # get the unique unit (household or employer) dataset for before and after
        before_units = before.groupby(unit_id_col)[address_cols].first()
        after_units = after.groupby(unit_id_col)[address_cols].first()
        # reindex in case there are unique units to one of the datasets
        total_index = before_units.index.intersection(after_units.index)
        before_units = before_units.reindex(total_index)
        after_units = after_units.reindex(total_index)
        mask_moved_units = before_units[address_id_col] != after_units[address_id_col]

        if unit_id_col == "employer_id":
            fuzzy_checker.fuzzy_assert_proportion(
                name="Domestic migration of businesses",
                observed_numerator=mask_moved_units.sum(),
                observed_denominator=len(mask_moved_units),
                target_proportion=from_yearly(
                    data_values.BUSINESS_MOVE_RATE_YEARLY,
                    pd.Timedelta(days=sim.configuration.time.step_size),
                ),
                name_additional=f"Time step {time_step}",
            )
        else:
            fuzzy_checker.fuzzy_assert_proportion(
                name="Domestic migration of households",
                observed_numerator=mask_moved_units.sum(),
                observed_denominator=len(mask_moved_units),
                target_proportion=multiplicative_drifts_to_bounds_at_timestep(
                    target_migration_rates["household"],
                    target_migration_rates["multiplicative_drift"]["lower_bound"],
                    target_migration_rates["multiplicative_drift"]["upper_bound"],
                    time_step,
                ),
                name_additional=f"Time step {time_step}",
            )

        all_time_moved.append(mask_moved_units)

        # Check that address details do not change if address_id does not change
        pd.testing.assert_frame_equal(
            before_units[~mask_moved_units], after_units[~mask_moved_units]
        )

        # Check that most movers are in a new state
        new_state_movers = (
            before_units.loc[mask_moved_units, state_id_col]
            != after_units.loc[mask_moved_units, state_id_col]
        )
        fuzzy_checker.fuzzy_assert_proportion(
            name=f"Domestic migration: proportion {unit_id_col} moving into a new state",
            observed_numerator=new_state_movers.sum(),
            observed_denominator=len(new_state_movers),
            target_proportion=(1 - true_proportion_moving_within_state),
            name_additional=f"Time step {time_step}",
        )

        # Check that the vast majority of movers are in a different PUMA
        new_puma_movers = (
            before_units.loc[mask_moved_units, [state_id_col, puma_col]]
            == after_units.loc[mask_moved_units, [state_id_col, puma_col]]
        ).all(axis=1)
        fuzzy_checker.fuzzy_assert_proportion(
            name=f"Domestic migration: proportion {unit_id_col} moving into a new PUMA",
            observed_numerator=new_puma_movers.sum(),
            observed_denominator=len(new_puma_movers),
            target_proportion=(1 / len(state_puma_options)),
            name_additional=f"Time step {time_step}",
        )

    # Check that at least some units moved during the sim
    all_time_moved = pd.concat(all_time_moved, ignore_index=True)
    if unit_id_col == "employer_id":
        fuzzy_checker.fuzzy_assert_proportion(
            name="Domestic migration of businesses",
            observed_numerator=all_time_moved.sum(),
            observed_denominator=len(all_time_moved),
            target_proportion=from_yearly(
                data_values.BUSINESS_MOVE_RATE_YEARLY,
                pd.Timedelta(days=sim.configuration.time.step_size),
            ),
            name_additional="All time steps",
        )
    else:
        fuzzy_checker.fuzzy_assert_proportion(
            name="Domestic migration of households",
            observed_numerator=all_time_moved.sum(),
            observed_denominator=len(all_time_moved),
            target_proportion=multiplicative_drifts_to_bounds_through_timestep(
                target_migration_rates["household"],
                target_migration_rates["multiplicative_drift"]["lower_bound"],
                target_migration_rates["multiplicative_drift"]["upper_bound"],
                time_step,
            ),
            name_additional="All time steps",
        )


def test_po_box(tracked_live_populations, fuzzy_checker: FuzzyChecker):
    """Tests the prevalence of PO Boxes."""
    all_time_no_po_box = []

    for time_step, pop in enumerate(tracked_live_populations):
        po_box_values = (
            pop[pop["household_details.housing_type"] == "Household"]
            .groupby("household_details.address_id")["household_details.po_box"]
            .first()
        )
        no_po_box = po_box_values == data_values.NO_PO_BOX
        fuzzy_checker.fuzzy_assert_proportion(
            name="Proportion without PO box",
            observed_numerator=no_po_box.sum(),
            observed_denominator=len(no_po_box),
            target_proportion=data_values.PROBABILITY_OF_SAME_MAILING_PHYSICAL_ADDRESS,
            name_additional=f"Time step {time_step}",
        )
        all_time_no_po_box.append(no_po_box.reset_index())

        # Check that PO Boxes are within the min and max defined in constants
        assert (
            (po_box_values == data_values.NO_PO_BOX)
            | (
                (po_box_values <= data_values.MAX_PO_BOX)
                & (po_box_values >= data_values.MIN_PO_BOX)
            )
        ).all()

    all_time_no_po_box = pd.concat(all_time_no_po_box)
    # An individual address_id either has a PO box or doesn't.
    assert (
        all_time_no_po_box.groupby("household_details.address_id")[
            "household_details.po_box"
        ].nunique()
        == 1
    ).all()
    # We only consider each unique address ID once, since repeated observations
    # of the same address ID are not independent.
    all_time_no_po_box = all_time_no_po_box.groupby("household_details.address_id")[
        "household_details.po_box"
    ].first()

    fuzzy_checker.fuzzy_assert_proportion(
        name="Proportion without PO box",
        observed_numerator=all_time_no_po_box.sum(),
        observed_denominator=len(all_time_no_po_box),
        target_proportion=data_values.PROBABILITY_OF_SAME_MAILING_PHYSICAL_ADDRESS,
        name_additional="All time steps",
    )
