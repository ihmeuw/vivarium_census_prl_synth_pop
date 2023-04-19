import numpy as np
import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.constants import data_values, metadata, paths


def test_individuals_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        individual_movers = before["household_id"] != after["household_id"]
        assert individual_movers.any()
        assert (
            before[individual_movers]["household_details.address_id"]
            != after[individual_movers]["household_details.address_id"]
        ).all()
        check_po_box_collisions(after, before, individual_movers)


def test_individuals_move_into_new_households(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        # NOTE: This set is not exactly the same as "new-household movers," as
        # implemented in the component, because it can in rare cases include
        # non-reference-person movers who join a household that was *just*
        # established by a new-household mover (within the same time step).
        movers_into_new_households = (
            (before["household_id"] != after["household_id"])
            & (after["household_details.housing_type"] == "Standard")
            & (~after["household_id"].isin(before["household_id"]))
        )
        assert movers_into_new_households.any()

        assert (
            after[movers_into_new_households]["relation_to_reference_person"]
            # Handling the non-reference-person movers described above.
            .isin(["Reference person", "Other nonrelative"]).all()
        )

        # These are the true new-household movers, as that term is used in the
        # migration component: the movers who establish a new household.
        new_household_movers = movers_into_new_households & (
            after["relation_to_reference_person"] == "Reference person"
        )
        assert new_household_movers.any()
        # There is exactly one new-household mover for each new household.
        assert (
            new_household_movers[movers_into_new_households]
            .groupby(after[movers_into_new_households]["household_id"])
            .sum()
            == 1
        ).all()
        # These should be the vast majority, since non-reference-person moves to
        # new households will be quite rare.
        # NOTE: This is sensitive to small population size, eg running 20k simulants
        # was causing it to break w/ a ratio 19/20 = 0.95
        assert new_household_movers.sum() / movers_into_new_households.sum() >= 0.95

        # Household IDs moved to are unique
        new_households = after[new_household_movers]["household_id"]
        assert new_households.nunique() == len(new_households)

        # Address IDs moved to are unique and new
        new_addresses = after[new_household_movers]["household_details.address_id"]
        assert new_addresses.nunique() == len(new_addresses)
        assert not new_addresses.isin(before["household_details.address_id"]).any()


def test_individuals_move_into_group_quarters(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        gq_movers = (before["household_id"] != after["household_id"]) & (
            after["household_details.housing_type"] != "Standard"
        )
        assert gq_movers.any()
        assert (before[gq_movers]["household_details.housing_type"] == "Standard").any()
        assert after[gq_movers]["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP).all()
        assert (
            after[gq_movers]["relation_to_reference_person"]
            .isin(["Institutionalized GQ pop", "Noninstitutionalized GQ pop"])
            .all()
        )


def get_households_with_stable_reference_person(after, before):
    reference_persons = before["relation_to_reference_person"] == "Reference person"
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


def test_individual_movers_have_correct_relationship(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        non_reference_person_movers = (
            (before["household_id"] != after["household_id"])
            & (after["household_details.housing_type"] == "Standard")
            & (after["household_id"].isin(before["household_id"]))
        )
        assert non_reference_person_movers.any()

        # They move in as nonrelative, which doesn't change unless the reference person
        # of their new household also moved or died
        in_household_with_reference_person = get_households_with_stable_reference_person(
            after, before
        )

        mover_to_household_with_reference_person = (
            non_reference_person_movers & in_household_with_reference_person
        )
        assert (
            after.loc[mover_to_household_with_reference_person, "relation_to_reference_person"]
            == "Other nonrelative"
        ).all()

        mover_to_household_without_reference_person = (
            non_reference_person_movers & ~in_household_with_reference_person
        )
        assert (
            after.loc[
                mover_to_household_without_reference_person, "relation_to_reference_person"
            ].isin(["Other nonrelative", "Reference person", "Roommate"])
        ).all()


def test_households_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        household_movers = (before["household_id"] == after["household_id"]) & (
            before["household_details.address_id"] != after["household_details.address_id"]
        )
        assert household_movers.any()

        # Household moves don't change household structure unless the reference person left
        in_household_with_reference_person = get_households_with_stable_reference_person(
            after, before
        )
        movers_with_reference_person = household_movers & in_household_with_reference_person
        assert (
            before[movers_with_reference_person]["relation_to_reference_person"]
            == after[movers_with_reference_person]["relation_to_reference_person"]
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
            before[household_movers]["household_details.housing_type"] == "Standard"
        ).all()

        check_po_box_collisions(after, before, household_movers)


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
    simulants_on_adjacent_timesteps, unit_id_col, address_id_col, state_id_col, puma_col
):
    """Check that unit (household and business) address details change after a move."""
    address_cols = [address_id_col, state_id_col, puma_col]
    us_locs = metadata.UNITED_STATES_LOCATIONS
    states_pumas = pd.read_csv(paths.PUMA_TO_ZIP_DATA_PATH)
    if us_locs:  # Only include states in us_locs
        states_pumas = states_pumas[states_pumas["state"].isin(us_locs)]

    total_num_moved = 0
    for before, after in simulants_on_adjacent_timesteps:
        # get the unique unit (household or employer) dataset for before and after
        before_units = before.groupby(unit_id_col)[address_cols].first()
        after_units = after.groupby(unit_id_col)[address_cols].first()
        # reindex in case there are unique units to one of the datasets
        total_index = before_units.index.intersection(after_units.index)
        before_units = before_units.reindex(total_index)
        after_units = after_units.reindex(total_index)
        mask_moved_units = before_units[address_id_col] != after_units[address_id_col]
        if not mask_moved_units.any():
            continue

        total_num_moved += mask_moved_units.sum()
        # Check that the number of moved units that have the same details is very low
        # NOTE: we cannot assert that all moved units have different details
        # because they are free to move within the same state or even PUMA
        assert (
            (
                before_units.loc[mask_moved_units, address_cols]
                == after_units.loc[mask_moved_units, address_cols]
            )
            .all(axis=1)
            .mean()
        ) < 0.001  # TODO: pick a smarter number?
        # address details do not change if address_id does not change
        pd.testing.assert_frame_equal(
            before_units[~mask_moved_units], after_units[~mask_moved_units]
        )

        # Check that most movers are in a new state-PUMA combination
        if mask_moved_units.sum() > 1:
            assert (
                before_units.loc[mask_moved_units, [state_id_col, puma_col]]
                == after_units.loc[mask_moved_units, [state_id_col, puma_col]]
            ).all(axis=1).mean() < 100 * 1 / len(states_pumas)

    # Check that at least some units moved during the sim
    assert total_num_moved > 0


def test_po_box(tracked_live_populations):
    """Tests the prevalence of PO Boxes."""
    for pop in tracked_live_populations:
        # Check that actual proportion of households without PO Boxes (i.e., physical
        # address is the same as mailing) is close to the expected proportion
        assert np.isclose(
            pop[pop["household_details.po_box"] == data_values.NO_PO_BOX][
                "household_id"
            ].nunique()
            / pop["household_id"].nunique(),
            data_values.PROBABILITY_OF_SAME_MAILING_PHYSICAL_ADDRESS,
            rtol=0.01,
        )

        # Check that PO Boxes are within the min and max defined in constants
        assert (
            (pop["household_details.po_box"] == data_values.NO_PO_BOX)
            | (
                (pop["household_details.po_box"] <= data_values.MAX_PO_BOX)
                & (pop["household_details.po_box"] >= data_values.MIN_PO_BOX)
            )
        ).all()
