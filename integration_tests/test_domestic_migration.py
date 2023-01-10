from typing import Dict

import pandas as pd

from vivarium_census_prl_synth_pop.constants import data_values
from vivarium_census_prl_synth_pop.utilities import get_state_puma_map

# TODO: Broader test coverage


def test_individuals_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        individual_movers = before["household_id"] != after["household_id"]
        assert individual_movers.any()
        assert (
            before[individual_movers]["address_id"] != after[individual_movers]["address_id"]
        ).all()


def test_individuals_move_into_new_households(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        # NOTE: This set is not exactly the same as "new-household movers," as
        # implemented in the component, because it can in rare cases include
        # non-reference-person movers who join a household that was *just*
        # established by a new-household mover (within the same time step).
        movers_into_new_households = (
            (before["household_id"] != after["household_id"])
            & (after["housing_type"] == "Standard")
            & (~after["household_id"].isin(before["household_id"]))
        )
        assert movers_into_new_households.any()

        assert (
            after[movers_into_new_households]["relation_to_household_head"]
            # Handling the non-reference-person movers described above.
            .isin(["Reference person", "Other nonrelative"]).all()
        )

        # These are the true new-household movers, as that term is used in the
        # migration component: the movers who establish a new household.
        new_household_movers = movers_into_new_households & (
            after["relation_to_household_head"] == "Reference person"
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
        assert new_household_movers.sum() / movers_into_new_households.sum() > 0.95

        # Household IDs moved to are unique
        new_households = after[new_household_movers]["household_id"]
        assert new_households.nunique() == len(new_households)

        # Address IDs moved to are unique and new
        new_addresses = after[new_household_movers]["address_id"]
        assert new_addresses.nunique() == len(new_addresses)
        assert not new_addresses.isin(before["address_id"]).any()

        # TODO: Add state/puma tests


def test_individuals_move_into_group_quarters(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        gq_movers = (before["household_id"] != after["household_id"]) & (
            after["housing_type"] != "Standard"
        )
        assert gq_movers.any()
        assert (before[gq_movers]["housing_type"] == "Standard").any()
        assert after[gq_movers]["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP).all()
        assert (
            after[gq_movers]["relation_to_household_head"]
            .isin(["Institutionalized GQ pop", "Noninstitutionalized GQ pop"])
            .all()
        )


def test_individuals_move_into_existing_households(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        non_reference_person_movers = (
            (before["household_id"] != after["household_id"])
            & (after["housing_type"] == "Standard")
            & (after["household_id"].isin(before["household_id"]))
        )
        assert non_reference_person_movers.any()

        # They move in as nonrelative
        assert (
            after[non_reference_person_movers]["relation_to_household_head"]
            == "Other nonrelative"
        ).all()


def test_households_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        household_movers = (before["household_id"] == after["household_id"]) & (
            before["address_id"] != after["address_id"]
        )
        assert household_movers.any()

        # Household moves don't change household structure
        assert (
            before[household_movers]["relation_to_household_head"]
            == after[household_movers]["relation_to_household_head"]
        ).all()
        assert (
            before[household_movers]["housing_type"]
            == after[household_movers]["housing_type"]
        ).all()

        # Address IDs moved to are new
        moved_households = before[household_movers]["household_id"].unique()
        new_addresses = after[household_movers]["address_id"]
        assert new_addresses.nunique() == len(moved_households)
        assert not before["address_id"].isin(new_addresses).any()

        # GQ households never move
        assert (before[household_movers]["housing_type"] == "Standard").all()


def test_only_living_people_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        movers = before["address_id"] != after["address_id"]
        assert after[movers]["tracked"].all()
        assert (after[movers]["alive"] == "alive").all()


def test_address_ids_states_pumas(simulants_on_adjacent_timesteps, sim) -> None:
    """Check that address_ids are mapping correctly to states/pumas. Note that
    this test does not distinguish between the different types of moves (eg
    household moves, business moves, individual moves, etc)
    """
    state_puma_map = get_state_puma_map(
        sim._data.artifact.load("population.households").reset_index()
    )
    for before, after in simulants_on_adjacent_timesteps:
        mask_movers = before["address_id"] != after["address_id"]
        # TODO: Check that *most* are in a new state when we add all locations
        # Some movers are in a new state.
        assert any(before.loc[mask_movers, "state"] != after.loc[mask_movers, "state"])

        # Most movers are in a different puma
        assert (
            before.loc[mask_movers, "puma"] != after.loc[mask_movers, "puma"]
        ).sum() / mask_movers.sum() >= 0.95

        # puma/state do not change if address_id does not change
        # (except for gq locations - those can have the same address_id and
        # different states/pumas)
        same_addresses = before[before["address_id"] == after["address_id"]].index
        gq_simulants = after[after["housing_type"] != "Standard"].index
        same_non_gq_addresses = same_addresses.difference(gq_simulants)

        pd.testing.assert_frame_equal(
            before.loc[same_non_gq_addresses, ["state", "puma"]],
            after.loc[same_non_gq_addresses, ["state", "puma"]],
        )

        # Ensure pumas map to correct state. This will be an imperfect test
        # because pumas are only unique within states and so one puma can
        # exist in multiple states. Here we only ensure no impossible pumas
        # exist in each state
        for (state, puma) in before[["state", "puma"]].drop_duplicates().values:
            assert puma in state_puma_map[state]
        for (state, puma) in after[["state", "puma"]].drop_duplicates().values:
            assert puma in state_puma_map[state]

        # Each unique address_id has the same puma/state (except for GQ)
        assert (
            (
                before[before["housing_type"] == "Standard"]
                .groupby("address_id")["state", "puma"]
                .nunique()
                == 1
            )
            .all()
            .all()
        )
        assert (
            (
                after[after["housing_type"] == "Standard"]
                .groupby("address_id")["state", "puma"]
                .nunique()
                == 1
            )
            .all()
            .all()
        )
