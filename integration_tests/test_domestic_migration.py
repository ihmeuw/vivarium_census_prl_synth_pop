import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.constants import data_values

# TODO: Broader test coverage


def test_individuals_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        individual_movers = before["household_id"] != after["household_id"]
        assert individual_movers.any()
        assert (
            before[individual_movers]["household_details.address_id"]
            != after[individual_movers]["household_details.address_id"]
        ).all()


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
            after[gq_movers]["relation_to_household_head"]
            .isin(["Institutionalized GQ pop", "Noninstitutionalized GQ pop"])
            .all()
        )


def get_households_with_stable_reference_person(after, before):
    reference_persons = before["relation_to_household_head"] == "Reference person"
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
            after.loc[mover_to_household_with_reference_person, "relation_to_household_head"]
            == "Other nonrelative"
        ).all()

        mover_to_household_without_reference_person = (
            non_reference_person_movers & ~in_household_with_reference_person
        )
        assert (
            after.loc[
                mover_to_household_without_reference_person, "relation_to_household_head"
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
            before[movers_with_reference_person]["relation_to_household_head"]
            == after[movers_with_reference_person]["relation_to_household_head"]
        ).all()
        assert (
            before[movers_with_reference_person]["household_details.housing_type"]
            == after[movers_with_reference_person]["household_details.housing_type"]
        ).all()

        # Address IDs moved to are new
        moved_households = before[household_movers]["household_id"].unique()
        new_addresses = after[household_movers]["household_details.address_id"]
        assert new_addresses.nunique() == len(moved_households)
        assert not before["household_details.address_id"].isin(new_addresses).any()

        # Never GQ households
        assert (
            before[household_movers]["household_details.housing_type"] == "Standard"
        ).all()


def test_only_living_people_change_households(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        movers = before["household_id"] != after["household_id"]
        assert after[movers]["tracked"].all()
        assert (after[movers]["alive"] == "alive").all()


@pytest.mark.parametrize(
    "unit_id_col, address_id_col",
    [
        ("employer_id", "business_details.employer_address_id"),
        ("household_id", "household_details.address_id"),
    ],
)
def test_unit_address_uniqueness(populations, unit_id_col, address_id_col):
    """Check that all units (households and businesses) have unique details and
    also that simulants of the same unit have the same details
    """
    address_cols = [unit_id_col, address_id_col]  # TODO: add state/puma
    for pop in populations:
        # Check that all simulants in the same unit have the same address
        assert (pop.groupby(unit_id_col)[address_cols].nunique() == 1).all().all()
        # Check that all units have unique addresses
        assert (pop.groupby(address_cols)[unit_id_col].nunique() == 1).all()

    # # TODO implement state/puma checks
    # # Ensure pumas map to correct state. This will be an imperfect test
    # # because pumas are only unique within states and so one puma can
    # # exist in multiple states. Here we only ensure no impossible pumas
    # # exist in each state
    # state_puma_map = get_state_puma_map(
    #     sim._data.artifact.load("population.households").reset_index()
    # )
    # for pop in populations:
    #     for (state, puma) in pop[["state", "puma"]].drop_duplicates().values:
    #         assert puma in state_puma_map[state]


@pytest.mark.parametrize(
    "unit_id_col, address_id_col",
    [
        ("employer_id", "business_details.employer_address_id"),
        ("household_id", "household_details.address_id"),
    ],
)
@pytest.mark.skip(reason="Wait for state/puma to be added (MIC-3728)")
def test_addresses_during_moves(simulants_on_adjacent_timesteps, unit_id_col, address_id_col):
    """Check that unit (household and business) address details change after a move."""
    other_address_cols = []  # TODO: add state/puma
    address_cols = [address_id_col] + other_address_cols
    for before, after in simulants_on_adjacent_timesteps:
        # get the unique unit (household or employer) dataset for before and after
        before_units = before.groupby(unit_id_col)[address_cols].first()
        after_units = after.groupby(unit_id_col)[address_cols].first()
        # reindex in case there are unique units to one of the datasets
        total_index = before_units.index.intersection(after_units.index)
        before_units = before_units.reindex(total_index)
        after_units = after_units.reindex(total_index)
        mask_moved_units = before_units[address_id_col] != after_units[address_id_col]

        # check that the number of moved units that have the same details is very low
        # NOTE: we cannot assert that all moved units have different details
        # because they are free to move within the same state or even puma
        assert (
            (
                before_units.loc[mask_moved_units, other_address_cols]
                == after_units.loc[mask_moved_units, other_address_cols]
            )
            .all(axis=1)
            .mean()
        ) < 0.001  # TODO: pick a smarter number?

        # address details do not change if address_id does not change
        pd.testing.assert_frame_equal(
            before_units[~mask_moved_units], after_units[~mask_moved_units]
        )

        # TODO when puma/state is implemented
        # # TODO Check that *most* are in a new state when we add all locations
        # # Some movers are in a new state.
        # # assert any(before.loc[mask_movers, "state"] != after.loc[mask_movers, "state"])

        # # Most movers are in a different puma
        # assert (
        #     before.loc[mask_movers, "puma"] != after.loc[mask_movers, "puma"]
        # ).sum() / mask_movers.sum() >= 0.95
