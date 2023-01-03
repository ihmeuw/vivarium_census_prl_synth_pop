import pytest

from vivarium_census_prl_synth_pop.constants import data_values

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
        new_household_movers = (
            (before["household_id"] != after["household_id"])
            & (after["housing_type"] == "Standard")
            & (~after["household_id"].isin(before["household_id"]))
        )
        assert new_household_movers.any()

        # Household IDs moved to are unique
        new_households = after[new_household_movers]["household_id"]
        assert new_households.nunique() == len(new_households)

        # Address IDs moved to are unique and new
        new_addresses = after[new_household_movers]["address_id"]
        assert new_addresses.nunique() == len(new_addresses)
        assert not new_addresses.isin(before["address_id"]).any()

        # Households are single-person
        assert not after[~new_household_movers]["household_id"].isin(new_households).any()
        assert (
            after[new_household_movers]["relation_to_household_head"] == "Reference person"
        ).all()


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


def test_only_living_people_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        movers = before["address_id"] != after["address_id"]
        assert after[movers]["tracked"].all()
        assert (after[movers]["alive"] == "alive").all()
