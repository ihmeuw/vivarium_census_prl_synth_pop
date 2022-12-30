import pytest

# TODO: Broader test coverage


def test_individuals_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        individual_movers = before["household_id"] != after["household_id"]
        assert individual_movers.any()
        assert (
            before[individual_movers]["address_id"] != after[individual_movers]["address_id"]
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

        # Never GQ households
        assert (before[household_movers]["housing_type"] == "Standard").all()


def test_only_living_people_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        movers = before["address_id"] != after["address_id"]
        assert after[movers]["tracked"].all()
        assert (after[movers]["alive"] == "alive").all()
