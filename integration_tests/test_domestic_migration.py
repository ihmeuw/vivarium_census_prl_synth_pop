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


def test_only_living_people_move(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        movers = before["address_id"] != after["address_id"]
        assert after[movers]["tracked"].all()
        assert (after[movers]["alive"] == "alive").all()
