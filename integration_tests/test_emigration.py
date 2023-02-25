import pandas as pd

from .conftest import FuzzyTest


def test_there_is_emigration(simulants_on_adjacent_timesteps, fuzzy_tester: FuzzyTest):
    # This is a very rare event, so we can't assert that it happens
    # on every timestep; instead, we aggregate across all timesteps.
    all_simulant_links, all_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        subset_to_living_tracked=False,
    )

    # Only occurs to living, tracked simulants
    living_tracked = all_simulant_links["tracked_before"] & (
        all_simulant_links["alive_before"] == "alive"
    )
    assert living_tracked[all_emigration_status].all()

    emigrants = all_simulant_links[all_emigration_status]

    assert (emigrants["exit_time_after"] == emigrants["time_after"]).all()

    # Does not change other attributes
    assert (emigrants["household_id_before"] == emigrants["household_id_after"]).all()
    assert (
        emigrants["household_details.housing_type_before"]
        == emigrants["household_details.housing_type_after"]
    ).all()
    assert (
        emigrants["relation_to_household_head_before"]
        == emigrants["relation_to_household_head_after"]
    ).all()

    # TODO: Uncomment once emigration rates are fixed
    # fuzzy_tester.fuzzy_assert_proportion(
    #     # Emigration is pretty rare! Around 1.25m people emigrate per year.
    #     all_emigration_status[living_tracked],
    #     true_value=(1_250_000 / 330_000_000) / 12,
    # )


def test_individuals_emigrate(simulants_on_adjacent_timesteps, fuzzy_tester: FuzzyTest):
    _, all_individual_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        lambda before, after: before["household_id"].isin(
            after[after["in_united_states"]]["household_id"]
        ),
    )

    # TODO: Uncomment once emigration rates are fixed
    # fuzzy_tester.fuzzy_assert_proportion(
    #     # Emigration is pretty rare! Around 1.25m people emigrate per year.
    #     # About half is individuals.
    #     all_individual_emigration_status,
    #     true_value=((1_250_000 / 2) / 330_000_000) / 12,
    # )


def test_non_gq_individuals_emigrate(
    simulants_on_adjacent_timesteps, fuzzy_tester: FuzzyTest
):
    all_simulant_links, all_non_gq_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        lambda before, after: (
            after["household_id"].isin(after[after["in_united_states"]]["household_id"])
            & (before["household_details.housing_type"] == "Standard")
        ),
    )

    emigrants = all_simulant_links[all_non_gq_emigration_status]
    assert (emigrants["relation_to_household_head_before"] != "Reference person").all()

    # TODO: Uncomment once emigration rates are fixed
    # fuzzy_tester.fuzzy_assert_proportion(
    #     # Non-GQ emigration is basically all of the individual emigration
    #     all_non_gq_emigration_status,
    #     true_value=((1_250_000 / 2) / 330_000_000) / 12,
    # )


def test_gq_individuals_emigrate(simulants_on_adjacent_timesteps, fuzzy_tester: FuzzyTest):
    _, all_gq_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        lambda before, after: before["household_details.housing_type"] != "Standard",
    )

    # TODO: Uncomment once emigration rates are fixed
    # fuzzy_tester.fuzzy_assert_proportion(
    #     # GQ emigration is *very* rare! Around 100,000 per year (according to the data
    #     # from which we derived our input rates).
    #     all_gq_emigration_status,
    #     true_value=(100_000 / 330_000_000) / 12,
    # )


def test_households_emigrate(simulants_on_adjacent_timesteps, fuzzy_tester: FuzzyTest):
    all_simulant_links, all_household_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        lambda before, after: ~before["household_id"].isin(
            after[after["in_united_states"]]["household_id"]
        ),
    )

    emigrants = all_simulant_links[all_household_emigration_status]

    # GQ households never emigrate
    assert (emigrants["household_details.housing_type_before"] == "Standard").all()

    # TODO: Uncomment once emigration rates are fixed
    # fuzzy_tester.fuzzy_assert_proportion(
    #     # We assume household emigration is the other half besides individual.
    #     all_household_emigration_status,
    #     true_value=((1_250_000 / 2) / 330_000_000) / 12,
    # )


def test_emigrated_people_are_untracked(populations):
    # For now, those who are outside the US are untracked and nothing happens to them
    # May change if we want to allow emigrants to come *back* into the US
    for pop in populations:
        assert not pop[~pop["in_united_states"]]["tracked"].any()


def test_nothing_happens_to_untracked_people(
    simulants_on_adjacent_timesteps, pipeline_columns
):
    for before, after in simulants_on_adjacent_timesteps:
        untracked = ~before["tracked"]
        if untracked.sum() == 0:
            continue

        known_changing_cols = pipeline_columns + ["time", "state_id_for_lookup"]
        columns_that_should_not_change = [
            c for c in before.columns if c not in known_changing_cols
        ]
        assert before.loc[untracked, columns_that_should_not_change].equals(
            after.loc[untracked, columns_that_should_not_change]
        )


def all_time_emigration_condition(
    simulants_on_adjacent_timesteps,
    condition_func=lambda before, after: True,
    subset_to_living_tracked=True,
):
    all_time_links = []
    all_time_condition = []

    for before, after in simulants_on_adjacent_timesteps:
        assert before.index.equals(after.index)
        # It is important that the subset to living and tracked happens before
        # condition_func -- otherwise the custom condition may consider simulants
        # who were already dead/untracked before this time step
        # (e.g. when determining which households still exist in the US)
        if subset_to_living_tracked:
            living_tracked = (before["alive"] == "alive") & before["tracked"]
            before = before[living_tracked]
            after = after[living_tracked]

        all_time_links.append(
            before.join(after, lsuffix="_before", rsuffix="_after")
            .reset_index()
            .rename(columns={"index": "simulant_id"})
        )
        all_time_condition.append(
            before["in_united_states"]
            & ~after["in_united_states"]
            & condition_func(before, after)
        )

    all_time_links = pd.concat(all_time_links, ignore_index=True)
    all_time_condition = pd.concat(all_time_condition, ignore_index=True)

    return all_time_links, all_time_condition
