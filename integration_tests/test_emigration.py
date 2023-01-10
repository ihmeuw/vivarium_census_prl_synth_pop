import pandas as pd


def test_there_is_emigration(simulants_on_adjacent_timesteps):
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
    assert (emigrants["housing_type_before"] == emigrants["housing_type_after"]).all()
    assert (
        emigrants["relation_to_household_head_before"]
        == emigrants["relation_to_household_head_after"]
    ).all()

    # VERY conservative upper bound on how often this should be occurring, as a proportion
    # of living, tracked simulant-steps
    assert 0 < all_emigration_status[living_tracked].mean() < 0.1


def test_individuals_emigrate(simulants_on_adjacent_timesteps):
    _, all_individual_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        lambda before, after: before["household_id"].isin(
            after[after["tracked"]]["household_id"]
        ),
    )

    assert 0 < all_individual_emigration_status.mean() < 0.1


def test_non_gq_individuals_emigrate(simulants_on_adjacent_timesteps):
    all_simulant_links, all_non_gq_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        lambda before, after: (
            after["household_id"].isin(after[after["in_united_states"]]["household_id"])
            & (before["housing_type"] == "Standard")
        ),
    )

    emigrants = all_simulant_links[all_non_gq_emigration_status]
    assert (emigrants["relation_to_household_head_before"] != "Reference person").all()

    assert 0 < all_non_gq_emigration_status.mean() < 0.1


def test_gq_individuals_emigrate(simulants_on_adjacent_timesteps):
    _, all_gq_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        lambda before, after: before["housing_type"] != "Standard",
    )

    assert 0 < all_gq_emigration_status.mean() < 0.1


def test_households_emigrate(simulants_on_adjacent_timesteps):
    all_simulant_links, all_household_emigration_status = all_time_emigration_condition(
        simulants_on_adjacent_timesteps,
        lambda before, after: ~before["household_id"].isin(
            after[after["tracked"]]["household_id"]
        ),
    )

    emigrants = all_simulant_links[all_household_emigration_status]

    # Never GQ households
    assert (emigrants["housing_type_before"] == "Standard").all()

    assert 0 < all_household_emigration_status.mean() < 0.1


def test_emigrated_people_are_untracked(populations):
    # For now, those who are outside the US are untracked and nothing happens to them
    # May change if we want to allow emigrants to come *back* into the US
    for pop in populations:
        assert not pop[~pop["in_united_states"]]["tracked"].any()


def test_nothing_happens_to_untracked_people(simulants_on_adjacent_timesteps):
    for before, after in simulants_on_adjacent_timesteps:
        untracked = ~before["tracked"]
        if untracked.sum() == 0:
            continue

        columns_do_not_change = [c for c in before.columns if c != "time"]
        assert before.loc[untracked, columns_do_not_change].equals(
            after.loc[untracked, columns_do_not_change]
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
        all_time_links.append(before.join(after, lsuffix="_before", rsuffix="_after"))
        all_time_condition.append(
            before["in_united_states"]
            & ~after["in_united_states"]
            & condition_func(before, after)
        )

    all_time_links = pd.concat(all_time_links, ignore_index=True)
    all_time_condition = pd.concat(all_time_condition, ignore_index=True)

    if subset_to_living_tracked:
        living_tracked = (all_time_links["alive_before"] == "alive") & all_time_links[
            "tracked_before"
        ]
        all_time_links = all_time_links[living_tracked]
        all_time_condition = all_time_condition[living_tracked]

    return all_time_links, all_time_condition
