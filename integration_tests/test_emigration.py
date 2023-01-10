import pandas as pd


def test_individuals_emigrate(simulants_on_adjacent_timesteps):
    # This is a very rare event, so we can't assert that it happens
    # on every timestep; instead, we aggregate across all timesteps.
    all_individual_emigration_status = []

    for before, after in simulants_on_adjacent_timesteps:
        households_remaining_in_us = after[after["in_united_states"]]["household_id"]
        individual_emigrants = (
            before["in_united_states"]
            & ~after["in_united_states"]
            & after["household_id"].isin(households_remaining_in_us)
        )

        # Only occurs to living, tracked people
        living_tracked = before["tracked"] & (before["alive"] == "alive")
        assert living_tracked[individual_emigrants].all()

        assert after["time"].nunique() == 1
        current_time = after["time"].iloc[0]
        assert (after[individual_emigrants]["exit_time"] == current_time).all()

        all_individual_emigration_status.append(individual_emigrants[living_tracked])

    all_individual_emigration_status = pd.concat(
        all_individual_emigration_status, ignore_index=True
    )
    # VERY conservative upper bound on how often this should be occurring, as a proportion
    # of living, tracked simulant-steps
    assert 0 < all_individual_emigration_status.mean() < 0.1


def test_non_gq_individuals_emigrate(simulants_on_adjacent_timesteps):
    # This is a very rare event, so we can't assert that it happens
    # on every timestep; instead, we aggregate across all timesteps.
    all_non_gq_emigration_status = []

    for before, after in simulants_on_adjacent_timesteps:
        households_remaining_in_us = after[after["in_united_states"]]["household_id"]
        non_gq_individual_emigrants = (
            before["in_united_states"]
            & ~after["in_united_states"]
            & after["household_id"].isin(households_remaining_in_us)
            & (before["housing_type"] == "Standard")
        )

        # Only occurs to living, tracked people
        living_tracked = before["tracked"] & (before["alive"] == "alive")
        assert living_tracked[non_gq_individual_emigrants].all()

        assert (
            before[non_gq_individual_emigrants]["relation_to_household_head"]
            != "Reference person"
        ).all()

        all_non_gq_emigration_status.append(non_gq_individual_emigrants[living_tracked])

    all_non_gq_emigration_status = pd.concat(all_non_gq_emigration_status, ignore_index=True)
    # VERY conservative upper bound on how often this should be occurring, as a proportion
    # of living, tracked simulant-steps
    assert 0 < all_non_gq_emigration_status.mean() < 0.1


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
