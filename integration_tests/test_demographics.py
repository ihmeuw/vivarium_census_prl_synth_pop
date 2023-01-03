import pandas as pd

from vivarium_census_prl_synth_pop.constants import metadata


# TODO: Broader test coverage
def test_initial_population_size(sim, populations):
    pop_size = sim.configuration.population.population_size
    pop = populations[0]

    assert pop.index.size == pop_size
    assert pop["tracked"].all()
    assert (pop["alive"] == "alive").all()


# Todo: test that sex is unvarying
# Todo: test that race is unvarying


def test_sex_is_categorical(tracked_live_populations):
    for pop in tracked_live_populations:
        sex = pop["sex"]

        # Assert the dtype is correct and that there are no NaNs
        assert sex.dtype == pd.CategoricalDtype(categories=metadata.SEXES)
        assert not sex.isnull().any()


def test_race_ethnicity_is_categorical(tracked_live_populations):
    for pop in tracked_live_populations:
        race_ethnicity = pop["race_ethnicity"]

        # Assert the dtype is correct and that there are no NaNs
        assert race_ethnicity.dtype == pd.CategoricalDtype(
            categories=metadata.RACE_ETHNICITIES
        )
        assert not race_ethnicity.isnull().any()
