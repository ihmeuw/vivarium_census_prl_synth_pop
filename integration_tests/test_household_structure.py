import numpy as np
import pytest

from integration_tests.conftest import TIME_STEPS_TO_TEST
from vivarium_census_prl_synth_pop.constants import data_values


# TODO: Broader test coverage
def test_initial_population_size(sim, tracked_live_populations):
    pop_size = sim.configuration.population.population_size
    pop = tracked_live_populations[0]

    assert pop.index.size == pop_size
    assert (pop["alive"] == "dead").sum() == 0


# TODO stop skipping once MIC-3527 and MIC-3714 have been implemented
@pytest.mark.parametrize("time_step", TIME_STEPS_TO_TEST)
@pytest.mark.skip
def test_all_households_have_reference_person(tracked_live_populations, time_step):
    pop = tracked_live_populations[time_step]
    non_gq_household_ids = pop[~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)][
        "household_id"
    ].unique()
    reference_person_household_ids = pop.loc[
        pop["relation_to_household_head"] == "Reference person", "household_id"
    ].values

    # Assert these two sets are identical
    assert non_gq_household_ids.size == reference_person_household_ids.size
    assert np.setxor1d(non_gq_household_ids, reference_person_household_ids).size == 0
