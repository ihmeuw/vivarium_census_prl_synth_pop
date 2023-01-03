import numpy as np
import pytest

from integration_tests.conftest import TIME_STEPS_TO_TEST
from vivarium_census_prl_synth_pop.constants import data_values

# TODO: Broader test coverage


def test_gq_proportion(sim, tracked_live_populations):
    pop_size = sim.configuration.population.population_size
    pop = tracked_live_populations[0]
    expected_gq_population = pop_size * data_values.PROP_POPULATION_IN_GQ
    min_gq_population = expected_gq_population - (data_values.MAX_HOUSEHOLD_SIZE - 1)
    max_gq_population = expected_gq_population + (data_values.MAX_HOUSEHOLD_SIZE - 1)
    actual_gq_population = pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP).sum()

    # GQ proportion matches expectation
    assert min_gq_population <= actual_gq_population <= max_gq_population


@pytest.mark.parametrize("time_step", TIME_STEPS_TO_TEST)
def test_gq_relationship_column(tracked_live_populations, time_step):
    pop = tracked_live_populations[time_step]
    # All people in institutional GQ have the correct "relation to household head"
    assert (
        (pop.household_id.isin(data_values.INSTITUTIONAL_GROUP_QUARTER_IDS.values()))
        == (pop.relation_to_household_head == "Institutionalized GQ pop")
    ).all()

    # All people in non-institutional GQ have the correct "relation to household head"
    assert (
        (pop.household_id.isin(data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS.values()))
        == (pop.relation_to_household_head == "Noninstitutionalized GQ pop")
    ).all()


@pytest.mark.parametrize("time_step", TIME_STEPS_TO_TEST)
def test_gq_housing_type_column(tracked_live_populations, time_step):
    pop = tracked_live_populations[time_step]
    # All people in institutional GQ have a correct housing type
    assert (
        (pop["housing_type"].isin(data_values.INSTITUTIONAL_GROUP_QUARTER_IDS.keys()))
        == (pop["relation_to_household_head"] == "Institutionalized GQ pop")
    ).all()

    # All people in non-institutional GQ have a correct housing type
    assert (
        (pop["housing_type"].isin(data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS.keys()))
        == (pop["relation_to_household_head"] == "Noninstitutionalized GQ pop")
    ).all()

    # All non-GQ people have standard housing type
    assert (
        (pop["housing_type"] == "Standard")
        == (
            ~pop["relation_to_household_head"].isin(
                ["Institutionalized GQ pop", "Noninstitutionalized GQ pop"]
            )
        )
    ).all()
