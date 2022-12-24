import numpy as np
import pytest

from integration_tests.conftest import TIME_STEPS_TO_TEST
from vivarium_census_prl_synth_pop.constants import data_values

# TODO: Broader test coverage


def test_gq_proportion(tracked_live_populations):
    pop = tracked_live_populations[0]
    # GQ proportion matches expectation
    assert np.isclose(
        pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP).mean(),
        data_values.PROP_POPULATION_IN_GQ,
    )


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
