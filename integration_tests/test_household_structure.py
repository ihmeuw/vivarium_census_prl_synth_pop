import numpy as np
import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.constants import data_values, metadata

# TODO: Broader test coverage


def test_housing_type_is_categorical(tracked_live_populations):
    for pop in tracked_live_populations:
        housing_type = pop["housing_type"]

        # Assert the dtype is correct and that there are no NaNs
        assert housing_type.dtype == pd.CategoricalDtype(categories=data_values.HOUSING_TYPES)
        assert not housing_type.isnull().any()


def test_relationship_is_categorical(tracked_live_populations):
    for pop in tracked_live_populations:
        relationship = pop["relation_to_household_head"]

        # Assert the dtype is correct and that there are no NaNs
        assert relationship.dtype == pd.CategoricalDtype(categories=metadata.RELATIONSHIPS)
        assert not relationship.isnull().any()


# TODO stop skipping once MIC-3527 and MIC-3714 have been implemented
@pytest.mark.skip
def test_all_households_have_reference_person(tracked_live_populations):
    for pop in tracked_live_populations:
        non_gq_household_ids = pop[
            ~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)
        ]["household_id"].unique()
        reference_person_household_ids = pop.loc[
            pop["relation_to_household_head"] == "Reference person", "household_id"
        ].values

        # Assert these two sets are identical
        assert non_gq_household_ids.size == reference_person_household_ids.size
        assert np.setxor1d(non_gq_household_ids, reference_person_household_ids).size == 0


def test_household_id_and_address_id_correspond(tracked_live_populations):
    for pop in tracked_live_populations:
        # 1-to-1 at any given point in time
        assert (pop.groupby("household_id")["address_id"].nunique() == 1).all()
        assert (pop.groupby("address_id")["household_id"].nunique() == 1).all()

    # Even over time, there is only 1 household_id for each address_id -- they are not reused
    all_time_pop = pd.concat(tracked_live_populations, ignore_index=True)
    assert (all_time_pop.groupby("address_id")["household_id"].nunique() == 1).all()
