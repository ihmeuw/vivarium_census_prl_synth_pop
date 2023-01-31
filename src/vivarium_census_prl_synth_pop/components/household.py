import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants import data_keys, data_values
from vivarium_census_prl_synth_pop.utilities import vectorized_choice

HOUSEHOLD_DETAILS_DTYPES = {
    "household_id": int,
    "address_id": int,
    "housing_type": pd.CategoricalDtype(categories=data_values.HOUSING_TYPES),
}


class Households:
    """Manages household details"""

    def __repr__(self) -> str:
        return "Households()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "households"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.pop_config = builder.configuration.population
        self.seed = builder.configuration.randomness.random_seed
        self.start_time = get_time_stamp(builder.configuration.time.start)

        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_used = [
            "tracked",
            "household_id",
        ]

        self.population_view = builder.population.get_view(self.columns_used)

        self._households = self.sample_initial_households(builder)
        self.household_details = builder.value.register_value_producer(
            "household_details",
            source=self.get_household_details,
            requires_columns=["household_id", "tracked"],
        )

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=["household_id"],
        )

    def sample_initial_households(self, builder: Builder) -> pd.DataFrame:
        """
        Initializes the households data structure. Samples as many standard
        households as the target number of simulants living in standard
        households to ensure we aren't short. This data structure contains the
        following columns:

        census_household_id: needed to match up with persons data for population
            initialization
        household_id: this will be set as the index after initialization, but
            for convenience this is a column at this stage
        housing type: either Standard or specific GQ type
        address_id: id for the household's address
        """
        input_household_data = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)
        gq_households = pd.DataFrame(
            {
                "census_household_id": "N/A",
                "household_id": data_values.GQ_HOUSING_TYPE_MAP.keys(),
                "address_id": data_values.GQ_HOUSING_TYPE_MAP.keys(),
                "housing_type": data_values.GQ_HOUSING_TYPE_MAP.values(),
            }
        )

        target_gq_pop_size = int(
            self.pop_config.population_size * data_values.PROP_POPULATION_IN_GQ
        )
        target_standard_housing_pop_size = (
            self.pop_config.population_size - target_gq_pop_size
        )

        sampled_census_ids = vectorized_choice(
            options=input_household_data["census_household_id"],
            n_to_choose=target_standard_housing_pop_size,
            weights=input_household_data["household_weight"],
            additional_key=f"sample_households_{self.seed}",
        )
        standard_household_ids = gq_households.index.size + np.arange(len(sampled_census_ids))
        sampled_standard_households = pd.DataFrame(
            {
                "census_household_id": sampled_census_ids,
                "household_id": standard_household_ids,
                "address_id": standard_household_ids,
                "housing_type": "Standard",
            }
        )

        households = pd.concat([gq_households, sampled_standard_households])
        households["housing_type"] = households["housing_type"].astype(
            HOUSEHOLD_DETAILS_DTYPES["housing_type"]
        )

        return households

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Remove unused households from the household datastructure immediately
        following initialization and set household id as index
        """
        if pop_data.creation_time < self.start_time:
            households = self._households.set_index("household_id")
            # Drop all households that aren't in population except for GQ households
            gq_index = households.index[households["housing_type"] != "Standard"]
            existing_households = pd.Index(
                self.population_view.subview(["household_id"])
                .get(pop_data.index)["household_id"]
                .drop_duplicates()
            )
            self.set_households(households.loc[gq_index.union(existing_households)])

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_household_details(self, idx: pd.Index) -> pd.DataFrame:
        """Source of the household_details pipeline"""
        pop = self.population_view.subview(["tracked", "household_id"]).get(idx)
        household_details = pop[["household_id"]].join(
            self._households,
            on="household_id",
        )
        household_details = household_details.astype(HOUSEHOLD_DETAILS_DTYPES)
        return household_details

    #######################
    # Getters and setters #
    #######################

    def get_households(self) -> pd.DataFrame:
        return self._households

    def set_households(self, households: pd.DataFrame) -> None:
        self._households = households
