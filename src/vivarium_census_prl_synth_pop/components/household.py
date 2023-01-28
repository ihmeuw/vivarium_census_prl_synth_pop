import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants.data_values import (
    GQ_HOUSING_TYPE_MAP,
    HOUSING_TYPES,
)


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
        self.start_time = get_time_stamp(builder.configuration.time.start)

        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_used = [
            "tracked",
            "household_id",
        ]

        self.population_view = builder.population.get_view(self.columns_used)

        self.households = None
        self.household_details = builder.value.register_value_producer(
            "household_details",
            source=self.get_household_details,
            requires_columns=["household_id", "tracked"],
        )

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=["household_id"],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        add addresses to each household in the population table
        """
        if pop_data.creation_time < self.start_time:  # initial pop
            self.households = self.generate_initial_households(pop_data)

    ##################
    # Helper methods #
    ##################

    def generate_initial_households(self, pop_data: SimulantData) -> pd.DataFrame():
        households = self.population_view.subview(["household_id"]).get(pop_data.index)
        households = (
            households.drop_duplicates().sort_values("household_id").set_index("household_id")
        )
        households["address_id"] = np.arange(len(households))
        households["housing_type"] = households.index.map(GQ_HOUSING_TYPE_MAP).fillna(
            "Standard"
        )
        # set housing type dtype
        households["housing_type"] = households["housing_type"].astype(
            pd.CategoricalDtype(categories=HOUSING_TYPES)
        )

        return households

    def get_household_details(self, idx: pd.Index) -> pd.DataFrame:
        """Source of the household_details pipeline"""
        pop = self.population_view.get(idx)[["household_id"]]
        household_details = pop.join(
            self.households,
            on="household_id",
        )
        # FIXME: why is the `housing_type` column losing its CategoricalDtype?
        household_details["housing_type"] = household_details["housing_type"].astype(
            pd.CategoricalDtype(categories=HOUSING_TYPES)
        )

        return household_details
