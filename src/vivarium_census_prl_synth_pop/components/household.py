import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants import data_values
from vivarium_census_prl_synth_pop.utilities import update_address_ids

HOUSEHOLD_ID_DTYPE = int
HOUSEHOLD_DETAILS_DTYPES = {
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

        # GQ households are special, with fixed IDs
        self._households = pd.DataFrame(
            {
                "address_id": data_values.GQ_HOUSING_TYPE_MAP.keys(),
                "housing_type": data_values.GQ_HOUSING_TYPE_MAP.values(),
            },
            index=data_values.GQ_HOUSING_TYPE_MAP.keys().astype(HOUSEHOLD_ID_DTYPE),
        ).astype(HOUSEHOLD_DETAILS_DTYPES)
        self._households.index.names = ["household_id"]

        self.household_details = builder.value.register_value_producer(
            "household_details",
            source=self.get_household_details,
            requires_columns=["household_id", "tracked"],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_household_details(self, idx: pd.Index) -> pd.DataFrame:
        """Source of the household_details pipeline"""
        pop = self.population_view.subview(["tracked", "household_id"]).get(idx)
        household_details = (
            pop[["household_id"]]
            .astype(HOUSEHOLD_ID_DTYPE)
            .join(
                self._households,
                on="household_id",
            )
        )
        household_details = household_details.astype(HOUSEHOLD_DETAILS_DTYPES)
        return household_details

    ##################
    # Public methods #
    ##################

    def create_households(
        self, num_households: int, housing_type: str = "Standard"
    ) -> pd.Series:
        """
        Create a specified number of new households, with new, unique address_ids.

        Parameters
        ----------
        num_households
            The number of households to create.
        housing_type
            The housing type of the new households.

        Returns
        -------
        household_ids for the new households.
        The length of this Series will be num_households.
        """
        household_ids = self._next_available_ids(num_households, taken=self._households.index)
        address_ids = self._next_available_ids(
            num_households, taken=self._households["address_id"]
        )

        new_households = pd.DataFrame(
            {
                "address_id": address_ids,
                "housing_type": housing_type,
            },
            index=household_ids.astype(HOUSEHOLD_ID_DTYPE),
        ).astype(HOUSEHOLD_DETAILS_DTYPES)

        self._households = pd.concat([self._households, new_households])

        return household_ids

    def update_household_addresses(self, household_ids: pd.Series) -> None:
        """
        Changes the address_id associated with each household_id in household_ids
        to a new, unique value.

        Parameters
        ----------
        household_ids
            The IDs of the households that should move to new addresses.
        """
        self._households, _ = update_address_ids(
            moving_units=self._households,
            units_that_move_ids=household_ids,
            starting_address_id=self._first_int_available(self._households["address_id"]),
            address_id_col_name="address_id",
        )

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def _next_available_ids(num_ids: int, taken: pd.Series):
        return Households._first_int_available(taken) + np.arange(num_ids)

    @staticmethod
    def _first_int_available(taken: pd.Series):
        if len(taken) == 0:
            return 0

        return taken.max() + 1
