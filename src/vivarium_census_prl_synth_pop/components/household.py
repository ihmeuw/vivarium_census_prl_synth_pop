import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder

from vivarium_census_prl_synth_pop.constants import data_values
from vivarium_census_prl_synth_pop.utilities import random_integers

HOUSEHOLD_ID_DTYPE = int
HOUSEHOLD_DETAILS_DTYPES = {
    "address_id": int,
    "housing_type": pd.CategoricalDtype(categories=data_values.HOUSING_TYPES),
    "po_box": int,
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
        self.columns_used = [
            "tracked",
            "household_id",
        ]

        self.population_view = builder.population.get_view(self.columns_used)
        # self.clock = builder.time.clock()

        # GQ households are special, with fixed IDs
        gq_household_ids = (
            pd.Series(data_values.GQ_HOUSING_TYPE_MAP.keys())
            .astype(HOUSEHOLD_ID_DTYPE)
            .rename("household_id")
        )
        self._households = pd.DataFrame(
            {
                "address_id": data_values.GQ_HOUSING_TYPE_MAP.keys(),
                "housing_type": data_values.GQ_HOUSING_TYPE_MAP.values(),
                "po_box": data_values.NO_PO_BOX,
            },
            index=gq_household_ids,
        ).astype(HOUSEHOLD_DETAILS_DTYPES)

        self.household_details = builder.value.register_value_producer(
            "household_details",
            source=self.get_household_details,
            requires_columns=["household_id", "tracked"],
        )

        self.randomness = builder.randomness.get_stream(self.name)

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
        po_boxes = self.generate_po_boxes(num_households)

        new_households = pd.DataFrame(
            {
                "address_id": address_ids,
                "housing_type": housing_type,
                "po_box": po_boxes,
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
        new_address_ids = self._next_available_ids(
            len(household_ids), taken=self._households["address_id"]
        )
        self._households.loc[household_ids, "address_id"] = new_address_ids

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def _next_available_ids(num_ids: int, taken: pd.Series):
        # NOTE: Our approach to finding available IDs assumes that households are never deleted --
        # because if the household with the highest ID were deleted, and we only used the current
        # state to find available IDs, we would re-assign that ID!
        # No deletion is guaranteed by the inability to delete households through the public
        # methods above.
        return Households._first_int_available(taken) + np.arange(num_ids)

    @staticmethod
    def _first_int_available(taken: pd.Series):
        if len(taken) == 0:
            return 0

        return taken.max() + 1

    def generate_po_boxes(self, num_addresses) -> np.array:
        different_mailing_physical_addresses = self.randomness.filter_for_probability(
            pd.Index(list(range(num_addresses))),
            1 - data_values.PROBABILITY_OF_SAME_MAILING_PHYSICAL_ADDRESS,
            # additional_key=f"po_box_filter_{self.clock}",
            additional_key="po_box_filter",
        )
        po_boxes = pd.Series(
            data_values.NO_PO_BOX, index=pd.Index(list(range(num_addresses)))
        )
        po_boxes.loc[different_mailing_physical_addresses] = random_integers(
            min_val=1,
            max_val=20_000,
            index=different_mailing_physical_addresses,
            randomness=self.randomness,
            # additional_key=f"po_box_int_{self.clock}",
            additional_key="po_box_number",
        )
        return np.array(po_boxes)
