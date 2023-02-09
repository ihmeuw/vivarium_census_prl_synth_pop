from typing import Optional

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants import data_values
from vivarium_census_prl_synth_pop.utilities import random_integers, vectorized_choice

HOUSEHOLD_ID_DTYPE = int
HOUSEHOLD_DETAILS_DTYPES = {
    "address_id": int,
    "housing_type": pd.CategoricalDtype(categories=data_values.HOUSING_TYPES),
    "po_box": int,
    "puma": int,
    "state_id": int,
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
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_used = [
            "tracked",
            "household_id",
        ]

        self.population_view = builder.population.get_view(self.columns_used)
        # FIXME: Use the US address static file for states/pumas
        self.state_puma_options = builder.data.load("population.households")[
            ["state", "puma"]
        ].drop_duplicates()

        # GQ households are special, with fixed IDs
        gq_household_ids = (
            pd.Series(data_values.GQ_HOUSING_TYPE_MAP.keys())
            .astype(HOUSEHOLD_ID_DTYPE)
            .rename("household_id")
        )
        states_pumas = self.randomly_sample_states_pumas(
            gq_household_ids, random_seed=builder.configuration.randomness.random_seed
        )
        self._households = pd.DataFrame(
            {
                "address_id": data_values.GQ_HOUSING_TYPE_MAP.keys(),
                "housing_type": data_values.GQ_HOUSING_TYPE_MAP.values(),
                "po_box": data_values.NO_PO_BOX,
                "state_id": states_pumas["state_id"],
                "puma": states_pumas["puma"],
            },
            index=gq_household_ids,
        ).astype(HOUSEHOLD_DETAILS_DTYPES)

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
        self,
        num_households: int,
        housing_type: str = "Standard",
        states_pumas: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Create a specified number of new households, with new, unique address_ids.
        This updates the _households data structure with the new household data
        and also returns a pd.Series of the new household IDs

        Parameters
        ----------
        num_households
            The number of households to create.
        housing_type
            The housing type of the new households.
        states_pumas (optional)
            A dataframe of the households' state_ids and pumas. If it is None, then
            new states ande pumas will be sampled from ACS data.

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
        if states_pumas is None:
            states_pumas = self.randomly_sample_states_pumas(household_ids)
        states = np.array(states_pumas["state_id"])
        pumas = np.array(states_pumas["puma"])

        new_households = pd.DataFrame(
            {
                "address_id": address_ids,
                "housing_type": housing_type,
                "po_box": po_boxes,
                "state_id": states,
                "puma": pumas,
            },
            index=household_ids.astype(HOUSEHOLD_ID_DTYPE),
        ).astype(HOUSEHOLD_DETAILS_DTYPES)

        self._households = pd.concat([self._households, new_households])

        return household_ids

    def update_household_addresses(self, household_ids: pd.Series) -> None:
        """
        Changes the address information (address_id, state, puma) associated with
        each household_id in household_ids to new values with unique address_ids.

        Parameters
        ----------
        household_ids
            The IDs of the households that should move to new addresses.
        """
        new_address_ids = self._next_available_ids(
            len(household_ids), taken=self._households["address_id"]
        )
        po_boxes = self.generate_po_boxes(len(household_ids))
        states_pumas = self.randomly_sample_states_pumas(household_ids)
        self._households.loc[household_ids, "address_id"] = new_address_ids
        self._households.loc[household_ids, "po_box"] = po_boxes
        self._households.loc[household_ids, ["state_id", "puma"]] = states_pumas

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
            additional_key="po_box_filter",
        )
        po_boxes = pd.Series(
            data_values.NO_PO_BOX, index=pd.Index(list(range(num_addresses)))
        )
        po_boxes.loc[different_mailing_physical_addresses] = random_integers(
            min_val=data_values.MIN_PO_BOX,
            max_val=data_values.MAX_PO_BOX,
            index=different_mailing_physical_addresses,
            randomness=self.randomness,
            additional_key="po_box_number",
        )
        return np.array(po_boxes)

    def randomly_sample_states_pumas(
        self, household_ids: pd.Series, random_seed: int = None
    ) -> pd.DataFrame:
        if random_seed is None:  # Use the randomness_stream
            randomness_stream = self.randomness
        else:
            randomness_stream = None
        states_pumas_idx = vectorized_choice(
            options=self.state_puma_options.index,
            n_to_choose=len(household_ids),
            randomness_stream=randomness_stream,
            additional_key="sample_states_pumas",
            random_seed=random_seed,
        )
        states_pumas = self.state_puma_options.loc[states_pumas_idx]
        states_pumas.index = household_ids
        states_pumas.rename(columns={"state": "state_id"}, inplace=True)
        return states_pumas
