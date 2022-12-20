import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants import data_keys, data_values, paths
from vivarium_census_prl_synth_pop.utilities import (
    filter_by_rate,
    update_address_id_for_unit_and_sims,
)


class HouseholdMigration:
    """
    - on simulant_initialization, adds address to population table per household_id
    - on time_step, updates some households to new addresses

    ASSUMPTION:
    - households will always move to brand-new addresses (as opposed to vacated addresses)
    - puma will not change (pumas and zip codes currently unrelated)
    """

    def __repr__(self) -> str:
        return "HouseholdMigration()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "household_migration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.config = builder.configuration
        self.location = builder.data.load(data_keys.POPULATION.LOCATION)
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.household_address_id_count = 0

        # TODO: consider subsetting to housing_type=="standard" rows if abie decides GQ never moves addresses
        move_rate_data = pd.read_csv(
                paths.HOUSEHOLD_MOVE_RATE_PATH,
                usecols=[
                    "sex",
                    "race_ethnicity",
                    "age_start",
                    "age_end",
                    "household_rate",
                    "housing_type",
                ],
        )
        move_rate_data = builder.lookup.build_table(
            data=move_rate_data.loc[move_rate_data["housing_type"] == "Standard"].drop(
                columns="housing_type"
            ),
            key_columns=["sex", "race_ethnicity"],
            parameter_columns=["age"],
            value_columns=["household_rate"],
        )
        self.household_move_rate = builder.value.register_rate_producer(
            f"{self.name}.move_rate", source=move_rate_data
        )
        self.proportion_households_leaving_country = builder.lookup.build_table(
            data=data_values.PROPORTION_HOUSEHOLDS_LEAVING_COUNTRY
        )

        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_created = ["address_id"]
        self.columns_used = [
            "household_id",
            "relation_to_household_head",
            "address_id",
            "tracked",
            "exit_time",
        ]
        self.population_view = builder.population.get_view(self.columns_used)

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=["household_id"],
            creates_columns=self.columns_created,
        )
        builder.event.register_listener("time_step", self.on_time_step)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        add addresses to each household in the population table
        """
        if pop_data.creation_time < self.start_time:
            households = self.population_view.subview(["household_id"]).get(pop_data.index)
            household_ids = households["household_id"].unique()
            n = np.float64(len(household_ids))
            address_assignments = {
                household: address_id
                for (household, address_id) in zip(household_ids, np.arange(n))
            }
            households["address_id"] = households["household_id"].map(address_assignments)
            self.household_address_id_count = n
            self.population_view.update(households)
        else:
            parent_ids = pop_data.user_data["parent_ids"]
            mothers = self.population_view.get(parent_ids.unique())
            new_births = pd.DataFrame(data={"parent_id": parent_ids}, index=pop_data.index)

            # assign babies inherited traits
            new_births = new_births.merge(
                mothers[self.columns_created], left_on="parent_id", right_index=True
            )
            self.population_view.update(new_births[self.columns_created])

    def on_time_step(self, event: Event):
        """
        choose which households move;
        move those households to a new address
        """
        pop = self.population_view.get(event.index)
        household_ids = pop.loc[
            pop["relation_to_household_head"] == "Reference person", "household_id"
        ]

        all_household_ids_that_move = self.randomness.filter_for_rate(
            household_ids,
            self.household_move_rate(household_ids.index),
            "all_moving_households",
        )

        # Determine which households move abroad
        abroad_household_ids = filter_by_rate(
            all_household_ids_that_move,
            self.randomness,
            self.proportion_households_leaving_country,
            "abroad_households",
        )
        domestic_household_ids = all_household_ids_that_move.loc[
            ~all_household_ids_that_move.isin(abroad_household_ids)
        ]

        # Process households moving abroad
        abroad_households_idx = pop.loc[pop["household_id"].isin(abroad_household_ids)].index
        if len(abroad_households_idx) > 0:
            pop.loc[abroad_households_idx, "exit_time"] = event.time
            pop.loc[abroad_households_idx, "tracked"] = False

        # Process households moving domestic
        # Make new address map
        households = pop.loc[household_ids.index, ["household_id", "address_id"]].set_index(
            "household_id"
        )
        domestic_households_idx = households.loc[domestic_household_ids].index
        # This is the same thing as domestic_household_ids but is a df with columns="address_id" with household_id as
        #   the index to match how we handle businesses in the following function

        # Update both state tables and address_id tracker.
        (
            pop,
            households,
            self.household_address_id_count,
        ) = update_address_id_for_unit_and_sims(
            pop,
            moving_units=households,
            units_that_move_ids=domestic_households_idx,
            total_address_id_count=self.household_address_id_count,
            unit_id_col_name="household_id",
            address_id_col_name="address_id",
        )

        self.population_view.update(pop)
