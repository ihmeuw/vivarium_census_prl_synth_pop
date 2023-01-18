import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants import metadata, paths
from vivarium_census_prl_synth_pop.constants.data_values import HOUSING_TYPES
from vivarium_census_prl_synth_pop.utilities import update_address_id_for_unit_and_sims


class HouseholdMigration:
    """
    - on simulant_initialization, adds household details
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
        self.start_time = get_time_stamp(builder.configuration.time.start)

        move_rate_data = builder.lookup.build_table(
            data=pd.read_csv(
                paths.HOUSEHOLD_DOMESTIC_MIGRATION_RATES_PATH,
            ),
            key_columns=["sex", "race_ethnicity"],
            parameter_columns=["age"],
            value_columns=["household_domestic_migration_rate"],
        )
        self.household_move_rate = builder.value.register_rate_producer(
            f"{self.name}.move_rate", source=move_rate_data
        )

        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_used = [
            "household_id",
            "housing_type",
            "relation_to_household_head",
            "address_id",
        ]

        self.population_view = builder.population.get_view(self.columns_used)

        self.households = None
        self.household_details = builder.value.register_value_producer(
            "household_details",
            source=self.generate_household_details,
            requires_columns=["household_id"],
        )

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=["household_id"],
        )
        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            priority=metadata.PRIORITY_MAP["household.on_time_step"],
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
            households = self.population_view.subview(["household_id"]).get(pop_data.index)
            household_ids = sorted(households["household_id"].unique())
            address_assignments = {
                household: address_id
                for (household, address_id) in zip(household_ids, np.arange(len(household_ids)))
            }
            households["address_id"] = households["household_id"].map(address_assignments)
            self.population_view.update(households)
        else:  # newborns
            parent_ids = pop_data.user_data["parent_ids"]
            mothers = self.population_view.get(parent_ids.unique())
            mothers["address_id"] = mothers["address_id"].astype(
                int
            )
            new_births = pd.DataFrame(data={"parent_id": parent_ids}, index=pop_data.index)

            # assign babies inherited traits
            new_births = new_births.merge(
                mothers["address_id"], left_on="parent_id", right_index=True
            )
            self.population_view.update(new_births["address_id"])

    def on_time_step(self, event: Event):
        """
        choose which households move;
        move those households to a new address
        """
        # NOTE: Currently, it is possible for a household not to have a living reference person;
        # in this case, that household can no longer move.
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",
        )
        reference_people = pop[pop["relation_to_household_head"] == "Reference person"]

        household_sizes = pop.groupby("household_id").size()

        multi_person_household_ids = household_sizes.index[household_sizes > 1]

        reference_people_eligible_to_move = reference_people[
            reference_people["household_id"].isin(multi_person_household_ids)
        ]

        movers = self.randomness.filter_for_rate(
            reference_people_eligible_to_move,
            self.household_move_rate(reference_people_eligible_to_move.index),
            "moving_households",
        )

        # Make household ID -> address ID map
        households = (
            pop[["household_id", "address_id"]]
            .drop_duplicates()
            .set_index("household_id")[["address_id"]]
        )

        max_household_address_id = households["address_id"].max()

        (pop, self.households, _,) = update_address_id_for_unit_and_sims(
            pop,
            moving_units=self.households,
            units_that_move_ids=movers["household_id"],
            starting_address_id=max_household_address_id + 1,
            unit_id_col_name="household_id",
            address_id_col_name="address_id",
        )

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def generate_initial_households(self, pop_data: SimulantData) -> pd.DataFrame():
        households = self.population_view.subview(["household_id", "housing_type"]).get(
            pop_data.index
        )
        households = (
            households.drop_duplicates().sort_values("household_id").set_index("household_id")
        )
        households["address_id"] = np.arange(len(households))

        return households

    def generate_household_details(self, idx: pd.Index) -> pd.DataFrame:
        pop = self.population_view.subview(["household_id"]).get(idx)
        household_details = pop.join(
            self.households,
            on="household_id",
        )
        # FIXME: why is the `housing_type` column losing its CategoricalDtype?
        # Set housing_type dtypes
        household_details["housing_type"] = household_details["housing_type"].astype(
            pd.CategoricalDtype(categories=HOUSING_TYPES)
        )

        return household_details
