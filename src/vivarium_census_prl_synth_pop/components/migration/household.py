import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants import metadata, data_keys
from vivarium_census_prl_synth_pop.constants import data_values


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

        move_rate_data = builder.lookup.build_table(data_values.HOUSEHOLD_MOVE_RATE_YEARLY)
        self.household_move_rate = builder.value.register_rate_producer(
            f"{self.name}.move_rate", source=move_rate_data
        )

        self.randomness = builder.randomness.get_stream(self.name)
        self.addresses = builder.components.get_component('Addresses')
        self.columns_created = ["address", "zipcode"]
        self.columns_used = ["household_id", "address", "zipcode", "tracked"]
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
            households = self.population_view.subview(["household_id", "tracked"]).get(
                pop_data.index
            )
            address_assignments = self.addresses.generate(
                pd.Index(households["household_id"].drop_duplicates()),
                state=metadata.US_STATE_ABBRV_MAP[self.location].lower(),
            )
            households["address"] = households["household_id"].map(
                address_assignments["address"]
            )
            households["zipcode"] = households["household_id"].map(
                address_assignments["zipcode"]
            )

            # handle untracked sims
            households.loc[households.household_id == 'NA', 'address'] = 'NA'
            households.loc[households.household_id == 'NA', 'zipcode'] = 'NA'

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
        households = self.population_view.subview(["household_id", "address", "zipcode"]).get(
            event.index
        )
        households_that_move = self.addresses.determine_if_moving(
            households["household_id"], self.household_move_rate
        ).index

        address_map, zipcode_map = self.addresses.get_new_addresses_and_zipcodes(
            households_that_move, state=metadata.US_STATE_ABBRV_MAP[self.location].lower()
        )

        households = self.addresses.update_address_and_zipcode(
            df=households,
            rows_to_update=households_that_move,
            address_map=address_map,
            zipcode_map=zipcode_map,
        )
        self.population_view.update(households)
