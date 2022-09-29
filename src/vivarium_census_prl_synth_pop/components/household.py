import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.components.synthetic_pii import update_address_and_zipcode
from vivarium_census_prl_synth_pop.constants import metadata, data_keys, paths
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

        #TODO: consider subsetting to housing_type=="standard" rows if abie decides GQ never moves addresses
        move_rate_data = pd.read_csv(
                paths.HOUSEHOLD_MOVE_RATE_PATH,
                usecols=["sex", "race_ethnicity", "age_start", "age_end", "household_rate", "housing_type"],
            )
        move_rate_data = builder.lookup.build_table(
            data=move_rate_data.loc[move_rate_data["housing_type"] == "Standard"].drop(columns="housing_type"),
            key_columns=["sex", "race_ethnicity"],
            parameter_columns=["age"],
            value_columns=["household_rate"],
        )
        self.household_move_rate = builder.value.register_rate_producer(
            f"{self.name}.move_rate", source=move_rate_data
        )

        self.randomness = builder.randomness.get_stream(self.name)
        self.addresses = builder.components.get_component("Address")
        self.columns_created = ["address", "zipcode"]
        self.columns_used = [
            "household_id",
            "relation_to_household_head",
            "address",
            "zipcode",
            "tracked",
        ]
        self.population_view = builder.population.get_view(self.columns_used)

        proportion_households_leaving_country_data = builder.lookup.build_table(
            data=data_values.PROPORTION_HOUSEHOLDS_LEAVING_COUNTRY
        )
        self.proportion_households_leaving_country = builder.value.register_rate_producer(
            "proportion_households_leaving_country", source=proportion_households_leaving_country_data
        )

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
            households = self.population_view.subview(["household_id"]).get(
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
        households = self.population_view.subview(
            ["household_id", "relation_to_household_head", "address", "zipcode"]
        ).get(event.index)
        household_heads = households.loc[
            households["relation_to_household_head"] == "Reference person"
        ]
        households_that_move = self.addresses.determine_if_moving(
            household_heads["household_id"], self.household_move_rate
        )
        # todo: add in call to find households that leave country
        # todo: separate into moving abroad and domestic again
        if len(households_that_move) > 0:
            address_map, zipcode_map = self.addresses.get_new_addresses_and_zipcodes(
                households_that_move, state=metadata.US_STATE_ABBRV_MAP[self.location].lower()
            )

            households_to_update = households.loc[
                households["household_id"].isin(households_that_move), "household_id"
            ]

            households = update_address_and_zipcode(
                df=households,
                rows_to_update=households_to_update.index,
                id_key=households_to_update,
                address_map=address_map,
                zipcode_map=zipcode_map,
            )
            # todo: OR update households who move abroad to untracked here
            self.population_view.update(households)
