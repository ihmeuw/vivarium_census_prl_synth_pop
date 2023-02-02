import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.time import get_time_stamp

from vivarium_census_prl_synth_pop.constants import metadata, paths


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
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.households = builder.components.get_component("households")

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
            "relation_to_household_head",
        ]

        self.population_view = builder.population.get_view(self.columns_used)

        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            priority=metadata.PRIORITY_MAP["household_migration.on_time_step"],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event):
        """
        choose which households move;
        move those households to a new address
        """
        pop = self.population_view.get(event.index, query="alive == 'alive'")
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

        self.households.update_household_addresses(movers["household_id"])
