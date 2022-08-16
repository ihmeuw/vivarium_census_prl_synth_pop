from typing import Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

from vivarium_census_prl_synth_pop.constants import metadata
from vivarium_census_prl_synth_pop.constants import data_values


class PersonMigration:
    """
    - needs to be able to update a household_id (to an existing id)
    - needs to be able to update a relationship to head of household (to other nonrelative)
    - on_time_step, needs to be able to look up probability of moving and determine if moving
    - needs to be able to choose a new household at random
    ASSUMPTIONS:
    - head of household never moves to new household_id
    """

    def __repr__(self) -> str:
        return "PersonMigration()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "person_migration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_needed = [
            "household_id",
            "relation_to_household_head",
            "address",
            "zipcode",
        ]
        self.population_view = builder.population.get_view(self.columns_needed)
        move_rate_data = builder.lookup.build_table(data_values.INDIVIDUAL_MOVE_RATE_YEARLY)
        self.person_move_rate = builder.value.register_rate_producer(
            f"{self.name}.move_rate", source=move_rate_data
        )

        builder.event.register_listener("time_step", self.on_time_step)

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        """
        Determines which simulants will move to a new household
        Moves those simulants to a new household_id
        Assigns those simulants relationship to head of household 'Other nonrelative'
        """
        persons = self.population_view.get(event.index)
        non_household_heads = persons.loc[
            persons.relation_to_household_head != "Reference person"
        ]
        persons_who_move = self.randomness.filter_for_rate(
            non_household_heads, self.person_move_rate(non_household_heads.index)
        )
        new_households = self._get_new_household_ids(persons_who_move, event)

        # get address and zipcode corresponding to selected households
        new_household_data = (
            self.population_view.subview(["household_id", "address", "zipcode"])
            .get(index=event.index)
            .drop_duplicates()
        )
        new_household_data = new_household_data.loc[
            new_household_data.household_id.isin(new_households)
        ]

        # create map from household_ids to addresses and zipcodes
        new_household_data["household_id"] = new_household_data["household_id"].astype(int)
        new_household_data_map = new_household_data.set_index("household_id").to_dict()

        # update household data for persons who move
        persons_who_move["household_id"] = new_households
        persons_who_move["address"] = persons_who_move["household_id"].map(
            new_household_data_map["address"]
        )
        persons_who_move["zipcode"] = persons_who_move["household_id"].map(
            new_household_data_map["zipcode"]
        )
        persons_who_move["relation_to_household_head"] = "Other nonrelative"

        self.population_view.update(persons_who_move)

    ##################
    # Helper methods #
    ##################

    def _get_new_household_ids(
        self, persons_who_move: pd.DataFrame, event: Event
    ) -> pd.Series:
        households = self.population_view.subview(["household_id"]).get(event.index)
        all_household_ids = list(households.squeeze().drop_duplicates())

        new_household_ids = persons_who_move["household_id"].copy()
        additional_seed = 0
        while (new_household_ids == persons_who_move.household_id).any():
            unchanged_households = new_household_ids == persons_who_move.household_id
            new_household_ids[unchanged_households] = self.randomness.choice(
                new_household_ids.loc[unchanged_households].index,
                all_household_ids,
                additional_key=additional_seed,
            )
            additional_seed += 1

        return pd.Series(new_household_ids)
