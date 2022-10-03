from typing import Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.values import Pipeline

from vivarium_census_prl_synth_pop.components.synthetic_pii import (
    update_address_and_zipcode,
)
from vivarium_census_prl_synth_pop.constants import data_values, paths


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
        self.addresses = builder.components.get_component("Address")
        self.columns_needed = [
            "household_id",
            "relation_to_household_head",
            "address",
            "zipcode",
            "exit_time",
            "tracked",
            "housing_type",
        ]
        self.population_view = builder.population.get_view(self.columns_needed)

        move_rate_data = builder.lookup.build_table(
            data=pd.read_csv(
                paths.HOUSEHOLD_MOVE_RATE_PATH,
                usecols=[
                    "sex",
                    "race_ethnicity",
                    "age_start",
                    "age_end",
                    "person_rate",
                    "housing_type",
                ],
            ),
            key_columns=["sex", "race_ethnicity", "housing_type"],
            parameter_columns=["age"],
            value_columns=["person_rate"],
        )
        self.person_move_rate = builder.value.register_rate_producer(
            f"{self.name}.move_rate", source=move_rate_data
        )
        proportion_simulants_leaving_country_data = builder.lookup.build_table(
            data=data_values.PROPORTION_PERSONS_LEAVING_COUNTRY
        )
        self.proportion_simulants_leaving_country = builder.value.register_rate_producer(
            "proportion_simulants_leaving_country",
            source=proportion_simulants_leaving_country_data,
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

        # Get subsets of possible simulants that can move
        persons_who_move = self.randomness.filter_for_rate(
            non_household_heads, self.person_move_rate(non_household_heads.index)
        )

        # Find simulants that move out of the country
        persons_who_move = self.move_simulants_out_of_country(
            persons_who_move, self.proportion_simulants_leaving_country, event
        )

        # Separate simulants that move abroad vs domestic
        abroad_movers = persons_who_move.loc[persons_who_move["exit_time"] == event.time]
        domestic_movers = persons_who_move.loc[
            ~persons_who_move.index.isin(abroad_movers.index)
        ]
        abroad_movers = abroad_movers.copy()
        domestic_movers = domestic_movers.copy()

        new_households = self._get_new_household_ids(domestic_movers, event)

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
        new_household_data_map = new_household_data.set_index("household_id")

        # update household data for domestic movers
        domestic_movers["household_id"] = new_households
        domestic_movers = update_address_and_zipcode(
            df=domestic_movers,
            rows_to_update=domestic_movers.index,
            id_key=domestic_movers["household_id"],
            address_map=new_household_data_map["address"],
            zipcode_map=new_household_data_map["zipcode"],
        )

        # update relation to head of household data
        domestic_movers["relation_to_household_head"] = "Other nonrelative"
        domestic_movers.loc[
            domestic_movers["household_id"].isin(
                data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS.values()
            ),
            "relation_to_household_head",
        ] = "Noninstitutionalized GQ pop"
        domestic_movers.loc[
            domestic_movers["household_id"].isin(
                data_values.INSTITUTIONAL_GROUP_QUARTER_IDS.values()
            ),
            "relation_to_household_head",
        ] = "Institutionalized GQ pop"

        # Update housing type
        domestic_movers.loc[
            domestic_movers["household_id"].isin(data_values.HOUSING_TYPE_MAP.keys()),
            "housing_type",
        ] = domestic_movers["household_id"].map(data_values.HOUSING_TYPE_MAP)
        domestic_movers.loc[
            ~domestic_movers["household_id"].isin(data_values.HOUSING_TYPE_MAP.keys()),
            "housing_type",
        ] = "Standard"

        simulants_who_moved = pd.concat(
            [
                domestic_movers,
                abroad_movers,
            ]
        )
        self.population_view.update(simulants_who_moved)

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

    def move_simulants_out_of_country(
        self,
        df_moving: pd.DataFrame,
        proportion_simulants_leaving_country: Pipeline,
        event: Event,
    ) -> pd.DataFrame:
        """
        df_moving: Subset of population that will be changing addresses this time step
        """
        sims_that_move_abroad = self.randomness.filter_for_probability(
            df_moving, proportion_simulants_leaving_country(df_moving.index)
        ).index
        if len(sims_that_move_abroad) > 0:
            df_moving.loc[sims_that_move_abroad, "exit_time"] = event.time
            df_moving.loc[sims_that_move_abroad, "tracked"] = False

        return df_moving
