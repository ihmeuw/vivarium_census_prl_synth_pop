from typing import Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.randomness import RESIDUAL_CHOICE
from vivarium.framework.utilities import from_yearly
from vivarium.framework.values import Pipeline

from vivarium_census_prl_synth_pop.components.synthetic_pii import (
    update_address_and_zipcode,
)
from vivarium_census_prl_synth_pop.constants import data_values, paths
from vivarium_census_prl_synth_pop.utilities import filter_by_rate


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

        move_rates_data = pd.read_csv(
            paths.INDIVIDUAL_DOMESTIC_MIGRATION_RATES_PATH,
        )

        value_columns = [
            "gq_person_migration_rate",
            "new_household_migration_rate",
            "non_reference_person_migration_rate",
        ]
        move_rates_data[value_columns] = from_yearly(
            move_rates_data[value_columns],
            pd.Timedelta(days=builder.configuration.time.step_size),
        )
        move_rates_data = move_rates_data.rename(
            columns=lambda c: c.replace("_migration_rate", "")
        ).assign(no_move=RESIDUAL_CHOICE)

        move_probabilities_lookup_table = builder.lookup.build_table(
            data=move_rates_data,
            key_columns=["sex", "race_ethnicity"],
            parameter_columns=["age"],
            value_columns=["gq_person", "new_household", "non_reference_person", "no_move"],
        )
        self.person_move_probabilities = builder.value.register_value_producer(
            f"{self.name}.move_probabilities",
            source=move_probabilities_lookup_table,
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

        pop = self.population_view.get(event.index)

        # Get subsets of simulants that do each move type on this timestep
        move_type_probabilities = self.person_move_probabilities(pop.index)
        move_types_chosen = self.randomness.choice(
            pop.index,
            choices=move_type_probabilities.columns,
            p=move_type_probabilities.values,
        )

        domestic_movers_idx = pop.index[move_types_chosen != "no_move"]

        # TODO: Handle move types correctly -- this is code from the previous migration implementation
        # which only had a single type of individual move.

        # Get series of new household_ids the domestic_movers_idx will move to
        new_households = self._get_new_household_ids(pop, domestic_movers_idx)

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
        pop.loc[domestic_movers_idx, "household_id"] = new_households
        pop = update_address_and_zipcode(
            df=pop,
            rows_to_update=domestic_movers_idx,
            id_key=new_households,
            address_map=new_household_data_map["address"],
            zipcode_map=new_household_data_map["zipcode"],
        )

        # update relation to head of household data
        pop.loc[domestic_movers_idx, "relation_to_household_head"] = "Other nonrelative"
        pop.loc[
            (pop.index.isin(domestic_movers_idx))
            & (
                pop["household_id"].isin(
                    data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS.values()
                )
            ),
            "relation_to_household_head",
        ] = "Noninstitutionalized GQ pop"
        pop.loc[
            (pop.index.isin(domestic_movers_idx))
            & (
                pop["household_id"].isin(data_values.INSTITUTIONAL_GROUP_QUARTER_IDS.values())
            ),
            "relation_to_household_head",
        ] = "Institutionalized GQ pop"

        # Update housing type
        pop.loc[
            (pop.index.isin(domestic_movers_idx))
            & (pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP.keys())),
            "housing_type",
        ] = pop["household_id"].map(data_values.GQ_HOUSING_TYPE_MAP)
        pop.loc[
            (pop.index.isin(domestic_movers_idx))
            & (~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP.keys())),
            "housing_type",
        ] = "Standard"

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def _get_new_household_ids(self, pop: pd.DataFrame, sims_who_move: pd.Index) -> pd.Series:
        households = pop["household_id"]
        all_household_ids = list(
            households.squeeze().drop_duplicates()
        )  # all household_ids in simulation

        new_household_ids = pop.loc[
            sims_who_move, "household_id"
        ].copy()  # People who move household_ids

        additional_seed = 0
        while (new_household_ids == pop.loc[sims_who_move, "household_id"]).any():
            unchanged_households = new_household_ids == pop.loc[sims_who_move, "household_id"]
            new_household_ids[unchanged_households] = self.randomness.choice(
                new_household_ids.loc[unchanged_households].index,
                all_household_ids,
                additional_key=f"newHousehold_ids_{additional_seed}",
            )
            # Some sims can move to the same house from randomness
            additional_seed += 1

        return pd.Series(new_household_ids)
