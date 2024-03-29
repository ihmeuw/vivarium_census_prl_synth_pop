from typing import List

import pandas as pd
from loguru import logger
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RESIDUAL_CHOICE
from vivarium.framework.utilities import from_yearly

from vivarium_census_prl_synth_pop.constants import data_values, metadata, paths


class PersonMigration(Component):
    """
    Handles domestic (within the US) migration of individuals (not in household groups).

    There are three types of individual moves in domestic migration:
    - New-household moves, where the individual establishes a new single-person household.
    - GQ person moves, where the individual moves into a GQ type.
    - Non-reference-person moves, where the individual joins an existing household.

    Note that the names of these move types refer to the living situation *after* the move
    is complete.
    The population at risk for all three move types is all simulants.
    """

    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> List[str]:
        return [
            "household_id",
            "relationship_to_reference_person",
        ]

    @property
    def columns_created(self) -> List[str]:
        return ["previous_timestep_address_id"]

    @property
    def time_step_priority(sself) -> int:
        return metadata.PRIORITY_MAP["person_migration.on_time_step"]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.households = builder.components.get_component("households")
        self.household_details = builder.value.get_value("household_details")
        self.updated_relationship_to_reference_person = builder.value.get_value(
            "updated_relationship_to_reference_person"
        )

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

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                "previous_timestep_address_id": -1,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step_prepare(self, event: Event) -> None:
        pop = self.population_view.subview(["previous_timestep_address_id"]).get(event.index)
        pop["previous_timestep_address_id"] = self.household_details(pop.index)[
            ["address_id"]
        ]
        self.population_view.update(pop)

    def on_time_step(self, event: Event) -> None:
        """
        Determines which simulants will move with which move type
        and performs those move operations.
        """
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",
        )

        # Get subsets of simulants that do each move type on this timestep
        move_type_probabilities = self.person_move_probabilities(pop.index)
        move_types_chosen = self.randomness.choice(
            pop.index,
            choices=move_type_probabilities.columns,
            p=move_type_probabilities.values,
            additional_key="move_types",
        )

        pop = self._perform_new_household_moves(
            pop, pop.index[move_types_chosen == "new_household"]
        )

        pop = self._perform_gq_person_moves(pop, pop.index[move_types_chosen == "gq_person"])

        pop = self._perform_non_reference_person_moves(
            pop, pop.index[move_types_chosen == "non_reference_person"]
        )

        self.population_view.update(pop)

        new_relationship_to_reference_person = self.updated_relationship_to_reference_person(
            event.index
        )
        self.population_view.update(
            new_relationship_to_reference_person.rename("relationship_to_reference_person")
        )

    ##################
    # Helper methods #
    ##################

    def _perform_new_household_moves(
        self, pop: pd.DataFrame, movers: pd.Index
    ) -> pd.DataFrame:
        """
        Create a new single-person household for each person in movers and move them to it.
        """
        new_household_ids = self.households.create_households(len(movers))

        # update pop table
        pop.loc[movers, "household_id"] = new_household_ids
        pop.loc[movers, "relationship_to_reference_person"] = "Reference person"

        return pop

    def _perform_gq_person_moves(self, pop: pd.DataFrame, movers: pd.Index) -> pd.DataFrame:
        """
        Move each simulant in movers to a random GQ type.
        """
        if len(movers) == 0:
            return pop

        # The two GQ housing type categories (institutional, non-institutional) are
        # tracked in the "relationship_to_reference_person" column, even though
        # that column name doesn't really make sense in a GQ setting.
        categories = list(data_values.GROUP_QUARTER_IDS.keys())
        housing_type_category_values = self.randomness.choice(
            movers,
            choices=categories,
            additional_key="gq_person_move_housing_type_categories",
        )
        pop.loc[movers, "relationship_to_reference_person"] = housing_type_category_values

        for category, housing_types in data_values.GROUP_QUARTER_IDS.items():
            movers_in_category = movers.intersection(
                pop.index[pop["relationship_to_reference_person"] == category]
            )
            if len(movers_in_category) == 0:
                continue

            # NOTE: In rare cases, simulants will "move" to the same GQ type they are
            # already living in.
            # We allow this -- the underlying rate data from ACS is the rate of any
            # move in the last year, even e.g. between nursing homes.
            # Of course in our case it's a bit weird because there is only one household/
            # address for nursing homes, but that isn't a migration-specific issue.
            household_id_values = self.randomness.choice(
                movers_in_category,
                choices=list(housing_types.values()),
                additional_key="gq_person_move_household_ids",
            )

            pop.loc[movers_in_category, "household_id"] = household_id_values

        return pop

    def _perform_non_reference_person_moves(
        self, pop: pd.DataFrame, movers: pd.Index
    ) -> pd.DataFrame:
        """
        Move each simulant in movers to a random (different) non-GQ and non-empty
        household with relationship "Other nonrelative".
        """
        if len(movers) == 0:
            return pop

        # those households that contain non-movers
        non_gq_pop = pop[~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)]
        households_with_non_movers = non_gq_pop.loc[
            non_gq_pop.index.difference(movers), "household_id"
        ].drop_duplicates()

        # NOTE: Unlike in GQ person moves, we require that a "move"
        # not start and end in the same household.
        to_move = movers
        household_id_values = pop.loc[movers, "household_id"]

        # There is an edge case in very small populations where there are no households
        # to move to; in that case, the move is not performed
        if len(households_with_non_movers) > 0:
            for iteration in range(10):
                if to_move.empty:
                    break

                household_id_values[to_move] = self.randomness.choice(
                    to_move,
                    choices=households_with_non_movers,
                    additional_key=f"non_reference_person_move_household_ids_{iteration}",
                )
                to_move = movers[household_id_values == pop.loc[movers, "household_id"]]

        if len(to_move) > 0:
            logger.info(
                f"Maximum iterations for resampling of household_id reached. "
                f"The number of simulants whose household_id was not changed despite being "
                f"selected for a non-reference-person move is {len(to_move)}"
            )

        pop.loc[movers, "household_id"] = household_id_values
        pop.loc[movers, "relationship_to_reference_person"] = "Other nonrelative"

        return pop
