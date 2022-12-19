import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.randomness import RESIDUAL_CHOICE
from vivarium.framework.utilities import from_yearly
from vivarium.framework.values import Pipeline

from vivarium_census_prl_synth_pop.constants import data_values, paths
from vivarium_census_prl_synth_pop.utilities import update_address_id


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
            "address_id",
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
            additional_key="move_types",
        )

        pop = self._perform_new_household_moves(
            pop, pop.index[move_types_chosen == "new_household"]
        )

        pop = self._perform_gq_person_moves(pop, pop.index[move_types_chosen == "gq_person"])

        # TODO: This old code currently handles all of the *other* types of domestic moves.
        domestic_movers_idx = pop.index[
            ~move_types_chosen.isin(["no_move", "new_household", "gq_person"])
        ]

        # TODO: Handle move types correctly -- this is code from the previous migration implementation
        # which only had a single type of individual move.

        # Get series of new household_ids the domestic_movers_idx will move to
        new_households = self._get_new_household_ids(pop, domestic_movers_idx)

        # get address_id for new households being moved to
        new_household_data = (
            pop[["household_id", "address_id"]].drop_duplicates().set_index("household_id")
        )

        # update household data for domestic movers
        pop.loc[domestic_movers_idx, "household_id"] = new_households
        # Get map for new_address_ids and assign new address_id
        pop = pop.reset_index().rename(
            columns={"index": "simulant_id"}
        )  # Preserve index in merge
        new_address_ids = (
            pop[["simulant_id", "household_id"]]
            .merge(
                new_household_data[["address_id"]],
                how="left",
                left_on="household_id",
                right_on=new_household_data.index,
            )
            .set_index("simulant_id")["address_id"]
        )
        pop = pop.set_index("simulant_id")
        pop.loc[domestic_movers_idx, "address_id"] = new_address_ids

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

    def _perform_new_household_moves(
        self, pop: pd.DataFrame, movers: pd.Index
    ) -> pd.DataFrame:
        """
        Create a new single-person household for each person in movers and move them to it.
        """
        first_new_household_id = pop["household_id"].max() + 1
        first_new_household_address_id = pop["address_id"].max() + 1

        new_household_ids = first_new_household_id + np.arange(len(movers))

        pop.loc[movers, "household_id"] = new_household_ids
        pop = update_address_id(
            pop, movers, starting_address_id=first_new_household_address_id
        )

        pop.loc[movers, "relation_to_household_head"] = "Reference person"
        pop.loc[movers, "housing_type"] = "Standard"

        return pop

    def _perform_gq_person_moves(self, pop: pd.DataFrame, movers: pd.Index) -> pd.DataFrame:
        """
        Move each simulant in movers to a random GQ type.
        """
        if len(movers) == 0:
            return pop

        # The two GQ housing type categories (institutional, non-institutional) are
        # tracked in the "relation_to_household_head" column, even though
        # that column name doesn't really make sense in a GQ setting.
        categories = list(data_values.GROUP_QUARTER_IDS.keys())
        gq_address_ids = (
            pop[pop["relation_to_household_head"].isin(categories)][
                ["household_id", "address_id"]
            ]
            .drop_duplicates()
            .set_index("household_id")["address_id"]
        )
        assert gq_address_ids.index.is_unique

        housing_type_category_values = self.randomness.choice(
            movers,
            choices=categories,
            additional_key="gq_person_move_housing_type_categories",
        )
        pop.loc[movers, "relation_to_household_head"] = housing_type_category_values

        for category, housing_types in data_values.GROUP_QUARTER_IDS.items():
            movers_in_category = movers.intersection(
                pop.index[pop["relation_to_household_head"] == category]
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
            pop.loc[movers_in_category, "housing_type"] = household_id_values.map(
                data_values.GQ_HOUSING_TYPE_MAP
            )
            pop.loc[movers_in_category, "address_id"] = household_id_values.map(
                gq_address_ids
            )

        return pop

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
