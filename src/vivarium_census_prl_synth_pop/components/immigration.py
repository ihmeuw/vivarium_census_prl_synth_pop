import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.utilities import from_yearly

from vivarium_census_prl_synth_pop.constants import data_keys


class Immigration:
    """
    Handles migration of individuals *into* the US.
    """

    def __repr__(self) -> str:
        return "Immigration()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "immigration"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        # The order of setup is not guaranteed between components.
        # Whether this function runs before or after the Population component's
        # setup is completely random.
        # They both need the same (large) ACS dataset to do their work, and Population
        # needs to keep it all in memory until population initialization.
        # This component only needs to keep a small subset of the data (immigrants), and it keeps
        # it for the entire runtime of the simulation.

        # The solution: if Population goes first, we reuse the full data it already
        # has in memory.
        # If this component goes first, we load it here and subset it, and then the whole
        # thing is loaded again by Population.
        # Therefore, the full dataset is never loaded in memory in two places at once.
        population = builder.components.get_component("population")
        if population.population_data is not None:
            persons_data = population.population_data["persons"]
            households_data = population.population_data["households"]
        else:
            persons_data = builder.data.load(data_keys.POPULATION.PERSONS)
            households_data = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)

        self.total_person_weight = persons_data["person_weight"].sum()

        immigrants = persons_data[persons_data["immigrated_in_last_year"]]

        gq_households = households_data[households_data["household_type"] != "Housing unit"]
        is_gq = immigrants["census_household_id"].isin(gq_households["census_household_id"])
        self.gq_immigrants = immigrants[is_gq]
        self.gq_immigrants_per_time_step = self._immigrants_per_time_step(
            self.gq_immigrants,
            builder.configuration,
        )

        non_gq_immigrants = immigrants[~is_gq]
        immigrant_reference_people = non_gq_immigrants[
            non_gq_immigrants["relation_to_household_head"] == "Reference person"
        ]

        is_household_immigrant = non_gq_immigrants["census_household_id"].isin(
            immigrant_reference_people["census_household_id"]
        )

        self.household_immigrants = non_gq_immigrants[is_household_immigrant]
        self.household_immigrants_per_time_step = self._immigrants_per_time_step(
            self.household_immigrants,
            builder.configuration,
        )
        self.non_reference_person_immigrants = non_gq_immigrants[~is_household_immigrant]
        self.non_reference_person_immigrants_per_time_step = self._immigrants_per_time_step(
            self.non_reference_person_immigrants,
            builder.configuration,
        )

        # Get the *household* (not person) weights for each household that can immigrate
        # in a household move, for use in sampling.
        self.immigrant_household_weights = households_data.set_index(
            "census_household_id"
        ).loc[
            immigrant_reference_people["census_household_id"],
            "household_weight",
        ]

    ##################
    # Helper methods #
    ##################

    def _immigrants_per_time_step(self, immigrants, configuration):
        immigrants_per_year = (
            # We rescale the proportion between immigrant population and total population to the
            # simulation's initial population size.
            # This value will not change over time during the simulation.
            (immigrants["person_weight"].sum() / self.total_person_weight)
            * configuration.population.population_size
        )
        return from_yearly(
            immigrants_per_year, pd.Timedelta(days=configuration.time.step_size)
        )
