import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.utilities import from_yearly

from vivarium_census_prl_synth_pop.constants import data_keys, data_values


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
        persons_data = builder.data.load(data_keys.POPULATION.PERSONS)

        immigrants = persons_data[persons_data["immigrated_in_last_year"]]

        is_gq = immigrants["relation_to_household_head"].isin(
            [
                "Insitutionalized GQ pop",
                "Noninstitutionalized GQ pop",
            ]
        )
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
        households_data = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)
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
            # The ACS weight is interpretable as the number of people represented
            # by these rows -- that is, the number of immigrants (of this type) in the last year
            # there were on average between 2016-2020.
            immigrants["person_weight"].sum()
            # Rescale to the proportion of the US population being simulated
            * (configuration.population.population_size / data_values.US_POPULATION)
        )
        return from_yearly(
            immigrants_per_year, pd.Timedelta(days=configuration.time.step_size)
        )
