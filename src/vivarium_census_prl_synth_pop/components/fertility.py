import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium_public_health.population.add_new_birth_cohorts import (
    PREGNANCY_DURATION,
    FertilityAgeSpecificRates,
)

from vivarium_census_prl_synth_pop.constants import data_values


class Fertility(FertilityAgeSpecificRates):
    """
    - On each timestep, children will be born according to ASFR.
    - Each birth will be twins with probability X
    - Each child will inherit from their mother:
        - race/ethnicity
        - household id
        - geography attribute
        - street address
        - surname
    """

    def __repr__(self) -> str:
        return "Fertility()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "fertility"

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event):
        """Produces new children and updates parent status on time steps.
        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        """
        # Get a view on all living women who haven't had a child in at least nine months.
        nine_months_ago = pd.Timestamp(event.time - PREGNANCY_DURATION)
        # TODO: not only females can give birth
        population = self.population_view.get(
            event.index, query='alive == "alive" and sex =="Female"'
        )
        eligible_women = population[population.last_birth_time < nine_months_ago]

        rate_series = self.fertility_rate(eligible_women.index)
        had_children = self.randomness.filter_for_rate(eligible_women, rate_series).copy()

        had_children.loc[:, "last_birth_time"] = event.time
        self.population_view.update(had_children["last_birth_time"])

        # decide which births are twins
        twins_probability = [data_values.PROBABILITY_OF_TWINS] * len(had_children)
        had_twins = self.randomness.filter_for_probability(
            had_children, twins_probability, additional_key=event.time
        )
        had_children = pd.concat([had_children, had_twins])

        # If children were born, add them to the state table and record
        # who their mother was.
        num_babies = len(had_children)
        if num_babies:
            self.simulant_creator(
                num_babies,
                population_configuration={
                    "age_start": 0,
                    "age_end": 0,
                    "sim_state": "time_step",
                    "parent_ids": had_children.index,
                    "current_population_index": event.index
                },
            )

    ###########
    # Helpers #
    ###########

    def load_age_specific_fertility_rate_data(self, builder: Builder):
        asfr_data = builder.data.load("covariate.age_specific_fertility_rate.estimate")
        columns = ["year_start", "year_end", "age_start", "age_end", "value"]
        asfr_data = asfr_data.loc[asfr_data.sex == "Female"][columns]
        return asfr_data
