import pandas as pd
from vivarium.framework.engine import Builder
from vivarium_public_health import utilities
from vivarium_public_health.population.add_new_birth_cohorts import FertilityAgeSpecificRates, PREGNANCY_DURATION


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
        return 'Fertility()'

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "fertility"

    ##############
    ##############

    def on_initialize_simulants(self, pop_data):
        """Adds 'last_birth_time' and 'parent' columns to the state table."""
        pop = self.population_view.subview(["sex"]).get(pop_data.index)
        women = pop.loc[pop.sex == "Female"].index

        if pop_data.user_data["sim_state"] == "setup":
            parent_id = -1
        else:  # 'sim_state' == 'time_step'
            parent_id = pop_data.user_data["parent_ids"]
        pop_update = pd.DataFrame(
            {"last_birth_time": pd.NaT, "parent_id": parent_id}, index=pop_data.index
        )
        # FIXME: This is a misuse of the column and makes it invalid for
        #    tracking metrics.
        # Do the naive thing, set so all women can have children
        # and none of them have had a child in the last year.
        pop_update.loc[women, "last_birth_time"] = pop_data.creation_time - pd.Timedelta(
            days=utilities.DAYS_PER_YEAR
        )

        self.population_view.update(pop_update)

    def on_time_step(self, event):
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
        can_have_children = population.last_birth_time < nine_months_ago
        eligible_women = population[can_have_children]

        rate_series = self.fertility_rate(eligible_women.index)
        had_children = self.randomness.filter_for_rate(eligible_women, rate_series).copy()

        # decide which births twins
        twins_probability = [self.probability_of_twins]*len(had_children)
        if len(had_children) > 0:
            had_twins = self.randomness.filter_for_probability(had_children, twins_probability)
            had_children_incl_twins = pd.concat([had_children, had_twins])

        had_children.loc[:, "last_birth_time"] = event.time
        self.population_view.update(had_children["last_birth_time"])

        # If children were born, add them to the state table and record
        # who their mother was.
        num_babies = len(had_children_incl_twins)
        if num_babies:
            self.simulant_creator(
                num_babies,
                population_configuration={
                    "age_start": 0,
                    "age_end": 0,
                    "sim_state": "time_step",
                    "parent_ids": had_children_incl_twins.index,
                },
            )

    def load_age_specific_fertility_rate_data(self, builder):
        asfr_data = builder.data.load("covariate.age_specific_fertility_rate.estimate")
        columns = ["year_start", "year_end", "age_start", "age_end", "value"]
        asfr_data = asfr_data.loc[asfr_data.sex == "Female"][columns]
        return asfr_data
