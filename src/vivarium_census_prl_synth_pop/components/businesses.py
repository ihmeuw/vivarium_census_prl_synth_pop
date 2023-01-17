from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health import utilities

from vivarium_census_prl_synth_pop.constants import data_values, metadata, paths
from vivarium_census_prl_synth_pop.utilities import (
    filter_by_rate,
    update_address_id_for_unit_and_sims,
)


class Businesses:
    """
    IMPROVE DESCRIPTION

    on init:
        assign everyone of working age an employer

    on timestep:
        new job if turning working age
        change jobs at rate of 50 changes per 100 person years

    FROM ABIE:  please use a skewed distribution for the business sizes:
    np.random.lognormal(4, 1) for now, and I'll come up with something better in the future.
    # people = # businesses * E[people per business]
    NOTE: there will be a fixed number of businesses over the course of the simulation.
    their addresses will not change in this ticket.
    """

    def __repr__(self) -> str:
        return "Businesses()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "businesses"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.randomness = builder.randomness.get_stream(self.name)
        self.employer_address_id_count = 0
        self.columns_created = [
            "employer_id",
            "personal_income_propensity",
            "employer_income_propensity",
        ]
        self.columns_used = [
            "address_id",
            "previous_timestep_address_id",
            "age",
            "household_id",
        ] + self.columns_created
        self.population_view = builder.population.get_view(self.columns_used)
        self.businesses = None

        job_change_rate_data = builder.lookup.build_table(data_values.YEARLY_JOB_CHANGE_RATE)
        self.job_change_rate = builder.value.register_rate_producer(
            f"{self.name}.job_change_rate", source=job_change_rate_data
        )

        move_rate_data = builder.lookup.build_table(data_values.BUSINESS_MOVE_RATE_YEARLY)
        self.businesses_move_rate = builder.value.register_rate_producer(
            f"{self.name}.move_rate", source=move_rate_data
        )
        self.income_distributions_data = builder.lookup.build_table(
            data=pd.read_csv(paths.INCOME_DISTRIBUTIONS_DATA_PATH),
            key_columns=["sex", "race_ethnicity"],
            parameter_columns=["age"],
            value_columns=["s", "scale"],
        )
        self.income = builder.value.register_value_producer(
            "income",
            source=self.calculate_income,
            requires_columns=["personal_income_propensity", "employer_income_propensity"],
        )
        self.business_details = builder.value.register_value_producer(
            "business_details",
            source=self.generate_business_details,
            requires_columns=["employer_id"],
        )

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=["age"],
            creates_columns=self.columns_created,
        )
        # note: priority must be later than that of migration components
        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            priority=metadata.PRIORITY_MAP["businesses.on_time_step"],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Assign everyone working age and older an employer
        """
        if pop_data.creation_time < self.start_time:
            self.businesses = self.generate_businesses(pop_data)
            pop = self.population_view.subview(["age", "household_id"]).get(pop_data.index)
            pop["employer_id"] = data_values.Unemployed.EMPLOYER_ID
            working_age = pop.loc[pop["age"] >= data_values.WORKING_AGE].index
            pop.loc[working_age, "employer_id"] = self.assign_random_employer(working_age)

            # Give military gq sims military employment
            military_index = pop.loc[
                (
                    pop["household_id"]
                    == data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS["Military"]
                )
                & (pop["age"] >= data_values.WORKING_AGE)
            ].index
            if not military_index.empty:
                pop.loc[
                    military_index, "employer_id"
                ] = data_values.MilitaryEmployer.EMPLOYER_ID
            self.population_view.update(pop)
        else:
            pop = self.population_view.get(pop_data.index)
            pop["employer_id"] = data_values.Unemployed.EMPLOYER_ID

        # Create income propensity columns
        pop["personal_income_propensity"] = self.randomness.get_draw(
            pop.index,
            additional_key="personal_income_propensity",
        )
        pop["employer_income_propensity"] = self.randomness.get_draw(
            pop.index,
            additional_key="employer_income_propensity",
        )

        self.population_view.update(pop)

    def on_time_step(self, event: Event):
        """
        assign job if turning working age
        change jobs at rate of 50 changes / 100 person-years
        businesses change addresses at rate of 10 changes / 100 person-years
        """
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",
        )
        all_businesses = self.businesses.index[
            self.businesses.index != data_values.Unemployed.EMPLOYER_ID
        ]
        businesses_that_move_idx = filter_by_rate(
            all_businesses,
            self.randomness,
            self.businesses_move_rate,
            "moving_businesses",
        )

        (
            _,
            self.businesses,
            self.employer_address_id_count,
        ) = update_address_id_for_unit_and_sims(
            pop,
            moving_units=self.businesses,
            units_that_move_ids=businesses_that_move_idx,
            starting_address_id=self.employer_address_id_count,
            unit_id_col_name="employer_id",
            address_id_col_name="employer_address_id",
        )

        # change jobs if of working age already but exclude simulants living in military group quarters
        working_age_idx = pop.index[
            (pop["age"] >= data_values.WORKING_AGE)
            & (
                pop["household_id"]
                != data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS["Military"]
            )
        ]
        changing_jobs_idx = filter_by_rate(
            working_age_idx, self.randomness, self.job_change_rate, "changing jobs"
        )
        moved_idx = pop.index[pop["address_id"] != pop["previous_timestep_address_id"]]
        moved_working_age_idx = moved_idx.intersection(working_age_idx)
        changing_jobs_idx = changing_jobs_idx.union(moved_working_age_idx)
        if len(changing_jobs_idx) > 0:
            pop.loc[changing_jobs_idx, "employer_id"] = self.assign_different_employer(
                changing_jobs_idx
            )

        # assign job if turning working age
        turning_working_age = pop.loc[
            (
                pop["age"]
                >= data_values.WORKING_AGE - event.step_size.days / utilities.DAYS_PER_YEAR
            )
            & (pop["age"] < data_values.WORKING_AGE)
        ].index
        if len(turning_working_age) > 0:
            pop.loc[turning_working_age, "employer_id"] = self.assign_random_employer(
                turning_working_age
            )

        # Give military gq sims military employment
        new_military_idx = pop.index[
            (
                pop["household_id"]
                == data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS["Military"]
            )
            & (pop["age"] >= data_values.WORKING_AGE)
            & (pop["employer_id"] != data_values.MilitaryEmployer.EMPLOYER_ID)
        ]
        if len(new_military_idx) > 0:
            pop.loc[
                new_military_idx, "employer_id"
            ] = data_values.MilitaryEmployer.EMPLOYER_ID

        # Update income
        # Get new income propensity and update income for simulants who have new employers or joined the workforce
        employment_changing_sims_idx = changing_jobs_idx.union(turning_working_age).union(
            new_military_idx
        )
        pop.loc[
            employment_changing_sims_idx, "employer_income_propensity"
        ] = self.randomness.get_draw(
            employment_changing_sims_idx,
            additional_key="employer_income_propensity",
        )

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def generate_businesses(self, pop_data: SimulantData) -> pd.DataFrame():
        pop = self.population_view.subview(["age"]).get(pop_data.index)
        n_working_age = len(pop.loc[pop["age"] >= data_values.WORKING_AGE])

        # TODO: when have more known employers, maybe move to csv
        known_employers = pd.DataFrame(
            {
                "employer_id": [
                    data_values.Unemployed.EMPLOYER_ID,
                    data_values.MilitaryEmployer.EMPLOYER_ID,
                ],
                "employer_name": [
                    data_values.Unemployed.EMPLOYER_NAME,
                    data_values.MilitaryEmployer.EMPLOYER_NAME,
                ],
                "employer_address_id": [
                    data_values.Unemployed.EMPLOYER_ADDRESS_ID,
                    data_values.MilitaryEmployer.EMPLOYER_ADDRESS_ID,
                ],
                "prevalence": [
                    1 - data_values.PROPORTION_WORKFORCE_EMPLOYED,
                    data_values.MilitaryEmployer.PROPORTION_WORKFORCE_EMPLOYED,
                ],
            }
        )

        pct_adults_needing_employers = 1 - known_employers["prevalence"].sum()
        n_need_employers = np.round(n_working_age * pct_adults_needing_employers)
        employee_counts = np.random.lognormal(
            4, 1, size=int(n_need_employers // data_values.EXPECTED_EMPLOYEES_PER_BUSINESS)
        ).round()
        n_businesses = len(employee_counts)
        random_employers = pd.DataFrame(
            {
                "employer_id": np.arange(2, n_businesses + 2),
                "employer_name": ["not implemented"] * n_businesses,
                "prevalence": employee_counts
                / employee_counts.sum()
                * pct_adults_needing_employers,
                "employer_address_id": np.arange(
                    len(known_employers), n_businesses + len(known_employers)
                ),
            }
        )

        businesses = pd.concat([known_employers, random_employers], ignore_index=True)
        self.employer_address_id_count = len(businesses)  # So next address will be unique

        return businesses.set_index("employer_id")

    def assign_random_employer(
        self, sim_index: pd.Index, additional_seed: Any = None
    ) -> pd.Series:
        return self.randomness.choice(
            index=sim_index,
            choices=self.businesses.index,
            p=self.businesses["prevalence"],
            additional_key=additional_seed,
        )

    def assign_different_employer(self, changing_jobs: pd.Index) -> pd.Series:
        current_employers = self.population_view.subview(["employer_id"]).get(changing_jobs)[
            "employer_id"
        ]

        new_employers = current_employers.copy()
        additional_seed = 0
        while (current_employers == new_employers).any():
            unchanged_employers = current_employers == new_employers
            new_employers[unchanged_employers] = self.assign_random_employer(
                sim_index=new_employers[unchanged_employers].index,
                additional_seed=additional_seed,
            )
            additional_seed += 1

        return new_employers

    def calculate_income(self, idx: pd.Index) -> pd.Series:

        income = pd.Series(0.0, index=idx)
        pop = self.population_view.get(idx)
        employed_idx = pop.index[pop["employer_id"] != data_values.Unemployed.EMPLOYER_ID]

        # Get propensities for two components to get income propensity
        personal_income_component = data_values.PERSONAL_INCOME_PROPENSITY_DISTRIBUTION.ppf(
            pop.loc[employed_idx, "personal_income_propensity"]
        )
        employer_income_component = data_values.EMPLOYER_INCOME_PROPENSITY_DISTRIBUTION.ppf(
            pop.loc[employed_idx, "employer_income_propensity"]
        )
        # Income propensity = cdf(personal_component + employer_component)
        income_propensity = pd.Series(
            data=stats.norm.cdf(personal_income_component + employer_income_component),
            index=employed_idx,
        )
        # Get distributions from lookup table
        income_distributions = self.income_distributions_data(employed_idx)

        income[employed_idx] = stats.lognorm.ppf(
            q=income_propensity,
            s=income_distributions["s"],
            scale=income_distributions["scale"],
        )

        return income

    def generate_business_details(self, idx: pd.Index) -> pd.DataFrame:
        pop = self.population_view.subview(["employer_id"]).get(idx)
        business_details = pop.join(
            self.businesses[["employer_name", "employer_address_id"]], on="employer_id"
        )

        return business_details
