import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health import utilities

from vivarium_census_prl_synth_pop.constants import data_values


class Businesses:
    """
    IMPROVE DESCRIPTION

    on init:
        assign everyone 18 and up an employer

    on timestep:
        new job if turning 18
        change jobs at rate of 50 changes per 100 person years

    FROM ABIE:  please use a skewed distribution for the business sizes:
    np.random.lognormal(4, 1) for now, and I'll come up with something better in the future.
    # people = # businesses * E[people per business]
    NOTE: there will be a fixed number of businesses over the course of the simulation.
    their addresses will not change in this ticket.
    """

    def __repr__(self) -> str:
        return 'Businesses()'

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
        self.columns_created = ['employer_id', 'employer_name', 'employer_address']
        self.columns_used = ['age', 'tracked'] + self.columns_created
        self.population_view = builder.population.get_view(self.columns_used)
        self.businesses = None
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=['age'],
            creates_columns=self.columns_created
        )
        builder.event.register_listener("time_step", self.on_time_step)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Assign everyone 18 and older an employer
        """
        if pop_data.creation_time < self.start_time:
            self.businesses = self.generate_businesses(pop_data)

            pop = self.population_view.subview(['age', 'tracked']).get(pop_data.index)
            pop['employer_id'] = -1
            over_17 = pop.loc[pop.age >= data_values.WORKING_AGE].index
            pop.loc[over_17, 'employer_id'] = self.assign_random_employer(over_17)

            # merge on employer addresses and names
            pop = pop.merge(
                self.businesses[self.columns_created],
                on='employer_id',
                how='left'
            )

            # handle untracked sims
            pop.loc[pop.tracked == False, 'employer_id'] = -2
            pop.loc[pop.tracked == False, 'employer_name'] = 'NA'
            pop.loc[pop.tracked == False, 'employer_address'] = 'NA'
            self.population_view.update(
                pop
            )
        else:
            new_births = self.population_view.get(pop_data.index)

            new_births["employer_id"] = -1
            new_births["employer_name"] = 'unemployed'
            new_births["employer_address"] = 'NA'

            self.population_view.update(new_births)

    def on_time_step(self, event: Event):
        """
        assign job if turning 18
        change jobs at rate of 50 changes / 100 person-years
        """

        # change jobs
        pop = self.population_view.subview(self.columns_created + ['age']).get(event.index)
        employed = pop.loc[pop.employer_id > -1].index
        changing_jobs = self.randomness.filter_for_rate(
            employed,
            np.ones(len(employed))*(data_values.YEARLY_JOB_CHANGE_RATE * event.step_size.days / utilities.DAYS_PER_YEAR)
        )
        if len(changing_jobs) > 0:
            pop.loc[changing_jobs, "employer_id"] = self.assign_different_employer(changing_jobs)

            # add employer addresses and names
            pop.loc[changing_jobs, "employer_address"] = pop.loc[changing_jobs, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_address'].to_dict()
            )
            pop.loc[changing_jobs, "employer_name"] = pop.loc[changing_jobs, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_name'].to_dict()
            )

        # assign job if turning 18
        turned_18 = pop.loc[
            (pop.age >= data_values.WORKING_AGE - event.step_size.days / utilities.DAYS_PER_YEAR) &
            (pop.age < data_values.WORKING_AGE)
            ].index
        if len(turned_18) > 0:
            pop.loc[turned_18, 'employer_id'] = self.assign_random_employer(turned_18)

            # add employer addresses and names
            pop.loc[turned_18, "employer_address"] = pop.loc[turned_18, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_address'].to_dict()
            )
            pop.loc[turned_18, "employer_name"] = pop.loc[turned_18, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_name'].to_dict()
            )

        self.population_view.update(
            pop
        )

    ##################
    # Helper methods #
    ##################

    def generate_businesses(self, pop_data: SimulantData) -> pd.DataFrame():
        pop = self.population_view.subview(['age']).get(pop_data.index)
        over_17 = pop.loc[pop.age >= data_values.WORKING_AGE]

        n_employed = len(over_17)
        employee_counts = np.random.lognormal(
            4, 1, size=int(n_employed // data_values.EXPECTED_EMPLOYEES_PER_BUSINESS)
        ).round()
        n_businesses = len(employee_counts)
        businesses = pd.DataFrame({
            'employer_id': np.arange(n_businesses),
            'employer_name': ['not implemented']*n_businesses,
            'employer_address': ['not implemented']*n_businesses,
            'probability': employee_counts / employee_counts.sum(),
        })

        unemployed = pd.DataFrame({
            'employer_id': [-1],
            'employer_name': ['unemployed'],
            'employer_address': ['NA'],
            'probability': 0, #TODO: implement unemployment
        })

        untracked = pd.DataFrame({
            'employer_id': [-2],
            'employer_name': ['NA'],
            'employer_address': ['NA'],
            'probability': 0,
        })

        businesses = pd.concat([businesses, unemployed, untracked])
        return businesses

    def assign_random_employer(self, sim_index: pd.Index) -> pd.Series:
        return self.randomness.choice(
            index=sim_index,
            choices=self.businesses['employer_id'],
            p=self.businesses['probability']
        )

    def assign_different_employer(self, changing_jobs: pd.Index) -> pd.Series:
        current_employers = self.population_view.subview(['employer_id']).get(changing_jobs).squeeze()

        new_employers = current_employers.copy()
        additional_seed = 0
        while (current_employers == new_employers).any():
            unchanged_employers = (current_employers == new_employers)
            new_employers[unchanged_employers] = self.randomness.choice(
                new_employers[unchanged_employers].index,
                self.businesses['employer_id'].to_numpy(),
                additional_key=additional_seed
            )
            additional_seed += 1

        return new_employers
