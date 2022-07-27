import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp

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

            all_sims = self.population_view.subview(['age', 'tracked']).get(pop_data.index)
            all_sims['employer_id'] = -1
            over_17 = all_sims.loc[all_sims.age > 17].index
            all_sims.loc[over_17, 'employer_id'] = self.assign_new_employer(over_17)

            # merge on employer addresses and names
            all_sims = all_sims.merge(
                self.businesses[self.columns_created],
                on='employer_id',
                how='left'
            )

            # handle untracked sims
            all_sims.loc[all_sims.tracked == False, 'employer_id'] = np.nan
            all_sims.loc[all_sims.tracked == False, 'employer_name'] = 'NA'
            all_sims.loc[all_sims.tracked == False, 'employer_address'] = 'NA'
            self.population_view.update(
                all_sims
            )

    def on_time_step(self, event: Event):
        """
        assign job if turning 18
        change jobs at rate of 50 changes / 100 person-years
        """
        pass

    ##################
    # Helper methods #
    ##################

    def generate_businesses(self, pop_data: SimulantData) -> pd.DataFrame():
        all_sims = self.population_view.subview(['age']).get(pop_data.index)
        over_17 = all_sims.loc[all_sims.age > 17]

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

        businesses = pd.concat([businesses, unemployed])
        return businesses

    def assign_new_employer(self, sim_index) -> pd.Series:
        return self.randomness.choice(
            index=sim_index,
            choices=self.businesses['employer_id'],
            p=self.businesses['probability']
        )

