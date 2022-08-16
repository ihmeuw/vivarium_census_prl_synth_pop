import numpy as np
import pandas as pd
from typing import Any

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health import utilities

from vivarium_census_prl_synth_pop.constants import data_values, data_keys, metadata
from vivarium_census_prl_synth_pop.components.synthetic_pii import AddressGenerator
from vivarium_census_prl_synth_pop.constants.data_values import UNEMPLOYED_ID, UNTRACKED_ID, WORKING_AGE


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


    def __init__(self):
        self.address_generator = AddressGenerator('businesses')

    def __repr__(self) -> str:
        return 'Businesses()'

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "businesses"

    @property
    def sub_components(self):
        return [self.address_generator]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.location = builder.data.load(data_keys.POPULATION.LOCATION)
        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_created = ['employer_id', 'employer_name', 'employer_address', 'employer_zipcode']
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
        Assign everyone working age and older an employer
        """
        if pop_data.creation_time < self.start_time:
            self.businesses = self.generate_businesses(pop_data)

            pop = self.population_view.subview(['age', 'tracked']).get(pop_data.index)
            pop['employer_id'] = UNEMPLOYED_ID
            working_age = pop.loc[pop.age >= data_values.WORKING_AGE].index
            pop.loc[working_age, 'employer_id'] = self.assign_random_employer(working_age)

            # merge on employer addresses and names
            pop = pop.merge(
                self.businesses[self.columns_created],
                on='employer_id',
                how='left'
            )

            # handle untracked sims
            pop.loc[pop.tracked == False, 'employer_id'] = UNTRACKED_ID
            pop.loc[pop.tracked == False, 'employer_name'] = 'NA'
            pop.loc[pop.tracked == False, 'employer_address'] = 'NA'
            pop.loc[pop.tracked == False, 'employer_zipcode'] = 'NA'
            self.population_view.update(
                pop
            )
        else:
            new_births = self.population_view.get(pop_data.index)

            new_births["employer_id"] = UNEMPLOYED_ID
            new_births["employer_name"] = 'unemployed'
            new_births["employer_address"] = 'NA'
            new_births["employer_zipcode"] = 'NA'

            self.population_view.update(new_births)

    def on_time_step(self, event: Event):
        """
        assign job if turning working age
        change jobs at rate of 50 changes / 100 person-years
        businesses change addresses at rate of 10 changes / 100 person-years\
        """
        # TODO: employers change addresses at a rate of 10 changes per 100 person years

        # change jobs if of working age already
        pop = self.population_view.subview(self.columns_created + ['age']).get(event.index)
        working_age = pop.loc[pop.age >= WORKING_AGE].index
        changing_jobs = self.randomness.filter_for_rate(
            working_age,
            np.ones(len(working_age))*(data_values.YEARLY_JOB_CHANGE_RATE * event.step_size.days / utilities.DAYS_PER_YEAR)
        )
        if len(changing_jobs) > 0:
            pop.loc[changing_jobs, "employer_id"] = self.assign_different_employer(changing_jobs)

            # add employer addresses and names
            pop.loc[changing_jobs, "employer_address"] = pop.loc[changing_jobs, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_address'].to_dict()
            )
            pop.loc[changing_jobs, "employer_zipcode"] = pop.loc[changing_jobs, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_zipcode'].to_dict()
            )
            pop.loc[changing_jobs, "employer_name"] = pop.loc[changing_jobs, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_name'].to_dict()
            )

        # assign job if turning working age
        turning_working_age = pop.loc[
            (pop.age >= data_values.WORKING_AGE - event.step_size.days / utilities.DAYS_PER_YEAR) &
            (pop.age < data_values.WORKING_AGE)
            ].index
        if len(turning_working_age) > 0:
            pop.loc[turning_working_age, 'employer_id'] = self.assign_random_employer(turning_working_age)

            # add employer addresses and names
            pop.loc[turning_working_age, "employer_address"] = pop.loc[turning_working_age, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_address'].to_dict()
            )
            pop.loc[turning_working_age, "employer_zipcode"] = pop.loc[turning_working_age, "employer_id"].map(
                self.businesses.set_index("employer_id")['employer_zipcode'].to_dict()
            )
            pop.loc[turning_working_age, "employer_name"] = pop.loc[turning_working_age, "employer_id"].map(
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
        n_working_age = len(pop.loc[pop.age >= data_values.WORKING_AGE])

        # TODO: when have more known employers, maybe move to csv
        known_employers = pd.DataFrame({
            'employer_id': [UNEMPLOYED_ID],
            'employer_name': ['unemployed'],
            'employer_address': ['NA'],
            'prevalence': [1 - data_values.PROPORTION_WORKFORCE_EMPLOYED[self.location]],
        })

        pct_adults_needing_employers = 1 - known_employers['prevalence'].sum()
        n_need_employers = np.round(n_working_age * pct_adults_needing_employers)
        employee_counts = np.random.lognormal(
            4, 1, size=int(n_need_employers // data_values.EXPECTED_EMPLOYEES_PER_BUSINESS)
        ).round()
        n_businesses = len(employee_counts)
        random_employers = pd.DataFrame({
            'employer_id': np.arange(n_businesses),
            'employer_name': ['not implemented']*n_businesses,
            'prevalence': employee_counts / employee_counts.sum() * pct_adults_needing_employers,
        })
        address_assignments = self.address_generator.generate(
                random_employers.index,
                state=metadata.US_STATE_ABBRV_MAP[self.location].lower()
        )
        random_employers['employer_address'] = random_employers.index.map(address_assignments['address'])
        random_employers['employer_zipcode'] = random_employers.index.map(address_assignments['zipcode'])

        untracked = pd.DataFrame({
            'employer_id': [UNTRACKED_ID],
            'employer_name': ['NA'],
            'employer_address': ['NA'],
            'employer_zipcode': ['NA'],
            'prevalence': 0,
        })

        businesses = pd.concat([known_employers, random_employers, untracked])
        return businesses

    def assign_random_employer(self, sim_index: pd.Index, additional_seed: Any = None) -> pd.Series:
        return self.randomness.choice(
            index=sim_index,
            choices=self.businesses['employer_id'],
            p=self.businesses['prevalence'],
            additional_key=additional_seed
        )

    def assign_different_employer(self, changing_jobs: pd.Index) -> pd.Series:
        current_employers = self.population_view.subview(['employer_id']).get(changing_jobs)['employer_id']

        new_employers = current_employers.copy()
        additional_seed = 0
        while (current_employers == new_employers).any():
            unchanged_employers = (current_employers == new_employers)
            new_employers[unchanged_employers] = self.assign_random_employer(
                sim_index=new_employers[unchanged_employers].index,
                additional_seed=additional_seed
            )
            additional_seed += 1

        return new_employers
