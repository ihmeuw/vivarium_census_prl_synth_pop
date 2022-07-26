import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp


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
        self.columns_used = ['age'] # we'll use more as we increaes complexity
        self.population_view = builder.population.get_view(self.columns_used)
        self.employers = self.generate_employers()
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=['business_id']
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
            all_sims = self.population_view.get(pop_data.index)
            all_sims['employer_id'] = -1
            adults = all_sims.loc[all_sims.age > 17].index
            all_sims.loc[adults, 'employer_id'] = self.assign_new_employer(adults)
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

    def generate_employers(self) -> pd.DataFrame():
        test = np.random.lognormal(4, 1)
        pass

    def assign_new_employer(self, sim_index) -> pd.Series:
        pass
