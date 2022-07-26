from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


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
            pop_data['employer_id'] = -1
            adults = pop_data.loc[pop_data.age > 17]

    def on_time_step(self, event: Event):
        """
        assign job if turning 18
        change jobs at rate of 50 changes / 100 person-years
        """
        pass

    ##################
    # Helper methods #
    ##################
