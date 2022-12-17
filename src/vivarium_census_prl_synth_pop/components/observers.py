from abc import ABC, abstractmethod

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView

from vivarium_census_prl_synth_pop.constants import data_values, metadata
from vivarium_census_prl_synth_pop.utilities import build_output_dir


class BaseObserver(ABC):
    """Base class for observing and recording relevant state table results. It
    maintains a separate dataset per concrete observation class and allows for
    recording/updating on some subset of timesteps (defaults to every time step)
    and then writing out the results at the end of the sim.
    """

    def __repr__(self):
        return "BaseObserver()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "base_observer"

    @property
    @abstractmethod
    def output_filename(self):
        pass

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        # FIXME: move filepaths to data container
        # FIXME: settle on output dirs
        self.output_dir = build_output_dir(builder, subdir="results")
        self.population_view = self.get_population_view(builder)
        self.responses = self.get_responses()

        # Register the listener to update the responses
        builder.event.register_listener(
            "collect_metrics",
            self.on_collect_metrics,
        )

        # Register the listener for final write-out
        builder.event.register_listener(
            "simulation_end",
            self.on_simulation_end,
        )

    @abstractmethod
    def get_population_view(self, builder) -> PopulationView:
        """Get the population view to be used for observations"""
        pass

    @abstractmethod
    def get_responses(self) -> pd.DataFrame:
        """Initializes the observation/results data structure and schema"""
        pass

    ########################
    # Event-driven methods #
    ########################

    def on_collect_metrics(self, event: Event) -> None:
        if self.to_observe(event):
            self.do_observation(event)

    def to_observe(self, event: Event) -> bool:
        """If True, will make an observation. This defaults to always True
        (ie record at every time step) and should be overwritten in each
        concrete observer as appropriate.
        """
        return True

    @abstractmethod
    def do_observation(self, event: Event) -> None:
        """Define the observations in the concrete class"""
        pass

    def on_simulation_end(self, event: Event) -> None:
        self.responses.to_hdf(self.output_dir / self.output_filename, key="responses")


class DecennialCensusObserver(BaseObserver):
    """Class for observing columns relevant to a decennial census on April
    1 of each decadal year (2020, 2030, etc).  Resulting table
    includes columns about guardian and group quarters type that are
    relevant to adding row noise.
    """

    def __repr__(self):
        return f"DecennialCensusObserver()"

    @property
    def name(self):
        return f"decennial_census_observer"

    @property
    def output_filename(self):
        return f"decennial_census.hdf"

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.time_step = builder.configuration.time.step_size  # in days
        assert (
            self.time_step <= 30
        ), "DecennialCensusObserver requires model specification configuration with time.step_size <= 30"

    def get_population_view(self, builder) -> PopulationView:
        """Get the population view to be used for observations"""
        return builder.population.get_view(columns=metadata.DECENNIAL_CENSUS_COLUMNS_USED)

    def get_responses(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "first_name",
                "middle_initial",
                "last_name",
                "age",
                "date_of_birth",
                "address_id",
                "relation_to_household_head",
                "sex",
                "race_ethnicity",
                "census_year",
                "guardian_1",
                "guardian_1_address_id",
                "guardian_2",
                "guardian_2_address_id",
                "housing_type",
            ]
        )

    def to_observe(self, event: Event) -> bool:
        """Note: this method uses self.clock instead of event.time to handle
        the case where the sim starts on census day, e.g.  start time
        of 2020-04-01; in that case, the first event.time to appear in
        this function is 2020-04-29 (because the time.step_size is 28
        days)
        """
        return (
            (self.clock().year % 10 == 0)  # decennial year
            and (self.clock().month == 4)  # month of April
            and (
                self.clock().day <= self.time_step
            )  # time step containing first day of month
        )

    def do_observation(self, event) -> None:
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",  # census should include only living simulants
        )
        pop["middle_initial"] = pop["middle_name"].astype(str).str[0]
        pop = pop.drop(columns="middle_name")

        # merge address ids for guardian_1 and guardian_2 for the rows with guardians
        for i in [1, 2]:
            s_guardian_id = pop[f"guardian_{i}"].dropna()
            s_guardian_id = s_guardian_id[
                s_guardian_id != -1
            ]  # is it faster to remove the negative values?
            pop[f"guardian_{i}_address_id"] = s_guardian_id.map(pop["address_id"])

        pop["census_year"] = event.time.year

        self.responses = pd.concat([self.responses, pop])
