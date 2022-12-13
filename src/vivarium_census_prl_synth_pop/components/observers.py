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
    recording/updating on some subset of timesteps (including every time step)
    and then writing out the results at the end of the sim.
    """

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "base_observer"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        # FIXME: move filepaths to data container
        # FIXME: settle on output dirs
        self.output_dir = build_output_dir(builder, subdir="observers")
        self.output_filename = self.get_output_filename()
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
    def get_output_filename(self) -> str:
        """Define the output filename in the concrete class"""
        pass

    @abstractmethod
    def get_population_view(self, builder) -> PopulationView:
        """Define the population view to use in the concrete class"""
        pass

    @abstractmethod
    def get_responses(self) -> pd.DataFrame:
        """Define responses dataset schema to be used in the concrete class"""
        pass

    ########################
    # Event-driven methods #
    ########################

    def on_collect_metrics(self, event: Event) -> None:
        if self.to_observe():
            self.do_observation(event)
        
    def to_observe(self) -> bool:
        """If True, will make an observation. This defaults to always True
        (ie record at every time step) and should be overwritten in each
        concrete observer as appropriate
        """
        return True

    @abstractmethod
    def do_observation(self) -> None:
        """Define the observations in the concrete class"""
        pass

    def on_simulation_end(self, event: Event) -> None:
        key = self.get_hdf_key()
        self.responses.to_hdf(self.output_dir / self.output_filename, key=key)

    @abstractmethod
    def get_hdf_key(self) -> "str":
        """Define the output hdf key in the concrete class"""
        pass


class TestObserver(BaseObserver):
    """test"""

    def __repr__(self):
        return "TestObserver()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "test_observer"

    #################
    # Setup methods #
    #################

    def get_output_filename(self):
        return "test_file.hdf"

    def setup(self, builder: Builder):
        super().setup(builder)

    def get_population_view(self, builder) -> PopulationView:
        """Returns the population view of interest to the observer"""
        population_view = builder.population.get_view(columns=[])
        return population_view

    def get_responses(self) -> pd.DataFrame:
        """Returns the response schema"""
        pass

    ########################
    # Event-driven methods #
    ########################
    
    def do_observation(self, event) -> None:
        """Define the observations in the concrete class"""
        new_responses = self.population_view.get(event.index)[0:2]
        new_responses['response_date'] = event.time.date()
        self.responses = pd.concat([self.responses, new_responses])

    def get_hdf_key(self) -> "str":
        """Define the output hdf key in the concrete class"""
        return "test_observation"


# FIXME: give this a more descriptive name (eg CensusObserver)
class Observers:
    # FIXME: the docstring is no longer correct; update it
    """
    at the start of simulant initialization:
    save population table with / key = date

    at the end of simulation:
    save population table with / key = date
    """

    def __repr__(self) -> str:
        return "Observers()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "observers"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        self.start_date = builder.configuration.time.start
        self.end_date = builder.configuration.time.end
        self.clock = builder.time.clock()
        self.counter = 0
        self.output_path = build_output_dir(builder, subdir="population_table") / "state_table.hdf"
        self.decennial_path = build_output_dir(builder, subdir="observers") / "decennial_census.hdf"

        self.randomness = builder.randomness.get_stream(self.name)
        self.population_view = builder.population.get_view(columns=[])
        self.response_probability_decennial = builder.lookup.build_table(
            data=data_values.RESPONSE_PROBABILITY_DECENNIAL
        )

        # FIXME: register to happen "on_collect_metrics" (end of time step)
        builder.event.register_listener("time_step__prepare", self.on_time_step__prepare)
        builder.event.register_listener("simulation_end", self.on_simulation_end)

    def on_time_step__prepare(self, event: Event) -> None:
        if self.counter == 0:
            start_date_str = (
                f"ymd_{self.start_date.year}_{self.start_date.month}_{self.start_date.day}"
            )
            state_table = self.population_view.get(event.index)
            state_table.to_hdf(self.output_path, start_date_str)
        self.counter += 1
        if (self.clock().year % 10 == 0) & (self.clock().month == 4):
            if self.clock().day < 29:  # because we only want one observation in April
                self.decennial_census(event, hdf_key=f"year_{self.clock().year}")

    def on_simulation_end(self, event: Event) -> None:
        end_date_str = f"ymd_{self.end_date.year}_{self.end_date.month}_{self.end_date.day}"
        state_table = self.population_view.get(event.index)
        state_table.to_hdf(self.output_path, end_date_str)

    def decennial_census(self, event: Event, hdf_key) -> None:
        pop = self.population_view.subview(metadata.DECENNIAL_CENSUS_COLUMNS_USED).get(
            event.index
        )
        pop["middle_initial"] = pop["middle_name"].astype(str).str[0]
        pop = pop.drop(columns="middle_name")

        # we don't have a 100% census response rate:
        respondents = self.randomness.filter_for_probability(
            pop, self.response_probability_decennial(pop.index), "decennial_respondents"
        )

        respondents.to_hdf(self.decennial_path, hdf_key)
