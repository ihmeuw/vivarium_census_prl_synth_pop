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
        # FIXME: Are these the correct output locations?
        self.output_path = build_output_dir(builder, subdir="population_table") / "state_table.hdf"
        self.decennial_path = build_output_dir(builder, subdir="population_table") / "decennial_census.hdf"

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
