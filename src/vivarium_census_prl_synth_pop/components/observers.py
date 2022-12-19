from abc import ABC, abstractmethod

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import data_values, metadata


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

    @property
    @abstractmethod
    def input_columns(self):
        pass

    @property
    @abstractmethod
    def output_columns(self):
        pass


    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        # FIXME: move filepaths to data container
        # FIXME: settle on output dirs
        self.output_dir = utilities.build_output_dir(builder, subdir="results")
        self.population_view = self.get_population_view(builder)
        self.responses = self.get_response_schema()

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

    def get_population_view(self, builder) -> PopulationView:
        """Returns the population view of interest to the observer"""
        cols = self.input_columns
        population_view = builder.population.get_view(columns=cols)
        return population_view

    def get_response_schema(self) -> pd.DataFrame:
        """Returns the response schema"""
        cols = self.output_columns
        return pd.DataFrame(columns=cols)

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

    # TODO: consider using compressed csv instead of hdf
    def on_simulation_end(self, event: Event) -> None:
        self.responses.to_hdf(self.output_dir / self.output_filename, key="data")


class HouseholdSurveyObserver(BaseObserver):

    SAMPLING_RATE_PER_TIMESTEP = {
        "ACS": 12000,
        "CPS": 60000,
    }
    OVERSAMPLE_FACTOR = 2

    def __init__(self, survey):
        self.survey = survey

    def __repr__(self):
        return f"HouseholdSurveyObserver({self.survey})"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return f"household_survey_observer.{self.survey}"

    @property
    def output_filename(self):
        return f"{self.survey}.hdf"

    @property
    def input_columns(self):
        return [
            "alive",
            "household_id",
            "housing_type",
            "first_name",
            "middle_name",
            "last_name",
            "age",
            "sex",
            "race_ethnicity",  # For simulant omission
            "date_of_birth",
            "address_id",
            "state",
            "puma",
            "guardian_1",  # For noise functions
            "guardian_2",  # For noise functions
        ]

    @property
    def output_columns(self):
        return [
            "survey_date",
            "household_id",
            "housing_type",
            "first_name",
            "middle_initial",
            "last_name",
            "age",
            "sex",
            "race_ethnicity",  # For simulant omission
            "date_of_birth",
            "address_id",
            "state",
            "puma",
            "guardian_1",  # For noise functions
            "guardian_2",  # For noise functions
            "guardian_1_address_id",  # For noise functions
            "guardian_2_address_id",  # For noise functions
        ]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        super().setup(builder)
        self.randomness = builder.randomness.get_stream(self.name)
        self.samples_per_timestep = int(
            HouseholdSurveyObserver.OVERSAMPLE_FACTOR
            * HouseholdSurveyObserver.SAMPLING_RATE_PER_TIMESTEP[self.survey]  # households per month
            * 12  # months per year
            * builder.configuration.time.step_size  # days per timestep
            / data_values.DAYS_PER_YEAR  # days per year
            * builder.configuration.population.population_size  # sim population
            / data_values.US_POPULATION  # US population
        )

    ########################
    # Event-driven methods #
    ########################

    def do_observation(self, event) -> None:
        """Records the survey responses on this time step."""
        new_responses = self.population_view.get(event.index, query='alive == "alive"')
        respondent_households = utilities.vectorized_choice(
            options=list(new_responses["household_id"].unique()),
            n_to_choose=self.samples_per_timestep,
            randomness_stream=self.randomness,
            additional_key="sampling_households",
        )
        new_responses = new_responses[
            new_responses["household_id"].isin(respondent_households)
        ]
        new_responses["survey_date"] = event.time.date()
        new_responses = utilities.convert_middle_name_to_initial(new_responses)
        new_responses = utilities.add_guardian_address_ids(new_responses)
        # Apply column schema and concatenate
        new_responses = new_responses[self.responses.columns]
        self.responses = pd.concat([self.responses, new_responses])


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

    @property
    def input_columns(self):
        return metadata.DECENNIAL_CENSUS_COLUMNS_USED
    
    @property
    def output_columns(self):
        return [
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

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.time_step = builder.configuration.time.step_size  # in days
        assert (
            self.time_step <= 30
        ), "DecennialCensusObserver requires model specification configuration with time.step_size <= 30"

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
        pop = utilities.convert_middle_name_to_initial(pop)
        pop = utilities.add_guardian_address_ids(pop)
        pop["census_year"] = event.time.year
        self.responses = pd.concat([self.responses, pop])
