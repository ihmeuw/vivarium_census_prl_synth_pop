import datetime as dt
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.utilities import DAYS_PER_YEAR

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import data_values


class BaseObserver(ABC):
    """Base class for observing and recording relevant state table results. It
    maintains a separate dataset per concrete observation class and allows for
    recording/updating on some subset of timesteps (defaults to every time step)
    and then writing out the results at the end of the sim.
    """

    DEFAULT_INPUT_COLUMNS = [
        "first_name_id",
        "middle_name_id",
        "last_name_id",
        "age",
        "date_of_birth",
        "address_id",
        "sex",
        "race_ethnicity",
        "guardian_1",
        "guardian_2",
        "housing_type",
    ]
    DEFAULT_OUTPUT_COLUMNS = [
        "first_name_id",
        "middle_name_id",
        "last_name_id",
        "age",
        "sex",
        "race_ethnicity",
        "date_of_birth",
        "address_id",
        "guardian_1",
        "guardian_2",
        "guardian_1_address_id",
        "guardian_2_address_id",
    ]

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
        self.output_dir = Path(builder.configuration.output_data.results_directory)
        self.population_view = self.get_population_view(builder)
        self.responses = None

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

    ########################
    # Event-driven methods #
    ########################

    def on_collect_metrics(self, event: Event) -> None:
        if self.to_observe(event):
            observation = self.get_observation(event)
            if not pd.Series(self.output_columns).isin(observation.columns).all():
                raise RuntimeError(
                    f"{self.name} missing required column(s): "
                    f"{set(self.output_columns) - set(observation.columns)}"
                )
            if not pd.Series(observation.columns).isin(self.output_columns).all():
                raise RuntimeError(
                    f"{self.name} contains extra unexpected column(s): "
                    f"{set(observation.columns) - set(self.output_columns)}"
                )
            if self.responses is None:
                self.responses = observation
            else:
                self.responses = pd.concat([self.responses, observation])

    def to_observe(self, event: Event) -> bool:
        """If True, will make an observation. This defaults to always True
        (ie record at every time step) and should be overwritten in each
        concrete observer as appropriate.
        """
        return True

    @abstractmethod
    def get_observation(self, event: Event) -> pd.DataFrame:
        """Define the observations in the concrete class"""
        pass

    # TODO: consider using compressed csv instead of hdf
    def on_simulation_end(self, event: Event) -> None:
        output_dir = utilities.build_output_dir(self.output_dir, subdir="results")
        # 'fixed' format is not compatible with categorical data
        self.responses.to_hdf(output_dir / self.output_filename, key="data", format="t")


class HouseholdSurveyObserver(BaseObserver):

    ADDITIONAL_INPUT_COLUMNS = [
        "alive",
        "household_id",
        "state",
        "puma",
    ]
    ADDITIONAL_OUTPUT_COLUMNS = [
        "survey_date",
        "household_id",
        "housing_type",
        "state",
        "puma",
    ]
    SAMPLING_RATE_PER_MONTH = {
        "acs": 12000,
        "cps": 60000,
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
        return self.DEFAULT_INPUT_COLUMNS + self.ADDITIONAL_INPUT_COLUMNS

    @property
    def output_columns(self):
        return self.DEFAULT_OUTPUT_COLUMNS + self.ADDITIONAL_OUTPUT_COLUMNS

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        super().setup(builder)
        self.randomness = builder.randomness.get_stream(self.name)
        self.samples_per_timestep = int(
            HouseholdSurveyObserver.OVERSAMPLE_FACTOR
            * HouseholdSurveyObserver.SAMPLING_RATE_PER_MONTH[self.survey]
            * 12  # months per year
            * builder.configuration.time.step_size
            / DAYS_PER_YEAR
            * builder.configuration.population.population_size
            / data_values.US_POPULATION
        )

    ########################
    # Event-driven methods #
    ########################

    def get_observation(self, event: Event) -> pd.DataFrame:
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
        new_responses = utilities.add_guardian_address_ids(new_responses)
        # Apply column schema and concatenate
        return new_responses[self.output_columns]


class DecennialCensusObserver(BaseObserver):
    """Class for observing columns relevant to a decennial census on April
    1 of each decadal year (2020, 2030, etc).  Resulting table
    includes columns about guardian and group quarters type that are
    relevant to adding row noise.
    """

    ADDITIONAL_INPUT_COLUMNS = ["relation_to_household_head"]
    ADDITIONAL_OUTPUT_COLUMNS = [
        "relation_to_household_head",
        "census_year",
        "housing_type",
    ]

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
        return self.DEFAULT_INPUT_COLUMNS + self.ADDITIONAL_INPUT_COLUMNS

    @property
    def output_columns(self):
        return self.DEFAULT_OUTPUT_COLUMNS + self.ADDITIONAL_OUTPUT_COLUMNS

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.time_step = builder.configuration.time.step_size  # in days

    def to_observe(self, event: Event) -> bool:
        """Only observe if the census date falls during the time step"""
        census_year = 10 * (event.time.year // 10)
        census_date = dt.datetime(census_year, 4, 1)
        return self.clock() <= census_date < event.time

    def get_observation(self, event: Event) -> pd.DataFrame:
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",  # census should include only living simulants
        )
        pop = utilities.add_guardian_address_ids(pop)
        pop["census_year"] = event.time.year
        return pop


class WICObserver(BaseObserver):
    """Class for observing columns relevant to WIC administrative data."""

    ADDITIONAL_INPUT_COLUMNS = ["income"]
    ADDITIONAL_OUTPUT_COLUMNS = ["wic_year"]
    WIC_BASELINE_SALARY = 16_410
    WIC_SALARY_PER_HOUSEHOLD_MEMBER = 8_732
    WIC_RACE_ETHNICITIES = ["White", "Black", "Latino", "Other"]

    def __repr__(self):
        return f"WICObserver()"

    @property
    def name(self):
        return f"wic_observer"

    @property
    def output_filename(self):
        return f"wic.hdf"

    @property
    def input_columns(self):
        return self.DEFAULT_INPUT_COLUMNS + self.ADDITIONAL_INPUT_COLUMNS

    @property
    def output_columns(self):
        return self.DEFAULT_OUTPUT_COLUMNS + self.ADDITIONAL_OUTPUT_COLUMNS

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.time_step = builder.configuration.time.step_size  # in days
        self.randomness = builder.randomness.get_stream(self.name)

    def to_observe(self, event: Event) -> bool:
        """Only observe if Jan 1 occurs during the time step"""
        survey_date = dt.datetime(event.time.year, 1, 1)
        return self.clock() <= survey_date < event.time

    def get_observation(self, event: Event) -> pd.DataFrame:
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",  # WIC should include only living simulants
        )

        # add columns for output
        pop["wic_year"] = event.time.year
        pop = utilities.add_guardian_address_ids(pop)

        # add additional columns for simulating coverage
        pop["nominal_age"] = np.floor(pop["age"])

        # calculate household size and income for measuring WIC eligibility
        hh_size = pop["address_id"].value_counts()
        pop["hh_size"] = pop["address_id"].map(hh_size)

        hh_income = pop.groupby("address_id").income.sum()
        pop["hh_income"] = pop["address_id"].map(hh_income)

        # income eligibility for WIC is total household income less
        # than $16,410 + ($8,732 * number of people in the household)
        pop["wic_eligible"] = pop["hh_income"] <= (
            self.WIC_BASELINE_SALARY + self.WIC_SALARY_PER_HOUSEHOLD_MEMBER * pop["hh_size"]
        )

        # filter population to mothers and children under 5
        pop_u1 = pop[(pop["age"] < 1) & pop["wic_eligible"]]
        pop_1_to_5 = pop[(pop["age"] >= 1) & (pop["age"] < 5) & pop["wic_eligible"]]

        guardian_ids = np.union1d(pop_u1["guardian_1"], pop_u1["guardian_2"])
        pop_mothers = pop[
            (pop["sex"] == "Female") & pop.index.isin(guardian_ids) & pop["wic_eligible"]
        ]

        # determine who is covered using age/race-specific coverage probabilities
        # with additional constraint that all under-1 year olds with mother covered are also covered

        # first include some mothers
        pr_covered = data_values.COVERAGE_PROBABILITY_WIC["mothers"]
        mother_covered_probability = pop_mothers.race_ethnicity.map(pr_covered)
        pop_included_mothers = self.randomness.filter_for_probability(
            pop_mothers, mother_covered_probability
        )

        # then use same pattern for children aged 1 to 4
        pop_included = {}  # this dict will hold a pd.DataFrame for each age group
        for age, pop_age in pop_1_to_5.groupby("nominal_age"):
            pr_covered = data_values.COVERAGE_PROBABILITY_WIC[age]
            child_covered_pr = pop_age.race_ethnicity.map(pr_covered)
            pop_included[age] = self.randomness.filter_for_probability(
                pop_age, child_covered_pr
            )

        # selection for age 0 is more complicated; it should include
        # all simulants who have a mother enrolled and then a random
        # selection of additional simulants to reach the covered
        # probabilities

        simplified_race_ethnicity = pop_u1["race_ethnicity"].astype(
            pd.CategoricalDtype(categories=self.WIC_RACE_ETHNICITIES)
        )
        simplified_race_ethnicity[simplified_race_ethnicity.isnull()] = "Other"

        child_covered_pr = (
            pop_u1.guardian_1.isin(pop_included_mothers.index)
            | pop_u1.guardian_2.isin(pop_included_mothers.index)
        ).astype(
            float
        )  # pr is 1.0 for infants with mother on WIC
        for race_ethnicity in self.WIC_RACE_ETHNICITIES:
            race_eth_rows = simplified_race_ethnicity == race_ethnicity

            N = np.sum(race_eth_rows)  # total number of infants in this race group
            k = np.sum(
                race_eth_rows & (child_covered_pr == 1)
            )  # number included because their mother is on WIC
            if k < N:
                pr_covered = data_values.COVERAGE_PROBABILITY_WIC[0]
                child_covered_pr[race_eth_rows] = np.maximum(
                    child_covered_pr[
                        race_eth_rows
                    ],  # keep pr of 1.0 for the k infants with mother on WIC
                    (pr_covered[race_ethnicity] * N - k)
                    / (N - k)  # rescale probability for the remaining individuals
                    # so that expected number of infants on WIC matches target
                )
        pop_included[0] = self.randomness.filter_for_probability(pop_u1, child_covered_pr)

        return pd.concat([pop_included_mothers] + list(pop_included.values()))[
            self.output_columns
        ]


class SocialSecurityObserver(BaseObserver):
    """Class for observing columns relevant to Social Security registry."""

    ADDITIONAL_INPUT_COLUMNS = ["tracked", "alive", "entrance_time", "exit_time", "ssn"]
    OUTPUT_COLUMNS = [
        "first_name_id",
        "middle_name_id",
        "last_name_id",
        "date_of_birth",
        "event_type",
        "event_date",
    ]

    def __repr__(self):
        return f"SocialSecurityObserver()"

    @property
    def name(self):
        return f"social_security_observer"

    @property
    def output_filename(self):
        return f"social_security.hdf"

    @property
    def input_columns(self):
        return self.DEFAULT_INPUT_COLUMNS + self.ADDITIONAL_INPUT_COLUMNS

    @property
    def output_columns(self):
        return self.OUTPUT_COLUMNS

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.start_time = dt.datetime(**builder.configuration.time.start)
        self.end_time = dt.datetime(**builder.configuration.time.end)

    def to_observe(self, event: Event) -> bool:
        """Only observe if this is the final time step of the sim"""
        return self.clock() < self.end_time <= event.time

    def get_observation(self, event: Event) -> pd.DataFrame:
        pop = self.population_view.get(
            event.index,
            query="ssn == True",  # only include simulants with a SSN
        )
        df_creation = pop.filter(
            ["first_name_id", "middle_name_id", "last_name_id", "date_of_birth"]
        )
        df_creation["event_type"] = "creation"
        df_creation["event_date"] = np.where(
            pop["entrance_time"] <= self.start_time,
            pop["date_of_birth"],
            pop["entrance_time"],
        )

        df_death = pop[pop["alive"] == "dead"].filter(
            ["first_name_id", "middle_name_id", "last_name_id", "date_of_birth"]
        )
        df_death["event_type"] = "death"
        df_death["event_date"] = pop["exit_time"]

        return pd.concat([df_creation, df_death]).sort_values(["event_date", "date_of_birth"])
