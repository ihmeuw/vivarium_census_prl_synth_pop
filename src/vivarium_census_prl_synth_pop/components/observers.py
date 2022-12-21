import datetime as dt
from abc import ABC, abstractmethod

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
        "first_name",
        "middle_name",
        "last_name",
        "age",
        "date_of_birth",
        "address_id",
        "sex",
        "race_ethnicity",
        "guardian_1",
        "guardian_2",
        "housing_type",
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
            if not pd.Series(self.output_columns).isin(self.responses.columns).all():
                raise RuntimeError(
                    f"{self.name} missing required column(s): {set(self.output_columns) - set(self.responses.columns)}"
                )
            if not pd.Series(self.responses.columns).isin(self.output_columns).all():
                raise RuntimeError(
                    f"{self.name} contains extra unexpected column(s): {set(self.responses.columns) - set(self.output_columns)}"
                )

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

    SAMPLING_RATE_PER_MONTH = {
        "acs": 12000,
        "cps": 60000,
    }
    OVERSAMPLE_FACTOR = 2

    ADDITIONAL_INPUT_COLUMNS = [
        "alive",
        "household_id",
        "state",
        "puma",
    ]

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
        return [
            "survey_date",
            "household_id",
            "housing_type",
            "first_name",
            "middle_initial",
            "last_name",
            "age",
            "sex",
            "race_ethnicity",
            "date_of_birth",
            "address_id",
            "state",
            "puma",
            "guardian_1",
            "guardian_2",
            "guardian_1_address_id",
            "guardian_2_address_id",
        ]

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

    ADDITIONAL_INPUT_COLUMNS = ["relation_to_household_head"]

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

    def to_observe(self, event: Event) -> bool:
        """Only observe if the census date falls during the time step"""
        census_year = 10 * (event.time.year // 10)
        census_date = dt.datetime(census_year, 4, 1)
        return self.clock() <= census_date < event.time

    def do_observation(self, event) -> None:
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",  # census should include only living simulants
        )
        pop = utilities.convert_middle_name_to_initial(pop)
        pop = utilities.add_guardian_address_ids(pop)
        pop["census_year"] = event.time.year
        self.responses = pd.concat([self.responses, pop])


class WICObserver(BaseObserver):
    """Class for observing columns relevant to WIC administrative data."""

    ADDITIONAL_INPUT_COLUMNS = ["income"]

    WIC_BASELINE_SALARY = 16_410
    WIC_PER_HOUSEHOLD_SALARY = 8_732

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
        return [
            "address_id",
            "first_name",
            "middle_initial",
            "last_name",
            "age",
            "date_of_birth",
            "sex",
            "race_ethnicity",
            "wic_year",
            "guardian_1",
            "guardian_1_address_id",
            "guardian_2",
            "guardian_2_address_id",
        ]

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.time_step = builder.configuration.time.step_size  # in days
        self.randomness = builder.randomness.get_stream(self.name)

    def to_observe(self, event: Event) -> bool:
        """Only observe if Jan 1 occurs during the time step"""
        survey_date = dt.datetime(event.time.year, 1, 1)
        return self.clock() <= survey_date < event.time

    def do_observation(self, event) -> None:
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",  # WIC should include only living simulants
        )

        # add columns for output
        pop["wic_year"] = event.time.year
        pop = utilities.convert_middle_name_to_initial(pop)
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
            self.WIC_BASELINE_SALARY + self.WIC_PER_HOUSEHOLD_SALARY * pop["hh_size"]
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

        simplified_race_ethnicity = pop_u1["race_ethnicity"].copy()
        simplified_race_ethnicity[
            ~pop_u1["race_ethnicity"].isin(["Latino", "Black", "White"])
        ] = "Other"

        child_covered_pr = (
            pop_u1.guardian_1.isin(pop_included_mothers.index)
            | pop_u1.guardian_2.isin(pop_included_mothers.index)
        ).astype(
            float
        )  # pr is 1.0 for infants with mother on WIC
        for race_eth in ["Latino", "Black", "White", "Other"]:
            race_eth_rows = simplified_race_ethnicity == race_eth

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
                    (pr_covered[race_eth] * N - k)
                    / (N - k)  # rescale probability for the remaining individuals
                    # so that expected number of infants on WIC matches target
                )
        pop_included[0] = self.randomness.filter_for_probability(pop_u1, child_covered_pr)

        self.responses = pd.concat(
            [self.responses, pop_included_mothers] + list(pop_included.values())
        )[self.output_columns]
