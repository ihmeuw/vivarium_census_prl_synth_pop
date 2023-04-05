import datetime as dt
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium_public_health.utilities import DAYS_PER_YEAR

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import data_values, metadata, paths
from vivarium_census_prl_synth_pop.schema_entities import COLUMNS


class BaseObserver(ABC):
    """Base class for observing and recording relevant state table results. It
    maintains a separate dataset per concrete observation class and allows for
    recording/updating on some subset of timesteps (defaults to every time step)
    and then writing out the results at the end of the sim.
    """

    DEFAULT_INPUT_COLUMNS = [
        COLUMNS.FIRST_NAME_ID,
        COLUMNS.MIDDLE_NAME_ID,
        COLUMNS.LAST_NAME_ID,
        COLUMNS.AGE,
        COLUMNS.DOB,
        COLUMNS.SEX,
        COLUMNS.RACE_ETHNICITY,
        COLUMNS.GUARDIAN_1,
        COLUMNS.GUARDIAN_2,
    ]
    DEFAULT_OUTPUT_COLUMNS = [
        COLUMNS.FIRST_NAME_ID,
        COLUMNS.MIDDLE_NAME_ID,
        COLUMNS.LAST_NAME_ID,
        COLUMNS.AGE,
        COLUMNS.SEX,
        COLUMNS.RACE_ETHNICITY,
        COLUMNS.DOB,
        COLUMNS.ADDRESS_ID,
        COLUMNS.STATE_ID,
        COLUMNS.PO_BOX,
        COLUMNS.PUMA,
        COLUMNS.GUARDIAN_1,
        COLUMNS.GUARDIAN_2,
        COLUMNS.GUARDIAN_1_ADDRESS_ID,
        COLUMNS.GUARDIAN_2_ADDRESS_ID,
    ]

    configuration_defaults = {"observer": {"file_extension": "csv.bz2"}}

    def __init__(self):
        self.configuration_defaults = self._get_configuration_defaults()

    def _get_configuration_defaults(self) -> dict[str, dict]:
        return {self.name: BaseObserver.configuration_defaults["observer"]}

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
    def input_columns(self):
        pass

    @property
    def input_values(self):
        return []

    @property
    @abstractmethod
    def output_columns(self):
        pass

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder):
        # FIXME: move filepaths to data container
        self.seed = builder.configuration.randomness.random_seed
        self.file_extension = self.get_file_extension(builder)
        self.output_dir = Path(builder.configuration.output_data.results_directory)
        self.population_view = self.get_population_view(builder)
        self.responses = None
        self.pipelines = {
            pipeline: builder.value.get_value(pipeline) for pipeline in self.input_values
        }

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

    def get_file_extension(self, builder: Builder) -> str:
        extension = builder.configuration[self.name].file_extension
        if extension not in metadata.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension '{extension}' provided in configuration key "
                f"'{self.name}.file-extension'. Supported extensions are: "
                f"{metadata.SUPPORTED_EXTENSIONS}."
            )
        return extension

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

    def on_simulation_end(self, _: Event) -> None:
        output_dir = utilities.build_output_dir(
            self.output_dir / paths.RAW_RESULTS_DIR_NAME / self.name
        )
        if self.responses is None:
            logger.info(f"No results to write ({self.name})")
        else:
            self.responses.index.names = [COLUMNS.SIMULANT_ID]
            filepath = output_dir / f"{self.name}_{self.seed}.{self.file_extension}"
            if "hdf" == self.file_extension:
                self.responses.to_hdf(
                    filepath, "data", format="table", complib="bzip2", complevel=9
                )
            else:
                self.responses.to_csv(filepath)

    ##################
    # Helper methods #
    ##################

    def add_address(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds address columns to dataframe"""
        cols_to_add = [
            COLUMNS.ADDRESS_ID,
            COLUMNS.HOUSING_TYPE,
            COLUMNS.STATE_ID,
            COLUMNS.PUMA,
            COLUMNS.PO_BOX,
        ]
        df[cols_to_add] = self.pipelines["household_details"](df.index)[cols_to_add]
        return df


class HouseholdSurveyObserver(BaseObserver):
    INPUT_VALUES = ["household_details"]
    ADDITIONAL_INPUT_COLUMNS = [
        COLUMNS.ALIVE,
        COLUMNS.HOUSEHOLD_ID,
    ]
    ADDITIONAL_OUTPUT_COLUMNS = [
        COLUMNS.SURVEY_DATE,
        COLUMNS.HOUSEHOLD_ID,
        COLUMNS.HOUSING_TYPE,
    ]
    SAMPLING_RATE_PER_MONTH = {
        "acs": 12000,
        "cps": 60000,
    }
    OVERSAMPLE_FACTOR = 2

    def __init__(self, survey):
        self.survey = survey
        super().__init__()

    def __repr__(self):
        return f"HouseholdSurveyObserver({self.survey})"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return f"household_survey_observer_{self.survey}"

    @property
    def input_values(self):
        return self.INPUT_VALUES

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
            / builder.configuration.us_population_size
        )

    ########################
    # Event-driven methods #
    ########################

    def get_observation(self, event: Event) -> pd.DataFrame:
        """Records the survey responses on this time step."""
        new_responses = self.population_view.get(event.index, query="alive == 'alive'")
        respondent_households = utilities.vectorized_choice(
            options=list(new_responses[COLUMNS.HOUSEHOLD_ID].unique()),
            n_to_choose=self.samples_per_timestep,
            randomness_stream=self.randomness,
            additional_key="sampling_households",
        )
        new_responses = new_responses[
            new_responses[COLUMNS.HOUSEHOLD_ID].isin(respondent_households)
        ]
        # Must be a timestamp, not an actual `date` type, in order to save to HDF in table mode
        new_responses[COLUMNS.SURVEY_DATE] = pd.Timestamp(event.time.date())
        new_responses = self.add_address(new_responses)
        new_responses = utilities.add_guardian_address_ids(new_responses)
        # Apply column schema and concatenate
        return new_responses[self.output_columns]


class DecennialCensusObserver(BaseObserver):
    """Class for observing columns relevant to a decennial census on April
    1 of each decadal year (2020, 2030, etc).  Resulting table
    includes columns about guardian and group quarters type that are
    relevant to adding row noise.
    """

    INPUT_VALUES = ["household_details"]
    ADDITIONAL_INPUT_COLUMNS = [COLUMNS.RELATION_TO_HOUSEHOLD_HEAD]
    ADDITIONAL_OUTPUT_COLUMNS = [
        COLUMNS.RELATION_TO_HOUSEHOLD_HEAD,
        COLUMNS.YEAR,
        COLUMNS.HOUSING_TYPE,
    ]

    def __repr__(self):
        return f"DecennialCensusObserver()"

    @property
    def name(self):
        return f"decennial_census_observer"

    @property
    def input_values(self):
        return self.INPUT_VALUES

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
        pop = self.add_address(pop)
        pop = utilities.add_guardian_address_ids(pop)
        pop[COLUMNS.YEAR] = event.time.year

        return pop[self.output_columns]


class WICObserver(BaseObserver):
    """Class for observing columns relevant to WIC administrative data."""

    INPUT_VALUES = [COLUMNS.INCOME, "household_details"]
    ADDITIONAL_INPUT_COLUMNS = [
        COLUMNS.HOUSEHOLD_ID,
        COLUMNS.RELATION_TO_HOUSEHOLD_HEAD,
    ]
    ADDITIONAL_OUTPUT_COLUMNS = [
        COLUMNS.YEAR,
        COLUMNS.HOUSEHOLD_ID,
        COLUMNS.HOUSING_TYPE,
        COLUMNS.RELATION_TO_HOUSEHOLD_HEAD,
    ]
    WIC_BASELINE_SALARY = 16_410
    WIC_SALARY_PER_HOUSEHOLD_MEMBER = 8_732
    WIC_RACE_ETHNICITIES = ["White", "Black", "Latino", "Other"]

    def __repr__(self):
        return f"WICObserver()"

    @property
    def name(self):
        return f"wic_observer"

    @property
    def input_columns(self):
        return self.DEFAULT_INPUT_COLUMNS + self.ADDITIONAL_INPUT_COLUMNS

    @property
    def input_values(self):
        return self.INPUT_VALUES

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
        pop[COLUMNS.INCOME] = self.pipelines[COLUMNS.INCOME](pop.index)

        # add columns for output
        pop[COLUMNS.YEAR] = event.time.year
        pop = self.add_address(pop)
        pop = utilities.add_guardian_address_ids(pop)

        # add additional columns for simulating coverage
        pop["nominal_age"] = np.floor(pop[COLUMNS.AGE])

        # calculate household size and income for measuring WIC eligibility
        hh_size = pop[COLUMNS.ADDRESS_ID].value_counts()
        pop["hh_size"] = pop[COLUMNS.ADDRESS_ID].map(hh_size)

        hh_income = pop.groupby(COLUMNS.ADDRESS_ID)[COLUMNS.INCOME].sum()
        pop["hh_income"] = pop[COLUMNS.ADDRESS_ID].map(hh_income)

        # income eligibility for WIC is total household income less
        # than $16,410 + ($8,732 * number of people in the household)
        pop["wic_eligible"] = pop["hh_income"] <= (
            self.WIC_BASELINE_SALARY + self.WIC_SALARY_PER_HOUSEHOLD_MEMBER * pop["hh_size"]
        )

        # filter population to mothers and children under 5
        pop_u1 = pop[(pop[COLUMNS.AGE] < 1) & pop["wic_eligible"]]
        pop_1_to_5 = pop[
            (pop[COLUMNS.AGE] >= 1) & (pop[COLUMNS.AGE] < 5) & pop["wic_eligible"]
        ]

        guardian_ids = np.union1d(pop_u1[COLUMNS.GUARDIAN_1], pop_u1[COLUMNS.GUARDIAN_2])
        pop_mothers = pop[
            (pop[COLUMNS.SEX] == "Female")
            & pop.index.isin(guardian_ids)
            & pop["wic_eligible"]
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

        simplified_race_ethnicity = pop_u1[COLUMNS.RACE_ETHNICITY].astype(
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

    ADDITIONAL_INPUT_COLUMNS = [
        COLUMNS.TRACKED,
        COLUMNS.ALIVE,
        COLUMNS.ENTRANCE_TIME,
        COLUMNS.EXIT_TIME,
        COLUMNS.HAS_SSN,
    ]
    OUTPUT_COLUMNS = [
        COLUMNS.FIRST_NAME_ID,
        COLUMNS.MIDDLE_NAME_ID,
        COLUMNS.LAST_NAME_ID,
        COLUMNS.DOB,
        COLUMNS.EVENT_TYPE,
        COLUMNS.EVENT_DATE,
        COLUMNS.SEX,
        COLUMNS.RACE_ETHNICITY,
    ]
    POST_PROCESSING_FIRST_NAME_METADATA_COLS = [
        COLUMNS.FIRST_NAME_ID,
        COLUMNS.MIDDLE_NAME_ID,
        COLUMNS.LAST_NAME_ID,
        COLUMNS.DOB,
        COLUMNS.SEX,
        COLUMNS.RACE_ETHNICITY,
    ]

    def __repr__(self):
        return f"SocialSecurityObserver()"

    @property
    def name(self):
        return f"social_security_observer"

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
            query="has_ssn == True",  # only include simulants with a SSN
        )
        df_creation = pop.filter(self.POST_PROCESSING_FIRST_NAME_METADATA_COLS)
        df_creation[COLUMNS.EVENT_TYPE] = "creation"
        df_creation[COLUMNS.EVENT_DATE] = np.where(
            pop[COLUMNS.ENTRANCE_TIME] <= self.start_time,
            pop[COLUMNS.DOB],
            pop[COLUMNS.ENTRANCE_TIME],
        )

        df_death = pop[pop[COLUMNS.ALIVE] == "dead"].filter(
            self.POST_PROCESSING_FIRST_NAME_METADATA_COLS
        )
        df_death[COLUMNS.EVENT_TYPE] = "death"
        df_death[COLUMNS.EVENT_DATE] = pop[COLUMNS.EXIT_TIME]

        return pd.concat([df_creation, df_death]).sort_values(
            [COLUMNS.EVENT_DATE, COLUMNS.DOB]
        )


class TaxObserver:
    """Holder for three interdependent observers relevant to tax data"""

    @property
    def name(self):
        return f"tax_observer"

    def __init__(self):
        tax_w2 = TaxW2Observer()
        tax_1040 = Tax1040Observer(tax_w2)
        tax_dependents = TaxDependentsObserver(tax_w2)

        self._sub_components = [
            tax_w2,
            tax_1040,
            tax_dependents,
        ]  # following pattern from vivarium.examples.disease_model.disease.SISDiseaseModel
        # TODO: it would be cool if there was more documentation on this, and if it was easy to find!

    @property
    def sub_components(self):
        return self._sub_components


class TaxW2Observer(BaseObserver):
    """Class for observing columns relevant to W2 and 1099 tax data.

    Maintains a pd.Series for last year's income for each (person, employer)-pair,
    which the Tax1040Observer and TaxDependentObserver classes use
    """

    INPUT_VALUES = [COLUMNS.INCOME, "household_details", "business_details"]
    ADDITIONAL_INPUT_COLUMNS = [
        COLUMNS.ALIVE,
        COLUMNS.IN_UNITED_STATES,
        COLUMNS.TRACKED,
        COLUMNS.HAS_SSN,
        COLUMNS.EMPLOYER_ID,
    ]
    OUTPUT_COLUMNS = [
        COLUMNS.FIRST_NAME_ID,
        COLUMNS.MIDDLE_NAME_ID,
        COLUMNS.LAST_NAME_ID,
        COLUMNS.AGE,
        COLUMNS.DOB,
        COLUMNS.SEX,
        COLUMNS.HAS_SSN,
        COLUMNS.SSN_ID,  # simulant id for ssn from another simulant
        COLUMNS.ADDRESS_ID,
        COLUMNS.STATE_ID,
        COLUMNS.PUMA,
        COLUMNS.PO_BOX,
        COLUMNS.EMPLOYER_ID,
        COLUMNS.EMPLOYER_NAME,
        COLUMNS.EMPLOYER_ADDRESS_ID,
        COLUMNS.EMPLOYER_STATE_ID,
        COLUMNS.EMPLOYER_PUMA,
        COLUMNS.INCOME,
        COLUMNS.HOUSING_TYPE,
        COLUMNS.TAX_YEAR,
        COLUMNS.RACE_ETHNICITY,
        COLUMNS.TAX_FORM,
    ]

    def __repr__(self):
        return f"TaxW2Observer()"

    @property
    def name(self):
        return f"tax_w2_observer"

    @property
    def input_columns(self):
        return self.DEFAULT_INPUT_COLUMNS + self.ADDITIONAL_INPUT_COLUMNS

    @property
    def input_values(self):
        return self.INPUT_VALUES

    @property
    def output_columns(self):
        return self.OUTPUT_COLUMNS

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()

        self.vivarium_randomness = builder.randomness.get_stream(self.name)
        np_random_seed = 12345 + int(self.vivarium_randomness.seed)
        self.np_randomness = np.random.default_rng(np_random_seed)

        # increment income based on the job the simulant has during
        # the course of the time_step, which might change if we do
        # this check on_time_step instead of on_time_step__prepare
        builder.event.register_listener("time_step__prepare", self.on_time_step__prepare)
        self.income_this_year = empty_income_series()
        self.income_last_year = empty_income_series()
        self.time_step = builder.configuration.time.step_size  # in days

        # set income_last_year and reset income_this_year on
        # time_step__cleanup to make sure it is in the needed format
        # for all subcomponents of TaxObserver
        builder.event.register_listener("time_step__cleanup", self.on_time_step__cleanup)

    def on_time_step__prepare(self, event):
        """increment income based on the job the simulant has during
        the course of the time_step, which might change if we do
        this check on_time_step instead of on_time_step__prepare
        """
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive' and in_united_states and tracked",
        )
        pop[COLUMNS.INCOME] = self.pipelines[COLUMNS.INCOME](pop.index)

        # increment income for all person/employment pairs with income > 0
        income_this_time_step = pd.Series(
            pop[COLUMNS.INCOME].values * self.time_step / DAYS_PER_YEAR,
            index=pd.MultiIndex.from_arrays(
                [pop.index, pop[COLUMNS.EMPLOYER_ID]],
                names=[COLUMNS.SIMULANT_ID, COLUMNS.EMPLOYER_ID],
            ),
        )

        income_this_time_step = income_this_time_step[income_this_time_step > 0]

        self.income_this_year = self.income_this_year.add(
            income_this_time_step, fill_value=0.0
        )

    def on_time_step__cleanup(self, event):
        """set income_last_year and reset income_this_year on
        time_step__cleanup to make sure it is in the needed format
        for all subcomponents of TaxObserver
        """
        if self.to_observe(event):
            self.income_last_year = self.income_this_year
            self.income_last_year.name = (
                COLUMNS.INCOME  # HACK: it would be nice if this name stayed
            )
            self.income_this_year = empty_income_series()

    def to_observe(self, event: Event) -> bool:
        """Observe if Jan 1 falls during this time step"""
        tax_date = dt.datetime(event.time.year, 1, 1)
        return self.clock() < tax_date <= event.time

    def get_observation(self, event: Event) -> pd.DataFrame:
        pop_full = self.population_view.get(event.index)
        pop_full = self.add_address(pop_full)

        ### create dataframe of all person/employment pairs

        # start with income to date, which has simulant_id and employer_id as multi-index
        # with the pd.Series, but it is getting lost at some point in the computation

        df_w2 = self.income_last_year.reset_index()
        df_w2[COLUMNS.TAX_YEAR] = event.time.year - 1

        # merge in simulant columns based on simulant id
        for col in [
            COLUMNS.FIRST_NAME_ID,
            COLUMNS.MIDDLE_NAME_ID,
            COLUMNS.LAST_NAME_ID,
            COLUMNS.AGE,
            COLUMNS.DOB,
            COLUMNS.SEX,
            COLUMNS.HAS_SSN,
            COLUMNS.ADDRESS_ID,
            COLUMNS.STATE_ID,
            COLUMNS.PUMA,
            COLUMNS.HOUSING_TYPE,
            COLUMNS.RACE_ETHNICITY,
            COLUMNS.PO_BOX,
        ]:
            df_w2[col] = df_w2[COLUMNS.SIMULANT_ID].map(pop_full[col])

        # Tracked, US population to be dependents or get their SSNs borrowed
        pop = pop_full[
            (pop_full[COLUMNS.ALIVE] == COLUMNS.ALIVE)
            & pop_full[COLUMNS.TRACKED]
            & pop_full[COLUMNS.IN_UNITED_STATES]
        ]

        # for simulants without ssn, record a simulant_id for a random household
        # member with an ssn, if one exists
        simulants_wo_ssn = df_w2.loc[~df_w2[COLUMNS.HAS_SSN], COLUMNS.ADDRESS_ID]
        household_members_w_ssn = (
            pop[pop[COLUMNS.HAS_SSN]]
            .groupby(COLUMNS.ADDRESS_ID)
            .apply(lambda df_g: list(df_g.index))
        )
        household_members_w_ssn = simulants_wo_ssn.map(household_members_w_ssn).dropna()
        ssn_for_simulants_wo = household_members_w_ssn.map(self.np_randomness.choice)
        df_w2[COLUMNS.SSN_ID] = ssn_for_simulants_wo
        df_w2[COLUMNS.SSN_ID] = df_w2[COLUMNS.SSN_ID].fillna(-1).astype(int)

        # merge in *current* employer details based on employer_id
        business_details = (
            self.pipelines["business_details"](df_w2[COLUMNS.EMPLOYER_ID])
            .groupby(COLUMNS.EMPLOYER_ID)
            .first()
        )
        for col in [
            COLUMNS.EMPLOYER_ADDRESS_ID,
            COLUMNS.EMPLOYER_STATE_ID,
            COLUMNS.EMPLOYER_PUMA,
            COLUMNS.EMPLOYER_NAME,
        ]:
            df_w2[col] = df_w2[COLUMNS.EMPLOYER_ID].map(business_details[col])

        df_w2 = df_w2.set_index([COLUMNS.SIMULANT_ID])

        df_w2[COLUMNS.TAX_FORM] = self.vivarium_randomness.choice(
            index=df_w2.index,
            choices=["W2", "1099"],
            p=[
                data_values.Taxes.PERCENT_W2_RECEIVED,
                data_values.Taxes.PERCENT_1099_RECEIVED,
            ],
            additional_key="type_of_form",
        )
        return df_w2[self.output_columns]


class TaxDependentsObserver(BaseObserver):
    """Class for observing columns relevant to identifying dependents in
    tax data.

    This is most important for the 1040 data, but relies on data from
    the W2 observer, and is better represented in a "long" form with a
    row for each guardian/dependent pair

    NOTE: as implemented, this captures the dependents' age and
    address_id on Jan 1, while it might be more realistic to capture
    them on April 15

    """

    INPUT_VALUES = ["household_details"]
    ADDITIONAL_INPUT_COLUMNS = [
        COLUMNS.ALIVE,
        COLUMNS.IN_UNITED_STATES,
        COLUMNS.TRACKED,
        COLUMNS.HAS_SSN,
    ]
    OUTPUT_COLUMNS = [
        COLUMNS.GUARDIAN_ID,
        COLUMNS.DEPENDENT_ID,
        COLUMNS.FIRST_NAME_ID,
        COLUMNS.MIDDLE_NAME_ID,
        COLUMNS.LAST_NAME_ID,
        COLUMNS.AGE,
        COLUMNS.DOB,
        COLUMNS.ADDRESS_ID,
        COLUMNS.PO_BOX,
        COLUMNS.STATE_ID,
        COLUMNS.PUMA,
        COLUMNS.HOUSING_TYPE,
        COLUMNS.SEX,
        COLUMNS.HAS_SSN,
        COLUMNS.TAX_YEAR,
        COLUMNS.RACE_ETHNICITY,
    ]

    def __init__(self, w2_observer):
        super().__init__()
        self.w2_observer = w2_observer

    def __repr__(self):
        return f"TaxDependentsObserver()"

    @property
    def name(self):
        return f"tax_dependents_observer"

    @property
    def input_columns(self):
        return self.DEFAULT_INPUT_COLUMNS + self.ADDITIONAL_INPUT_COLUMNS

    @property
    def input_values(self):
        return self.INPUT_VALUES

    @property
    def output_columns(self):
        return self.OUTPUT_COLUMNS

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()

    def to_observe(self, event: Event) -> bool:
        """Observe if Jan 1 falls during this time step"""
        tax_date = dt.datetime(event.time.year, 1, 1)
        return self.clock() < tax_date <= event.time

    def get_observation(self, event: Event) -> pd.DataFrame:
        pop_full = self.population_view.get(event.index)
        pop_full = self.add_address(pop_full)

        # Tracked, US population to be dependents
        pop = pop_full[
            (
                pop_full[COLUMNS.ALIVE] == COLUMNS.ALIVE
            )  # really should include people who died in last year
            & pop_full[COLUMNS.TRACKED]  # ??? should we include untracked, too?
            & pop_full[
                COLUMNS.IN_UNITED_STATES
            ]  # and if they were in usa in last year, maybe they still count
        ]

        ### create dataframe of all guardian/potential dependent pairs

        # create lists of dependent ids and dependent address ids
        df_eligible_dependents = pd.concat(
            [
                pop[pop[COLUMNS.GUARDIAN_1] != -1].eval("guardian_id=guardian_1"),
                pop[pop[COLUMNS.GUARDIAN_2] != -1].eval("guardian_id=guardian_2"),
            ]
        )

        # not all simulants with a guardian are eligible to be dependents
        # need to know income of dependents in past year for this
        #
        # NOTE: this assumes that the
        # TaxDependentObserver.get_observation is called after
        # TaxW2Observer.get_observation, which is achieved by listing
        # them in this order in the TaxObserver
        last_year_income = self.w2_observer.income_last_year.groupby(
            COLUMNS.SIMULANT_ID
        ).sum()
        df_eligible_dependents["last_year_income"] = last_year_income
        df_eligible_dependents["last_year_income"] = df_eligible_dependents[
            "last_year_income"
        ].fillna(0)

        df_eligible_dependents = df_eligible_dependents[
            # Dependents must qualify as one of the following:
            # Be under the age of 19 (less than or equal to 18)
            (df_eligible_dependents[COLUMNS.AGE] < 19)
            # OR be less than 24, in GQ in college, and earn less than $10,000
            | (
                (df_eligible_dependents[COLUMNS.AGE] < 24)
                & (df_eligible_dependents[COLUMNS.HOUSING_TYPE] == "College")
                & (df_eligible_dependents["last_year_income"] < 10_000)
            )
            # OR be any age, but earn less than $4300
            | (df_eligible_dependents["last_year_income"] < 4_300)
        ]

        df = df_eligible_dependents
        df[COLUMNS.DEPENDENT_ID] = df.index
        df[COLUMNS.TAX_YEAR] = event.time.year - 1

        return df[self.OUTPUT_COLUMNS]


class Tax1040Observer(BaseObserver):
    """Class for observing columns relevant to 1040 tax data (most of
    these are already recorded by W2 observer, but there might be
    migration between Jan 1 and April 15)

    """

    INPUT_VALUES = [COLUMNS.INCOME, "household_details", "business_details"]
    ADDITIONAL_INPUT_COLUMNS = [
        COLUMNS.ALIVE,
        COLUMNS.IN_UNITED_STATES,
        COLUMNS.TRACKED,
        COLUMNS.HAS_SSN,
        COLUMNS.RELATION_TO_HOUSEHOLD_HEAD,
    ]
    OUTPUT_COLUMNS = [
        COLUMNS.FIRST_NAME_ID,
        COLUMNS.MIDDLE_NAME_ID,
        COLUMNS.LAST_NAME_ID,
        COLUMNS.AGE,
        COLUMNS.DOB,
        COLUMNS.SEX,
        COLUMNS.HAS_SSN,
        COLUMNS.ADDRESS_ID,  # we do not need to include household_id because we can find it from address_id
        COLUMNS.PO_BOX,
        COLUMNS.STATE_ID,
        COLUMNS.PUMA,
        COLUMNS.RACE_ETHNICITY,
        COLUMNS.RELATION_TO_HOUSEHOLD_HEAD,  # needed to identify couples filing jointly
        COLUMNS.HOUSING_TYPE,
        COLUMNS.TAX_YEAR,
        COLUMNS.ALIVE,
        COLUMNS.IN_UNITED_STATES,
        COLUMNS.JOINT_FILER,
    ]

    def __init__(self, w2_observer):
        super().__init__()
        self.w2_observer = w2_observer

    def __repr__(self):
        return f"Tax1040Observer()"

    @property
    def name(self):
        return f"tax_1040_observer"

    @property
    def input_columns(self):
        return self.DEFAULT_INPUT_COLUMNS + self.ADDITIONAL_INPUT_COLUMNS

    @property
    def input_values(self):
        return self.INPUT_VALUES

    @property
    def output_columns(self):
        return self.OUTPUT_COLUMNS

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream(self.name)

    def to_observe(self, event: Event) -> bool:
        """Observe if April 15 falls during this time step"""
        tax_date = dt.datetime(event.time.year, 4, 15)
        return self.clock() < tax_date <= event.time

    def get_observation(self, event: Event) -> pd.DataFrame:
        pop = self.population_view.get(event.index)
        pop = self.add_address(pop)

        # add derived columns
        pop[COLUMNS.TAX_YEAR] = event.time.year - 1
        # todo: Add joint filing random choice
        partners = [
            "Opp-sex spouse",
            "Opp-sex partner",
            "Same-sex spouse",
            "Same-sex partner",
        ]
        partners_of_household_head_idx = pop.index[
            pop[COLUMNS.RELATION_TO_HOUSEHOLD_HEAD].isin(partners)
        ]
        pop[COLUMNS.JOINT_FILER] = False
        pop.loc[partners_of_household_head_idx, COLUMNS.JOINT_FILER] = self.randomness.choice(
            index=partners_of_household_head_idx,
            choices=[True, False],
            p=[
                data_values.Taxes.PROBABILITY_OF_JOINT_FILER,
                1 - data_values.Taxes.PROBABILITY_OF_JOINT_FILER,
            ],
            additional_key="joint_filing_1040",
        )

        return pop[self.OUTPUT_COLUMNS]


def empty_income_series():
    return pd.Series(
        index=pd.MultiIndex.from_arrays(
            [[], []],
            names=[
                COLUMNS.SIMULANT_ID,
                COLUMNS.EMPLOYER_ID,
            ],
        ),
        dtype="float64",
    )
