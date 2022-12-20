from abc import ABC, abstractmethod

import numpy as np, pandas as pd

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
        assert self.time_step <= 30, 'DecennialCensusObserver requires model specification configuration with time.step_size <= 30'
        
    def get_population_view(self, builder) -> PopulationView:
        """Get the population view to be used for observations"""
        return builder.population.get_view(columns=metadata.DECENNIAL_CENSUS_COLUMNS_USED)

    def get_responses(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
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
        ])

    def to_observe(self, event: Event) -> bool:
        """Note: this method uses self.clock instead of event.time to handle
        the case where the sim starts on census day, e.g.  start time
        of 2020-04-01; in that case, the first event.time to appear in
        this function is 2020-04-29 (because the time.step_size is 28
        days)
        """
        return ((self.clock().year % 10 == 0)  # decennial year
                and (self.clock().month == 4)  # month of April
                and (self.clock().day <= self.time_step)  # time step containing first day of month
               )

    def do_observation(self, event) -> None:
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",  # census should include only living simulants
        )
        pop["middle_initial"] = pop["middle_name"].astype(str).str[0]
        pop = pop.drop(columns="middle_name")

        # merge address ids for guardian_1 and guardian_2 for the rows with guardians
        for i in [1,2]:
            s_guardian_id = pop[f"guardian_{i}"].dropna()
            s_guardian_id = s_guardian_id[s_guardian_id != -1] # is it faster to remove the negative values?
            pop[f"guardian_{i}_address_id"] = s_guardian_id.map(pop["address_id"])

        pop["census_year"] = event.time.year

        self.responses = pd.concat([self.responses, pop])


class WICObserver(BaseObserver):
    """Class for observing columns relevant to WIC administrative data.
    """

    def __repr__(self):
        return f"WICObserver()"

    @property
    def name(self):
        return f"wic_observer"

    @property
    def output_filename(self):
        return f"wic.hdf"

    def setup(self, builder: Builder):
        super().setup(builder)
        self.time_step = builder.configuration.time.step_size  # in days
        assert 1 <= self.time_step <= 30, 'WICObserver requires model specification configuration with 1 <= time.step_size <= 30'
        self.randomness = builder.randomness.get_stream(self.name)

    def get_population_view(self, builder) -> PopulationView:
        """Get the population view to be used for observations"""
        return builder.population.get_view(columns=metadata.WIC_OBSERVER_COLUMNS_USED)

    def get_responses(self) -> pd.DataFrame:
        self.response_columns = [
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
        ]  # NOTE: Steve is going to refactor this method, and I
           # expect this list will be more relevant in the refactored
           # version
        
        return pd.DataFrame(columns=self.response_columns)

    def to_observe(self, event: Event) -> bool:
        return ((event.time.month == 1)  # month of Jan
                and (1 < event.time.day <= 1+self.time_step)  # time step containing first day of month
               )

    def do_observation(self, event) -> None:
        pop = self.population_view.get(
            event.index,
            query="alive == 'alive'",  # WIC should include only living simulants
        )

        # add columns for output
        pop["wic_year"] = event.time.year
        pop["middle_initial"] = pop["middle_name"].astype(str).str[0]
        pop = pop.drop(columns="middle_name")

        # merge address ids for guardian_1 and guardian_2 for the rows with guardians
        for i in [1,2]:
            s_guardian_id = pop[f"guardian_{i}"].dropna()
            s_guardian_id = s_guardian_id[s_guardian_id != -1] # is it faster to remove the negative values?
            pop[f"guardian_{i}_address_id"] = s_guardian_id.map(pop["address_id"])

        # add additional columns for simulating coverage
        pop["nominal_age"] = np.floor(pop["age"])

        
        # calculate household size and income for measuring WIC eligibility
        hh_size = pop["address_id"].value_counts()
        pop["hh_size"] = pop["address_id"].map(hh_size)

        hh_income = pop.groupby("address_id").income.sum()
        pop["hh_income"] = pop["address_id"].map(hh_income)

        
        # income eligibility for WIC is total household income less
        # than $16,410 + ($8,732 * number of people in the household)
        pop["wic_eligible"] = (pop["hh_income"] <= (16_410 + 8_732*pop["hh_size"]))
        
        
        # filter population to mothers and children under 5
        pop_u1 = pop[(pop["age"] < 1) & pop["wic_eligible"]]
        pop_1_to_5 = pop[(pop["age"] >= 1) & (pop["age"] < 5) & pop["wic_eligible"]]

        guardian_ids = np.union1d(pop_u1["guardian_1"], pop_u1["guardian_2"])
        pop_mothers = pop[(pop["sex"] == "Female") & pop.index.isin(guardian_ids) & pop["wic_eligible"]]


        # determine who is covered using age/race-specific coverage probabilities
        # with additional constraint that all under-1 year olds with mother covered are also covered

        # first include some mothers
        pr_covered = data_values.COVERAGE_PROBABILITY_WIC['mothers']
        mother_covered_probability = pop_mothers.race_ethnicity.map(pr_covered)
        pop_included_mothers = self.randomness.filter_for_probability(pop_mothers, mother_covered_probability)

        # then use same pattern for children aged 1 to 4
        pop_included = {}  # this dict will hold a pd.DataFrame for each age group
        for age, pop_age in pop_1_to_5.groupby("nominal_age"):
            pr_covered = data_values.COVERAGE_PROBABILITY_WIC[age]
            child_covered_pr = pop_age.race_ethnicity.map(pr_covered)
            pop_included[age] = self.randomness.filter_for_probability(pop_age, child_covered_pr)


        # selection for age 0 is more complicated; it should include
        # all simulants who have a mother enrolled and then a random
        # selection of additional simulants to reach the covered
        # probabilities

        simplified_race_ethnicity = pop_u1["race_ethnicity"].copy()
        simplified_race_ethnicity[~pop_u1["race_ethnicity"].isin(["Latino", "Black", "White"])] = "Other"

        child_covered_pr = (pop_u1.guardian_1.isin(pop_included_mothers.index)
                            | pop_u1.guardian_2.isin(pop_included_mothers.index)).astype(float)  # pr is 1.0 for infants with mother on WIC
        for race_eth in ["Latino", "Black", "White", "Other"]:
            race_eth_rows = (simplified_race_ethnicity == race_eth)

            N = np.sum(race_eth_rows)  # total number of infants in this race group
            k = np.sum(race_eth_rows & (child_covered_pr==1))  # number included because their mother is on WIC
            if k < N:
                pr_covered = data_values.COVERAGE_PROBABILITY_WIC[0]
                child_covered_pr[race_eth_rows] = np.maximum(
                    child_covered_pr[race_eth_rows],  # keep pr of 1.0 for the k infants with mother on WIC
                    (pr_covered[race_eth]*N - k) / (N-k)  # rescale probability for the remaining individuals
                                                          # so that expected number of infants on WIC matches target
                )
        pop_included[0] = self.randomness.filter_for_probability(pop_u1, child_covered_pr)

        self.responses = pd.concat([self.responses, pop_included_mothers] + list(pop_included.values()))
        self.responses = self.responses.filter(self.response_columns)
