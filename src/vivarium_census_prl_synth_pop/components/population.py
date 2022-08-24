import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.utilities import to_years

from vivarium_census_prl_synth_pop.components.synthetic_pii import NameGenerator
from vivarium_census_prl_synth_pop.components.synthetic_pii import SSNGenerator
from vivarium_census_prl_synth_pop.constants import data_keys
from vivarium_census_prl_synth_pop.constants import metadata
from vivarium_census_prl_synth_pop.utilities import vectorized_choice


class Population:
    def __init__(self):
        self.name_generator = NameGenerator()
        self.ssn_generator = SSNGenerator()

    @property
    def name(self):
        return "population"

    @property
    def sub_components(self):
        return [self.name_generator, self.ssn_generator]

    def setup(self, builder: Builder):
        self.config = builder.configuration.population
        self.seed = builder.configuration.randomness.random_seed
        self.randomness = builder.randomness.get_stream(
            "household_sampling", for_initialization=True
        )
        self.start_time = get_time_stamp(builder.configuration.time.start)

        self.columns_created = [
            "household_id",
            "state",
            "puma",
            "relation_to_household_head",
            "sex",
            "age",
            "race_ethnicity",
            "first_name",
            "middle_name",
            "last_name",
            "ssn",
            "alive",
            "entrance_time",
            "exit_time",
        ]
        self.register_simulants = builder.randomness.register_simulants
        self.population_view = builder.population.get_view(self.columns_created + ["tracked"])
        self.population_data = self._load_population_data(builder)

        builder.population.initializes_simulants(
            self.initialize_simulants,
            creates_columns=self.columns_created,
            requires_columns=["tracked"],
        )

        builder.event.register_listener("time_step__cleanup", self.on_time_step__prepare)

    def _load_population_data(self, builder: Builder):
        households = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)
        persons = builder.data.load(data_keys.POPULATION.PERSONS)
        return {"households": households, "persons": persons}

    def initialize_simulants(self, pop_data: SimulantData) -> None:
        # at start of sim, generate base population
        if pop_data.creation_time < self.start_time:
            self.generate_base_population(pop_data)

        # if new simulants are born into the sim
        else:
            self.initialize_newborns(pop_data)

    def generate_base_population(self, pop_data: SimulantData) -> None:
        approx_in_group_quarters = metadata.P_GROUP_QUARTERS * self.config.population_size
        n_in_households = self.config.population_size - approx_in_group_quarters

        # oversample households
        chosen_households = vectorized_choice(
            options=self.population_data["households"]["census_household_id"],
            n_to_choose=n_in_households,
            randomness_stream=self.randomness,
            weights=self.population_data["households"]["household_weight"],
        )

        # create unique id for resampled households
        chosen_households = pd.DataFrame(
            {
                "census_household_id": chosen_households,
                "household_id": np.arange(len(chosen_households)),
            }
        )

        # pull back on state and puma
        chosen_households = pd.merge(
            chosen_households,
            self.population_data["households"][["state", "puma", "census_household_id"]],
            on="census_household_id",
            how="left",
        )

        # get all simulants per household
        chosen_persons = pd.merge(
            chosen_households,
            self.population_data["persons"],
            on="census_household_id",
            how="left",
        )

        # get rid simulants in excess of desired pop size
        households_to_discard = chosen_persons.loc[
            n_in_households:, "household_id"
        ].unique()
        chosen_persons = chosen_persons.query(
            f"household_id not in {list(households_to_discard)}"
        )

        # drop non-unique household_id
        chosen_persons = chosen_persons.drop(columns="census_household_id")

        # give names
        first_and_middle = self.name_generator.generate_first_and_middle_names(chosen_persons)
        last_names = self.name_generator.generate_last_names(chosen_persons)
        chosen_persons = pd.concat([chosen_persons, first_and_middle, last_names], axis=1)

        # format
        n_chosen = chosen_persons.shape[0]
        chosen_persons["ssn"] = self.ssn_generator.generate(chosen_persons).ssn
        chosen_persons["entrance_time"] = pop_data.creation_time
        chosen_persons["exit_time"] = pd.NaT
        chosen_persons["alive"] = "alive"
        chosen_persons["tracked"] = True

        # # add back in extra simulants to reach desired pop size
        # remainder = n_in_households - n_chosen
        # if remainder > 0:
        #     extras = pd.DataFrame(
        #         data={
        #             "household_id": ["NA"],
        #             "state": [-1],
        #             "puma": ["NA"],
        #             "age": [np.NaN],
        #             "relation_to_household_head": ["NA"],
        #             "sex": ["NA"],
        #             "race_ethnicity": ["NA"],
        #             "first_name": ["NA"],
        #             "middle_name": ["NA"],
        #             "last_name": ["NA"],
        #             "ssn": ["NA"],
        #             "entrance_time": [pd.NaT],
        #             "exit_time": [pd.NaT],
        #             "alive": ["alive"],
        #             "tracked": [False],
        #         },
        #         index=range(remainder),
        #     )
        #     chosen_persons = pd.concat([chosen_persons, extras])

        # add typing
        chosen_persons["age"] = chosen_persons["age"].astype("float64")
        chosen_persons["state"] = chosen_persons["state"].astype("int64")
        chosen_persons = chosen_persons.set_index(pop_data.index)

        group_quarters_pop = self.sample_group_housing_inhabitants(
            n=self.config.population_size - len(chosen_persons),
            pop_data=pop_data
        )

        self.population_view.update(chosen_persons)

    def sample_group_housing_inhabitants(self, n: int, pop_data: SimulantData) -> pd.DataFrame:



    def initialize_newborns(self, pop_data: SimulantData) -> None:
        parent_ids = pop_data.user_data["parent_ids"]
        mothers = self.population_view.get(parent_ids.unique())
        new_births = pd.DataFrame(data={"parent_id": parent_ids}, index=pop_data.index)

        inherited_traits = [
            "household_id",
            "state",
            "puma",
            "race_ethnicity",
            "relation_to_household_head",
            "last_name",
            "alive",
            "tracked",
        ]

        # assign babies inherited traits
        new_births = new_births.merge(
            mothers[inherited_traits], left_on="parent_id", right_index=True
        )
        new_births["relation_to_household_head"] = new_births[
            "relation_to_household_head"
        ].map(metadata.NEWBORNS_RELATION_TO_HOUSEHOLD_HEAD_MAP)

        # assign babies uninherited traits
        new_births["age"] = 0.0
        new_births["sex"] = self.randomness.choice(
            new_births.index,
            choices=["Female", "Male"],
            p=[0.5, 0.5],
            additional_key="sex_of_child",
        )
        new_births["alive"] = "alive"
        new_births["ssn"] = self.ssn_generator.generate(new_births).ssn
        new_births["entrance_time"] = pop_data.creation_time
        new_births["exit_time"] = pd.NaT
        new_births["tracked"] = True

        # add first and middle names
        names = self.name_generator.generate_first_and_middle_names(new_births)
        new_births = pd.concat([new_births, names], axis=1)

        self.population_view.update(new_births[self.columns_created])

    def on_time_step__prepare(self, event: Event):
        """Ages simulants each time step.

        Parameters
        ----------
        event : vivarium.framework.event.Event

        """
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population["age"] += to_years(event.step_size)
        self.population_view.update(population)
