import numpy as np
import pandas as pd
from typing import (Tuple, Union)
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.utilities import DAYS_PER_YEAR, to_years

from vivarium_census_prl_synth_pop.components.synthetic_pii import (
    NameGenerator,
    SSNGenerator,
)
from vivarium_census_prl_synth_pop.constants import data_keys, data_values, metadata
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
        self.proportion_with_no_ssn = builder.lookup.build_table(
            data=data_values.PROPORTION_INITIALIZATION_NO_SSN
        )
        self.proportion_newborns_no_ssn = builder.lookup.build_table(
            data=data_values.PROPORTION_NEWBORNS_NO_SSN
        )

        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.step_size_days = builder.configuration.time.step_size

        self.columns_created = [
            "household_id",
            "state",
            "puma",
            "relation_to_household_head",
            "sex",
            "age",
            "date_of_birth",
            "race_ethnicity",
            "first_name",
            "middle_name",
            "last_name",
            "ssn",
            "alive",
            "entrance_time",
            "exit_time",
            "housing_type",
            "guardian",
        ]
        self.register_simulants = builder.randomness.register_simulants
        self.population_view = builder.population.get_view(self.columns_created)
        self.population_data = self._load_population_data(builder)

        builder.population.initializes_simulants(
            self.initialize_simulants, creates_columns=self.columns_created
        )

        builder.event.register_listener("time_step__cleanup", self.on_time_step__prepare)

    def _load_population_data(self, builder: Builder):
        households = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)
        persons = builder.data.load(data_keys.POPULATION.PERSONS)
        return {"households": households, "persons": persons}

    def initialize_simulants(self, pop_data: SimulantData) -> None:
        # at start of sim, generate base population
        if pop_data.creation_time < self.start_time:
            self.generate_initial_population(pop_data)
        # if new simulants are born into the sim
        else:
            self.initialize_newborns(pop_data)

    def generate_initial_population(self, pop_data: SimulantData) -> None:
        target_gq_pop_size = int(
            self.config.population_size * data_values.PROP_POPULATION_IN_GQ
        )
        target_standard_housing_pop_size = self.config.population_size - target_gq_pop_size

        chosen_households = self.choose_standard_households(target_standard_housing_pop_size)
        chosen_group_quarters = self.choose_group_quarters(
            self.config.population_size - len(chosen_households)
        )

        pop = pd.concat([chosen_households, chosen_group_quarters])

        # pull back on state and puma
        pop = pd.merge(
            pop,
            self.population_data["households"][["state", "puma", "census_household_id"]],
            on="census_household_id",
            how="left",
        )

        # drop non-unique household_id
        pop = pop.drop(columns="census_household_id")

        # give names
        first_and_middle = self.name_generator.generate_first_and_middle_names(pop)
        last_names = self.name_generator.generate_last_names(pop)
        pop = pd.concat([pop, first_and_middle, last_names], axis=1)

        pop["age"] = pop["age"].astype("float64")
        pop["age"] = pop["age"] + self.randomness.get_draw(pop.index, "age")
        pop["date_of_birth"] = self.start_time - pd.to_timedelta(
            np.round(pop["age"] * 365.25), unit="days"
        )

        # Add Social Security Numbers
        pop["ssn"] = self.ssn_generator.generate(pop).ssn
        pop["ssn"] = self.ssn_generator.remove_ssn(pop["ssn"], self.proportion_with_no_ssn)

        pop["entrance_time"] = pop_data.creation_time
        pop["exit_time"] = pd.NaT
        pop["alive"] = "alive"
        # add typing
        pop["state"] = pop["state"].astype("int64")
        pop = pop.set_index(pop_data.index)

        # todo: initialize guardians for simulants: Should this be the last thing we do in initialize simulants?
        pop = self.assign_gen_pop_guardians(pop)

        self.population_view.update(pop)

    def choose_standard_households(self, target_number_sims: int) -> pd.DataFrame:
        # oversample households
        chosen_households = vectorized_choice(
            options=self.population_data["households"]["census_household_id"],
            n_to_choose=target_number_sims,
            randomness_stream=self.randomness,
            weights=self.population_data["households"]["household_weight"],
        )

        # create unique id for resampled households
        chosen_households = pd.DataFrame(
            {
                "census_household_id": chosen_households,
                "household_id": np.arange(
                    data_values.N_GROUP_QUARTER_TYPES,
                    len(chosen_households) + data_values.N_GROUP_QUARTER_TYPES,
                ),
            }
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
                                target_number_sims:, "household_id"
                                ].unique()

        chosen_persons = chosen_persons.loc[
            ~chosen_persons["household_id"].isin(households_to_discard)
        ]

        chosen_persons["housing_type"] = "Standard"

        return chosen_persons

    def choose_group_quarters(self, target_number_sims: int) -> pd.Series:
        group_quarters = self.population_data["households"]["census_household_id"]
        group_quarters = group_quarters.loc[
            ["GQ" in household_id for household_id in group_quarters]
        ]

        # group quarters each house one person per census_household_id
        # they have NA household weights, but appropriately weighted person weights.
        chosen_units = vectorized_choice(
            options=group_quarters,
            n_to_choose=target_number_sims,
            randomness_stream=self.randomness,
            weights=self.population_data["households"][["person_weight"]],
        )

        # get simulants per GQ unit
        chosen_persons = pd.merge(
            chosen_units,
            self.population_data["persons"],
            on="census_household_id",
            how="left",
        )

        noninstitutionalized = chosen_persons.loc[
            chosen_persons["relation_to_household_head"] == "Noninstitutionalized GQ pop"
            ]
        institutionalized = chosen_persons.loc[
            chosen_persons["relation_to_household_head"] == "Institutionalized GQ pop"
            ]
        noninstitutionalized = noninstitutionalized.copy()
        institutionalized = institutionalized.copy()

        noninstitutionalized_gq_types = vectorized_choice(
            options=list(data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS.values()),
            n_to_choose=len(noninstitutionalized),
            randomness_stream=self.randomness,
        )

        institutionalized_gq_types = vectorized_choice(
            options=list(data_values.INSTITUTIONAL_GROUP_QUARTER_IDS.values()),
            n_to_choose=len(institutionalized),
            randomness_stream=self.randomness,
        )

        noninstitutionalized["household_id"] = noninstitutionalized_gq_types
        institutionalized["household_id"] = institutionalized_gq_types

        group_quarters = pd.concat([noninstitutionalized, institutionalized])
        group_quarters["housing_type"] = group_quarters["household_id"].map(
            data_values.GQ_HOUSING_TYPE_MAP
        )

        return group_quarters

    def initialize_newborns(self, pop_data: SimulantData) -> None:
        parent_ids = pop_data.user_data["parent_ids"]
        pop_index = pop_data.user_data["current_population_index"]
        mothers = self.population_view.get(parent_ids.unique())
        ssns = self.population_view.subview(['ssn']).get(pop_index).squeeze()
        new_births = pd.DataFrame(data={"parent_id": parent_ids}, index=pop_data.index)

        inherited_traits = [
            "household_id",
            "housing_type",
            "state",
            "puma",
            "race_ethnicity",
            "relation_to_household_head",
            "last_name",
            "alive",
        ]

        # assign babies inherited traits
        new_births = new_births.merge(
            mothers[inherited_traits], left_on="parent_id", right_index=True
        )
        new_births["relation_to_household_head"] = new_births[
            "relation_to_household_head"
        ].map(metadata.NEWBORNS_RELATION_TO_HOUSEHOLD_HEAD_MAP)

        # Make age negative so age + step size gets simulants correct age at the end of the time step (ref line 284)
        new_births["age"] = -self.randomness.get_draw(new_births.index, "age") * (
                self.step_size_days / DAYS_PER_YEAR
        )
        new_births["date_of_birth"] = pop_data.creation_time - pd.to_timedelta(
            np.round(new_births["age"] * DAYS_PER_YEAR), unit="days"
        )

        new_births["sex"] = self.randomness.choice(
            new_births.index,
            choices=["Female", "Male"],
            p=[0.5, 0.5],
            additional_key="sex_of_child",
        )
        new_births["alive"] = "alive"
        new_births["entrance_time"] = pop_data.creation_time
        new_births["exit_time"] = pd.NaT

        # Generate SSNs for newborns
        # Check for SSN duplicates with existing SSNs
        to_generate = pd.Series(True, index=new_births.index)
        additional_key = 1
        while to_generate.any():
            new_births.loc[to_generate, "ssn"] = self.ssn_generator.generate(new_births.loc[to_generate],
                                                                             additional_key).ssn
            additional_key += 1
            duplicate_mask = to_generate & new_births["ssn"].isin(ssns)
            ssns = pd.concat([ssns, new_births.loc[to_generate & ~duplicate_mask, "ssn"]])
            # Adds SSNs from new births to population SSNs series that are not duplicates
            to_generate = duplicate_mask

        new_births["ssn"] = self.ssn_generator.remove_ssn(
            new_births["ssn"], self.proportion_newborns_no_ssn
        )

        # add first and middle names
        names = self.name_generator.generate_first_and_middle_names(new_births)
        new_births = pd.concat([new_births, names], axis=1)

        # typing
        new_births["household_id"] = new_births["household_id"].astype(int)
        # todo: Add additional guardian if necessary

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

    def assign_gen_pop_guardians(self, pop: pd.DataFrame) -> pd.DataFrame:
        """

        Parameters
        ----------
        pop: State stable of simulants

        Returns
        -------
        pd.Dataframe that will be pop with an additional guardians column containing the index for that simulant's
          guardian..
        """
        # Initialize column
        pop["guardian_1"] = pd.NaN
        pop["guardian_2"] = pd.NaN
        # Helper lists
        non_relatives = ["Roommate", "Other nonrelative"]
        children = ["Biological child", "Adopted child", "Foster child", "Stepchild"]
        parents = ["Parent", "Parent-in-law"]
        partners = ["Opp-sex spouse", "Opp-sex partner", "Same-sex spouse", "Same-sex partner"]

        # Get household structure for population to vectorize choices
        # Non-GQ population
        gen_population = pop.loc[
            ~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)
        ]
        household_structure = self.get_household_structure(gen_population)

        # Get indexes of subgroups to work through decision tree
        # todo: Make subsets to get series
        # Children helper index groups
        children_of_rp_idx = household_structure.loc[
            household_structure["relation_to_household_head_x"].isin(children)
        ].index
        relative_children_of_rp_idx = household_structure.loc[
                ~household_structure["relation_to_household_head_x"].isin(non_relatives)
            ].index
        children_is_rp_idx = household_structure.loc[
            household_structure["relation_to_household_head_x"] == "Reference person"
        ].index

        # Potential guardian index groups
        relatives_of_rp_idx = household_structure.loc[
            ~household_structure["relation_to_household_head_y"].isin(non_relatives)
        ].index
        non_relatives_of_rp_idx = household_structure.loc[
            household_structure["relation_to_household_head_y"].isin(non_relatives)
        ].index
        age_bound_idx = household_structure.loc[
            (household_structure["age_difference"] >= 20) &
            (household_structure["age_difference"] < 46)
        ].index
        parents_idx = household_structure.loc[
            household_structure["relation_to_household_head_y"].isin(parents)
        ].index
        over_18_idx = household_structure.loc[
            household_structure["age_y"] >= 18
        ].index

        # Assign guardians across groups
        # Children of reference person - red box
        reference_person_ids = household_structure.loc[
            children_of_rp_idx
            .intersection(
                household_structure.loc[
                    household_structure["relation_to_household_head_y"] == "Reference person"
                ].index)
        ].reset_index()["id"]
        pop.loc[children_of_rp_idx, "guardian_1"] = reference_person_ids
        # Assign partners of reference person as partners
        partners_of_reference_person_ids = household_structure.loc[
            children_of_rp_idx
            .intersection(
                household_structure.loc[
                    household_structure["relation_to_household_head_y"].isin(partners)
                ].index)
        ].reset_index().groupby(["ch_id"])["id"].apply(np.random.choice)
        pop.loc[children_of_rp_idx, "guardian_2"] = partners_of_reference_person_ids

        # Children are relative of reference person with relative(s) in age bound - orange box
        age_bound_relatives_of_rp_ids = household_structure.loc[
            relative_children_of_rp_idx
            .intersection(relatives_of_rp_idx)
            .intersection(age_bound_idx)
        ]
        # Save index to find difference for yellow box (where there will be no relatives in age bounds)
        children_relatives_of_rp_with_age_bound_relative_idx = age_bound_relatives_of_rp_ids.index
        age_bound_relatives_of_rp_ids = (
            age_bound_relatives_of_rp_ids
            .reset_index()
            .groupby(["ch_id"])["id"]
            .apply(np.random.choice)
        )
        pop.loc[
            age_bound_relatives_of_rp_ids.index, "guardian_1"
        ] = age_bound_relatives_of_rp_ids

        # Children are relative of reference person with no age bound relative(s) - yellow box
        rp_with_no_age_bound_relative_ids = household_structure.loc[
            relative_children_of_rp_idx
            .difference(children_relatives_of_rp_with_age_bound_relative_idx)
            .intersection(household_structure.loc[
                              household_structure["relation_to_household_head_y"] == "Reference person"
                              ].index
                          )
        ].reset_index()["id"]
        pop.loc[
            rp_with_no_age_bound_relative_ids.index, "guardian_1"
        ] = rp_with_no_age_bound_relative_ids
        # Assign guardian to spouse/partner of reference person
        partners_of_rp_for_relatives_ids = household_structure.loc[
            relative_children_of_rp_idx
            .difference(children_relatives_of_rp_with_age_bound_relative_idx)
            .intersection(household_structure.loc[
                              household_structure["relation_to_household_head_y"] == "Reference person"
                              ].index
                          )
        ].reset_index().groupby(["ch_id"])["id"].apply(np.random.choice)
        pop.loc[partners_of_rp_for_relatives_ids.index, "guardian_2"] = partners_of_rp_for_relatives_ids

        # Children are reference person and have a parent in household - green box
        rp_children_with_parent = household_structure.loc[
            children_is_rp_idx
            .intersection(parents_idx)
        ].reset_index().groupby(["ch_id"])["id"].apply(np.random.choice)
        pop.loc[rp_children_with_parent.index, "guardian_1"] = rp_children_with_parent

        # Child is reference person with no parent, assign to age bound relative - blue box
        age_bound_relatives_of_children_rp_ids = household_structure.loc[
            children_is_rp_idx
            .intersection(relatives_of_rp_idx)
            .difference(parents_idx)
            .intersection(age_bound_idx)
        ].reset_index().groupby(["ch_id"])["id"].apply(np.random.choice)
        pop.loc[age_bound_relatives_of_children_rp_ids.index, "guardian_1"] = age_bound_relatives_of_children_rp_ids

        # Child is not related to and is not reference person, assign to age bound non-relative - blurple box
        age_bound_non_relatives_of_rp_ids = household_structure.loc[
            non_relatives_of_rp_idx
            .intersection(age_bound_idx)
        ].reset_index().groupby(["ch_id"])["id"].apply(np.random.choice)
        pop.loc[age_bound_non_relatives_of_rp_ids.index, "guardian_1"] = age_bound_non_relatives_of_rp_ids

        # Child is not reference person, no age bound non-relative but there is adult non-relative - purple box
        adult_non_relatives_of_rp_ids = household_structure.loc[
            non_relatives_of_rp_idx
            .intersection(over_18_idx)
            .difference(age_bound_idx)
        ].reset_index().groupby(["ch_id"])["id"].apply(np.random.choice)
        pop.loc[adult_non_relatives_of_rp_ids.index, "guardian_1"] = adult_non_relatives_of_rp_ids
        # todo: Work through decision tree for < 24 year olds in GQ

    def get_household_structure(self, pop: pd.DataFrame) -> pd.DataFrame:
        # Returns a 3 level multi-index with levels ["household_id", "child_id", "person_id"]
        # Columns will contain data for child alongside each household member
        # This will allow us to do lookups related to both a child and other household members
        under_18 = (
                    pop.loc[pop["age"] < 18, ["household_id", "relation_to_household_head", "age"]]
                    .reset_index()
                    .rename(columns={"index": "child_id"})
                    .set_index(["household_id", "child_id"])
                )
        household_info = (
            pop[["household_id", "relation_to_household_head", "age"]]
            .reset_index()
            .rename(columns={"index": "person_id"})
            .set_index(["household_id", "person_id"])
        )
        # Merge dataframes to cast on household_id and child_id
        household_structure = under_18.merge(household_info, left_index=True, right_index=True)
        household_structure["age_difference"] = household_structure["age_y"] - household_structure["age_x"]

        return household_structure
