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
        pop["guardian"] = pd.NaN
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
        under_18_idx = household_structure.loc[
            household_structure["age_x"] < 18
        ].index

        relatives_of_rp_idx = household_structure.loc[
            ~household_structure["relation_to_household_head_y"].isin(non_relatives)
        ].index
        non_relative_of_rp_idx = household_structure.loc[
            household_structure["relation_to_household_head_y"].isin(non_relatives)
        ].index
        age_bound_idx = household_structure.loc[
            (household_structure["age_difference"] >= 20) &
            (household_structure["age_difference"] < 46) &
        ].index

        # Children of reference person - red box
        children_of_rp = household_structure.loc[
            under_18_idx.intersection(
                household_structure.loc[
                    (household_structure["relation_to_household_head_x"].isin(children)) &
                    (household_structure["relation_to_household_head_y"] == "Reference person")
                    ].index)
        ].reset_index().groupby(["ch_id"])["id"].apply(np.random.choice)
        pop.loc[children_of_rp.index, "guardian"] = children_of_rp



        # Children are reference person and have a parent in household - green box
        rp_children_with_parent = household_structure.loc[
            under_18_idx.intersection(
                household_structure.loc[
                    (household_structure["relation_to_household_head_x"] == "Reference person") &
                    (household_structure["relation_to_household_head_y"].isin(parents))
                    ].index)].reset_index().groupby(["ch_id"])["id"].apply(np.random.choice)
        pop.loc[rp_children_with_parent.index, "guardian"] = rp_children_with_parent


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

    def determine_general_pop_guardians_and_dependents(
            self, household: pd.DataFrame, child_idx: pd.Index) -> Union[Tuple, pd.Series]:
        """

        Parameters
        ----------
        household: pd.Dataframe containing columns "age", "household_id", and "relation_to_household_head"
        child_idx: pd.Index for a single simulant under 18 years old not in GQ.

        Returns
        -------
        Either a tuple or a pd.Series.  The tuple will contain either 2 or 3 pd.Series of length 1 with the
            value being the index for the guardian(s)/dependent of that simulant and the index being their index.
            Tuple will be formated either as Tuple(dependent, guardian, partner) or Tuple(dependent, guardian)
            If a pd.Series is returned it will be dependent = pd.Series(data=-1, index=child_idx) as that child has no
            assigned guardians.  -1 will be the value for "N/A".
        """

        non_relatives = ["Roommate", "Other nonrelative"]
        parents = ["Parent", "Parent-in-law"]
        partners = ["Opp-sex spouse", "Opp-sex partner", "Same-sex spouse", "Same-sex partner"]

        # Get index for reference person when child_dx is their child
        if household.loc[child_idx, "relation_to_household_head"] in [
            "Biological child", "Adopted child", "Foster child", "Stepchild"
        ]:
            dependent = pd.Series(data=household.loc[
                household["relation_to_household_head"] == "Reference person"
                ].index, index=[child_idx]
                                  )
            guardian = pd.Series(data=child_idx,
                                 index=household.loc[
                                     household["relation_to_household_head"] == "Reference person"
                                     ].index
                                 )
            partner = pd.Series(data=child_idx,
                                index=household.loc[
                                    household["relation_to_household_head"].isin(partners)
                                ].index
                                )
            if len(partner) == 1:
                dependent = pd.Series(data=[[guardian.index, partner.index]], index=[child_idx])
                return tuple([dependent, guardian, partner])
            else:
                return tuple([dependent, guardian])
        elif household.loc[child_idx, "relation_to_household_head"] in [
            "Other relative", "Grandchild", "Child-in-law", "Sibling"
        ]:
            # Find possible relatives related to reference person and in acceptible range
            related_relatives = household.loc[
                (~household["relation_to_household_head"].isin(non_relatives)) &
                (household["age"] >= household.loc[child_idx, "age"] + 20) &
                (household["age"] < household.loc[child_idx, "age"] + 46)
                ]
            if len(related_relatives) > 1:
                dependent = self.randomness.choice(child_idx,
                                                   choices=related_relatives.index,
                                                   additional_key="related_relatives_guardian_choice"
                                                   )
                guardian = pd.Series(child_idx, index=dependent)
                if household.loc[guardian, "relation_to_household_head"] in partners:
                    partner = pd.Series(child_idx, index=[household.loc[
                                                              household[
                                                                  "relation_to_household_head"] == "Reference person"
                                                              ].index]
                                        )
                    dependent = pd.Series(data=[[guardian.index, partner.index]], index=[child_idx])
                    return tuple([dependent, guardian, partner])
                else:
                    return tuple([dependent, guardian])
            elif len(related_relatives) == 1:
                dependent = pd.Series(data=related_relatives.index, index=[child_idx])
                guardian = pd.Series(data=child_idx, index=dependent)
                if household.loc[guardian, "relation_to_household_head"] in partners:
                    partner = pd.Series(child_idx, index=household.loc[
                                                              household[
                                                                  "relation_to_household_head"] == "Reference person"
                                                              ].index
                                        )
                    dependent = pd.Series(data=[[guardian.index, partner.index]], index=child_idx)
                    return tuple([dependent, guardian, partner])
                else:
                    return tuple([dependent, guardian])
            else:
                reference_person = household.loc[
                    (household["relation_to_household_head"] == "Reference person") &
                    (household["age"] >= 18)
                    ]
                if len(reference_person) == 1:
                    dependent = pd.Series(data=household.loc[
                        household["relation_to_household_head"] == "Reference person"
                        ].index, index=[child_idx]
                                          )
                    guardian = pd.Series(child_idx, dependent)
                    partner = household.loc[
                        household["relation_to_household_head"].isin(partners)
                    ].index
                    if len(partner) == 1:
                        dependent = pd.Series(data=[[guardian.index, partner.index]], index=child_idx)
                        return tuple([dependent, guardian, partner])
                    else:
                        return tuple([dependent, guardian])
                else:
                    return pd.Series(data=-1, index=[child_idx])
        # Child is roommate or other nonrelative to reference person
        elif household.loc[child_idx, "relation_toHousehold_head"] in non_relatives:
            non_relatives_of_reference_person = household.loc[
                (household["relation_to_household_head"].isin(non_relatives)) &
                (household["age"] >= household.loc[child_idx, "age"] + 20) &
                (household["age"] < household.loc[child_idx, "age"] + 46)
                ]
            if len(non_relatives_of_reference_person) > 1:
                dependent = self.randomness.choice(child_idx,
                                                   choices=non_relatives_of_reference_person.index,
                                                   additional_key="non_relative_age_bound_guardian_choice"
                                                   )
                guardian = pd.Series(data=child_idx, index=dependent)
                return tuple([dependent, guardian])
            elif len(non_relatives_of_reference_person) == 1:
                dependent = pd.Series(data=non_relatives_of_reference_person.index,
                                      index=[child_idx]
                                      )
                guardian = pd.Series(data=child_idx, index=dependent)
                return tuple([dependent, guardian])
            else:
                any_non_relative_of_reference_person = household.loc[
                    (household["relation_to_household_head"].isin(non_relatives)) &
                    (household["age"] >= 18)
                    ]
                if len(any_non_relative_of_reference_person) > 1:
                    dependent = self.randomness.choice(child_idx,
                                                       choices=non_relatives_of_reference_person.index,
                                                       additional_key="non_relative_guardian_choice"
                                                       )
                    guardian = pd.Series(data=child_idx, index=dependent)
                    return tuple([dependent, guardian])
                elif len(any_non_relative_of_reference_person) == 1:
                    dependent = pd.Series(data=any_non_relative_of_reference_person.index,
                                          index=[child_idx]
                                          )
                    guardian = pd.Series(data=child_idx, index=dependent)
                    return tuple([dependent, guardian])
                else:
                    return pd.Series(data=-1, index=[child_idx])
        # Child is reference person
        elif household.loc[child_idx, "relation_to_household_head"] == "Reference person":
            parents = household.loc[
                household["relation_to_household_head"].isin(parents)
            ]
            # Check for multiple parents, this could assign parent-in-law instead of parent
            if len(parents) > 1:
                dependent = self.randomness.choice(child_idx,
                                                   choices=parents.index,
                                                   additional_key="parents_guardian_choice"
                                                   )
                guardian = pd.Series(data=child_idx, index=dependent)
                partner = pd.Series(data=child_idx, index=parents.loc[
                    ~parents.isin(guardian)
                ].index
                                    )
                dependent = pd.Series(data=[[guardian.index, partner.index]], index=child_idx)
                return tuple([dependent, guardian, partner])
            elif len(parents) == 1:
                dependent = pd.Series(data=parents.index, index=[child_idx])
                guardian = pd.Series(data=child_idx, index=dependent)
                return tuple([dependent, guardian])
            else:
                # Find possible relatives related to child and in acceptible range
                related_relatives = household.loc[
                    (~household["relation_to_household_head"].isin(non_relatives)) &
                    (household["age"] >= household.loc[child_idx, "age"] + 20) &
                    (household["age"] < household.loc[child_idx, "age"] + 46)
                    ]
                if len(related_relatives) > 1:
                    dependent = self.randomness.choice(child_idx,
                                                       choices=related_relatives.index,
                                                       additional_key="child_ref_relatives_guardian_choice"
                                                       )
                    guardian = pd.Series(data=child_idx, index=dependent)
                    return tuple([dependent, guardian])
                elif len(related_relatives) == 1:
                    dependent = pd.Series(data=related_relatives.index, index=[child_idx])
                    guardian = pd.Series(data=child_idx, index=dependent)
                    return tuple([dependent, guardian])
                else:
                    return pd.Series(data=-1, index=child_idx)
        else:
            raise ValueError(f"Reached undefined relationship for {child_idx} in their household.")
