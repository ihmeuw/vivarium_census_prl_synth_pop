from typing import Dict, List, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
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

# Family/household relationships helper lists
NON_RELATIVES = ["Roommate", "Other nonrelative"]
CHILDREN = ["Biological child", "Adopted child", "Foster child", "Stepchild"]
CHILDREN_RELATIVES = ["Sibling", "Other relative", "Grandchild", "Child-in-law"]
PARENTS = ["Parent", "Parent-in-law"]
PARTNERS = [
    "Opp-sex spouse",
    "Opp-sex partner",
    "Same-sex spouse",
    "Same-sex partner",
]


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
        self.proportion_with_ssn = builder.lookup.build_table(
            data=data_values.PROPORTION_INITIALIZATION_WITH_SSN
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
            "guardian_1",
            "guardian_2",
            "born_in_us",
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
        persons = builder.data.load(data_keys.POPULATION.PERSONS)[
            metadata.PERSONS_COLUMNS_TO_INITIALIZE
        ]
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
        # Shift age so all households do not have the same birthday
        pop["age"] = pop["age"] + self.randomness.get_draw(pop.index, "age")
        pop["date_of_birth"] = self.start_time - pd.to_timedelta(
            np.round(pop["age"] * 365.25), unit="days"
        )

        # Deterimne if simulants have SSN
        pop["ssn"] = False
        # Give simulants born in US a SSN
        native_born_idx = pop.index[pop["born_in_us"]]
        pop.loc[native_born_idx, "ssn"] = True
        # Choose which non-native simulants get a SSN
        ssn_idx = self.randomness.filter_for_probability(
            pop.index.difference(native_born_idx),
            self.proportion_with_ssn(pop.index.difference(native_born_idx)),
            "ssn",
        )
        pop.loc[ssn_idx, "ssn"] = True

        pop["entrance_time"] = pop_data.creation_time
        pop["exit_time"] = pd.NaT
        pop["alive"] = "alive"
        # add typing
        pop["state"] = pop["state"].astype("int64")
        pop = pop.set_index(pop_data.index)

        pop = self.assign_general_population_guardians(pop)
        pop = self.assign_college_simulants_guaridans(pop)

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
        chosen_persons["age"] = self.perturb_household_age(chosen_persons)

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
            self.population_data["persons"][metadata.PERSONS_COLUMNS_TO_INITIALIZE],
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
        group_quarters["age"] = self.perturb_individual_age(group_quarters)

        return group_quarters

    def initialize_newborns(self, pop_data: SimulantData) -> None:
        parent_ids_idx = pop_data.user_data["parent_ids"]
        pop_index = pop_data.user_data["current_population_index"]
        mothers = self.population_view.get(parent_ids_idx.unique())
        households = self.population_view.subview(
            ["household_id", "relation_to_household_head"]
        ).get(pop_index)
        # Making separate subviews because SSNS will be moved to post-processing
        ssns = self.population_view.subview(["ssn"]).get(pop_index).squeeze()
        new_births = pd.DataFrame(data={"parent_id": parent_ids_idx}, index=pop_data.index)

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

        # birthday map between parent_ids and DOB (so twins get same bday)
        # note we use np.floor to guarantee birth at midnight
        dob_map = {
            parent: dob
            for (parent, dob) in zip(
                parent_ids_idx.unique(),
                pop_data.creation_time
                + pd.to_timedelta(
                    np.floor(
                        self.randomness.get_draw(parent_ids_idx.unique(), "dob")
                        * self.step_size_days
                    ),
                    unit="days",
                ),
            )
        }
        new_births["date_of_birth"] = new_births["parent_id"].map(dob_map)

        new_births["age"] = (
            pop_data.creation_time - new_births["date_of_birth"]
        ).dt.days / DAYS_PER_YEAR

        # add some noise because our randomness keys on entrance time and age,
        # so don't want people born same day to have same exact age
        # make birth-times between [0, 0.9*one_day] so that rounding will never push sims to be born the next day
        new_births["age"] += self.randomness.get_draw(new_births.index, "age") * (
            0.9 / DAYS_PER_YEAR
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
        new_births["ssn"] = True
        new_births["born_in_us"] = True

        # add first and middle names
        names = self.name_generator.generate_first_and_middle_names(new_births)
        new_births = pd.concat([new_births, names], axis=1)

        # typing
        new_births["household_id"] = new_births["household_id"].astype(int)
        # Assign guardian_1 to birth mother
        new_births["guardian_1"] = new_births["parent_id"]
        # Assign second guardian if one exists
        new_births = self.assign_second_guardian_to_newborns(new_births, households)

        self.population_view.update(new_births[self.columns_created])

    def on_time_step__prepare(self, event: Event):
        """Ages simulants each time step.

        Parameters
        ----------
        event : vivarium.framework.event.Event

        """
        population = self.population_view.subview(["age"]).get(
            event.index, query="alive == 'alive'"
        )
        population["age"] += to_years(event.step_size)
        self.population_view.update(population)

    def assign_general_population_guardians(self, pop: pd.DataFrame) -> pd.DataFrame:
        """

        Parameters
        ----------
        pop:  Population state table

        Returns
        -------
        pd.Dataframe that will be pop with two additional guardians column containing the index for that simulant's
          guardian..
        """

        # Initialize column
        pop["guardian_1"] = data_values.UNKNOWN_GUARDIAN_IDX
        pop["guardian_2"] = data_values.UNKNOWN_GUARDIAN_IDX

        # Get household structure for population to vectorize choices
        # Non-GQ population
        gen_population = pop.loc[~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)]
        under_18_idx = pop.loc[
            (pop["age"] < 18) & (~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP))
        ].index
        new_column_names = {
            "age_x": "child_age",
            "relation_to_household_head_x": "child_relation_to_household_head",
            "age_y": "member_age",
            "relation_to_household_head_y": "member_relation_to_household_head",
        }
        key_cols = ["household_id", "relation_to_household_head", "age"]

        child_households = self.get_household_structure(
            gen_population,
            query_sims=under_18_idx,
            key_columns=key_cols,
            column_names=new_column_names,
            lookup_id_level_name="child_id",
        )
        # Add age difference column to lookup age bounds for potential guardians
        child_households["age_difference"] = (
            child_households["member_age"] - child_households["child_age"]
        )

        # Children helper index groups
        # Ref_person = "Reference person"
        child_of_ref_person_idx = child_households.loc[
            child_households["child_relation_to_household_head"].isin(CHILDREN)
        ].index
        child_relative_of_ref_person_idx = child_households.loc[
            child_households["child_relation_to_household_head"].isin(CHILDREN_RELATIVES)
        ].index
        child_non_relative_of_ref_person_idx = child_households.loc[
            child_households["child_relation_to_household_head"].isin(NON_RELATIVES)
        ].index
        child_ref_person_idx = child_households.loc[
            child_households["child_relation_to_household_head"] == "Reference person"
        ].index

        # Potential guardian/household member index groups
        relatives_of_ref_person_idx = child_households.loc[
            ~child_households["member_relation_to_household_head"].isin(
                NON_RELATIVES + PARTNERS + ["Reference person"]
            )
        ].index
        non_relatives_of_ref_person_idx = child_households.loc[
            child_households["member_relation_to_household_head"].isin(NON_RELATIVES)
        ].index
        age_bound_idx = child_households.loc[
            (child_households["age_difference"] >= 18)
            & (child_households["age_difference"] < 46)
        ].index
        parents_of_ref_person_idx = child_households.loc[
            child_households["member_relation_to_household_head"].isin(PARENTS)
        ].index
        over_18_idx = child_households.loc[child_households["member_age"] >= 18].index
        ref_person_idx = child_households.loc[
            child_households["member_relation_to_household_head"] == "Reference person"
        ].index
        partners_of_ref_person_idx = child_households.loc[
            child_households["member_relation_to_household_head"].isin(PARTNERS)
        ].index

        # Assign guardians across groups
        # Children of reference person, assign reference person as guardian - red box
        ref_person_parent_ids = (
            child_households.loc[child_of_ref_person_idx.intersection(ref_person_idx)]
            .reset_index()
            .set_index("child_id")["person_id"]
        )
        pop.loc[ref_person_parent_ids.index, "guardian_1"] = ref_person_parent_ids

        # Assign partners of reference person as partners
        partners_of_ref_person_parent_ids = child_households.loc[
            child_of_ref_person_idx.intersection(partners_of_ref_person_idx)
        ]
        if len(partners_of_ref_person_parent_ids) > 0:
            # Select random partner of reference person and assign as second guardian
            partners_of_ref_person_parent_ids = self.choose_random_guardian(
                partners_of_ref_person_parent_ids, "child_id"
            )
            pop.loc[
                partners_of_ref_person_parent_ids.index, "guardian_2"
            ] = partners_of_ref_person_parent_ids

        # Children are relative of reference person with relative(s) in age bound, assign to random age bound
        # relative - orange box
        relatives_with_guardian = child_households.loc[
            child_relative_of_ref_person_idx.intersection(
                relatives_of_ref_person_idx
            ).intersection(age_bound_idx)
        ]
        if len(relatives_with_guardian) > 0:
            # Select random relative if multiple and assign as guardian
            relatives_with_guardian_ids = self.choose_random_guardian(
                relatives_with_guardian, "child_id"
            )
            pop.loc[
                relatives_with_guardian_ids.index, "guardian_1"
            ] = relatives_with_guardian_ids

        # Children are relative of reference person with no age bound relative(s), assign reference person - yellow box
        relative_ids = (
            child_households.loc[
                child_relative_of_ref_person_idx.drop(
                    relatives_with_guardian.index.unique("child_id"), level="child_id"
                ).intersection(ref_person_idx)
            ]
            .reset_index()
            .set_index("child_id")["person_id"]
        )
        pop.loc[relative_ids.index, "guardian_1"] = relative_ids
        # Assign guardian to spouse/partner of reference person
        relative_with_ref_person_partner_guardian_ids = child_households.loc[
            child_relative_of_ref_person_idx.drop(
                relatives_with_guardian.index.unique("child_id"), level="child_id"
            ).intersection(partners_of_ref_person_idx)
        ]
        if len(relative_with_ref_person_partner_guardian_ids) > 0:
            # Select random partner of reference person and assign as second guardian
            relative_with_ref_person_partner_guardian_ids = self.choose_random_guardian(
                relative_with_ref_person_partner_guardian_ids, "child_id"
            )
            pop.loc[
                relative_with_ref_person_partner_guardian_ids.index, "guardian_2"
            ] = relative_with_ref_person_partner_guardian_ids

        # Children are reference person and have a parent in household, assign max 2 random parents - green box
        child_ref_person_with_parent = child_households.loc[
            child_ref_person_idx.intersection(parents_of_ref_person_idx)
        ]
        if len(child_ref_person_with_parent) > 0:
            child_ref_person_with_parent_ids = self.choose_random_guardian(
                child_ref_person_with_parent, "child_id"
            )
            pop.loc[
                child_ref_person_with_parent_ids.index, "guardian_1"
            ] = child_ref_person_with_parent_ids

        # Child is reference person with no parent, assign to age bound relative - blue box
        child_ref_person_with_relative_ids = child_households.loc[
            child_ref_person_idx.drop(
                child_ref_person_with_parent.index.unique("child_id"), level="child_id"
            )
            .intersection(relatives_of_ref_person_idx)
            .intersection(age_bound_idx)
        ]
        if len(child_ref_person_with_relative_ids) > 0:
            # Select random relative if multiple and assign as guardian
            child_ref_person_with_relative_ids = self.choose_random_guardian(
                child_ref_person_with_relative_ids, "child_id"
            )
            pop.loc[
                child_ref_person_with_relative_ids.index, "guardian_1"
            ] = child_ref_person_with_relative_ids

        # Child is not related to and is not reference person, assign to age bound non-relative - blurple box
        non_relative_guardian = child_households.loc[
            child_non_relative_of_ref_person_idx.intersection(
                non_relatives_of_ref_person_idx
            ).intersection(age_bound_idx)
        ]
        if len(non_relative_guardian) > 0:
            # Select random non-relative if multiple and assign to guardian
            non_relative_guardian_ids = self.choose_random_guardian(
                non_relative_guardian, "child_id"
            )
            pop.loc[non_relative_guardian_ids.index, "guardian_1"] = non_relative_guardian_ids

        # Child is not reference person with no age bound non-relative, assign to adult non-relative - purple box
        other_non_relative_guardian_ids = child_households.loc[
            child_non_relative_of_ref_person_idx.drop(
                non_relative_guardian.index.unique("child_id"), level="child_id"
            )
            .intersection(non_relatives_of_ref_person_idx)
            .intersection(over_18_idx)
        ]
        if len(other_non_relative_guardian_ids) > 0:
            # Select random non-relative if multiple and assign to guardian
            other_non_relative_guardian_ids = self.choose_random_guardian(
                other_non_relative_guardian_ids, "child_id"
            )
            pop.loc[
                other_non_relative_guardian_ids.index, "guardian_1"
            ] = other_non_relative_guardian_ids

        return pop

    @staticmethod
    def choose_random_guardian(member_ids: pd.DataFrame, groupby_level: str) -> pd.Series:
        # member_ids is a subset of child_households dataframe
        # groupby_level will be index level to group by (the first level index of member_ids (child_id or mother_id).
        member_ids = (
            member_ids.reset_index()
            .groupby([groupby_level])["person_id"]
            .apply(np.random.choice)
        )
        return member_ids

    def assign_second_guardian_to_newborns(
        self, new_births: pd.DataFrame, households: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        new_births: pd.DataFrame of new births on this time step
        households: pd.DataFrame of the entire state table for the two columns of household_id
          and relation_to_household_head.

        Returns
        -------
        new_births with the additional guardian_2 column updated.
        """
        # Setup
        new_births["guardian_2"] = data_values.UNKNOWN_GUARDIAN_IDX
        key_cols = ["household_id", "relation_to_household_head"]
        new_column_names = {
            "relation_to_household_head_x": "mother_relation_to_household_head",
            "relation_to_household_head_y": "member_relation_to_household_head",
        }
        mothers_households = self.get_household_structure(
            households,
            query_sims=new_births["parent_id"],
            key_columns=key_cols,
            column_names=new_column_names,
            lookup_id_level_name="mother_id",
        )

        # Index helpers
        # Mother index groups
        mother_ref_person_idx = mothers_households.loc[
            mothers_households["mother_relation_to_household_head"] == "Reference person"
        ].index
        mother_partner_idx = mothers_households.loc[
            mothers_households["mother_relation_to_household_head"].isin(PARTNERS)
        ].index
        # Potential partner index groups
        partners_idx = mothers_households.loc[
            mothers_households["member_relation_to_household_head"].isin(PARTNERS)
        ].index
        ref_person_idx = mothers_households.loc[
            mothers_households["member_relation_to_household_head"] == "Reference person"
        ].index

        # Assign second guardian to random partner of mothers
        partner_ids = mothers_households.loc[mother_ref_person_idx.intersection(partners_idx)]
        if len(partner_ids) > 0:
            partner_ids = self.choose_random_guardian(partner_ids, "mother_id")
            new_births.loc[
                new_births["parent_id"].isin(partner_ids.index), "guardian_2"
            ] = new_births["parent_id"].map(partner_ids)

        reference_person_ids = (
            mothers_households.loc[mother_partner_idx.intersection(ref_person_idx)]
            .reset_index()
            .set_index("mother_id")["person_id"]
            .drop_duplicates()  # Checks for duplicates caused by twins in new_births
        )
        new_births.loc[
            new_births["parent_id"].isin(reference_person_ids.index), "guardian_2"
        ] = new_births["parent_id"].map(reference_person_ids)

        return new_births

    def assign_college_simulants_guaridans(self, pop: pd.DataFrame) -> pd.DataFrame:
        # Takes pop (simulation state table) and updates guardian_1 and guardian_2 atributes for college GQ simulants.

        college_sims = pop.loc[
            (pop["household_id"] == data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS["College"])
            & (pop["age"] < 24),
            ["race_ethnicity"],
        ]
        # Get subsets of pop to create data structure with household_id, reference_person_id, partner_id, sex, race
        #  for easier lookups
        reference_persons = (
            pop.loc[
                (pop["relation_to_household_head"] == "Reference person")
                & (pop["age"] >= 35)
                & (pop["age"] < 66),
                ["household_id", "sex", "race_ethnicity"],
            ]
            .reset_index()
            .set_index("household_id")
            .rename(columns={"index": "reference_person_id"})
        )
        partners_of_reference_persons = (
            pop.loc[pop["relation_to_household_head"].isin(PARTNERS), ["household_id"]]
            .reset_index()
            .rename(columns={"index": "partner_id"})
            .groupby("household_id")["partner_id"]
            .apply(np.random.choice)
        )
        potential_guardians_data = reference_persons.join(
            partners_of_reference_persons, on="household_id", how="left"
        )

        households_with_partners = potential_guardians_data.loc[
            ~potential_guardians_data["partner_id"].isnull()
        ]
        # Get single reference persons by sex
        households_with_single_female_reference_person = potential_guardians_data.loc[
            (potential_guardians_data["sex"] == "Female")
            & (potential_guardians_data["partner_id"].isnull())
        ]
        households_with_single_male_reference_person = potential_guardians_data.loc[
            (potential_guardians_data["sex"] == "Male")
            & (potential_guardians_data["partner_id"].isnull())
        ]

        # Handle percentage of single reference guardians vs reference persons with partners
        guardian_type_for_college_sims = self.randomness.choice(
            college_sims.index,
            choices=list(data_values.PROPORTION_GUARDIAN_TYPES.keys()),
            p=list(data_values.PROPORTION_GUARDIAN_TYPES.values()),
            additional_key="guardian_reference_person_relationship_type",
        )
        households_guardian_types = {
            "single_female": households_with_single_female_reference_person,
            "single_male": households_with_single_male_reference_person,
            "partnered": households_with_partners,
        }

        # Handle "Multiracial or Other" and other since we do not subset for that race
        other_college_sims_idx = college_sims.loc[
            college_sims["race_ethnicity"] == "Multiracial or Other"
        ].index
        for guardian_type in data_values.PROPORTION_GUARDIAN_TYPES:
            college_sims_with_guardian_idx = guardian_type_for_college_sims.loc[
                guardian_type_for_college_sims == guardian_type
            ].index
            other_college_sims_with_guardian_idx = other_college_sims_idx.intersection(
                college_sims_with_guardian_idx
            )
            other_guardian_ids = self.randomness.choice(
                other_college_sims_with_guardian_idx,
                choices=households_guardian_types[guardian_type]["reference_person_id"],
                additional_key=f"other_{guardian_type}_guardian_ids",
            )

            pop.loc[other_guardian_ids.index, "guardian_1"] = other_guardian_ids

            if guardian_type == "partnered":
                other_partner_ids = households_with_partners.reset_index().set_index(
                    "reference_person_id"
                )["partner_id"]
                pop.loc[other_guardian_ids.index, "guardian_2"] = pop["guardian_1"].map(
                    other_partner_ids
                )

            # Iterate through race/ethnicity and assign initial guardian
            races = [
                "AIAN",
                "Asian",
                "Black",
                "Latino",
                "NHOPI",
                "White",
            ]  # All races in state table excluding "Multiracial or other"
            for race in races:
                race_college_sims_idx = college_sims.loc[
                    college_sims["race_ethnicity"] == race
                ].index
                race_college_sims_with_guardian_idx = race_college_sims_idx.intersection(
                    college_sims_with_guardian_idx
                )
                race_reference_person_ids = households_guardian_types[guardian_type].loc[
                    households_guardian_types[guardian_type]["race_ethnicity"] == race,
                    "reference_person_id",
                ]
                if (
                    race_reference_person_ids.empty
                    and len(race_college_sims_with_guardian_idx) > 0
                ):
                    # Get all ids for race reference person who are not guardian_type
                    # This handles an unlikely edge case - leaving additional key attached to guardian_type
                    race_reference_person_ids = pd.concat(
                        [
                            df
                            for gt, df in households_guardian_types.items()
                            if gt != guardian_type
                        ]
                    )
                    race_reference_person_ids = race_reference_person_ids.loc[
                        race_reference_person_ids["race_ethnicity"] == race,
                        "reference_person_id",
                    ]
                race_guardian_ids = self.randomness.choice(
                    race_college_sims_with_guardian_idx,
                    choices=race_reference_person_ids,
                    additional_key=f"{race}_{guardian_type}_guardian_ids",
                )

                pop.loc[race_guardian_ids.index, "guardian_1"] = race_guardian_ids

                if guardian_type == "partnered":
                    race_partner_ids = households_with_partners.reset_index().set_index(
                        "reference_person_id"
                    )["partner_id"]
                    pop.loc[race_guardian_ids.index, "guardian_2"] = pop["guardian_1"].map(
                        race_partner_ids
                    )

        return pop

    @staticmethod
    def get_household_structure(
        pop: pd.DataFrame,
        query_sims: Union[pd.Series, pd.Index],
        key_columns: List,
        column_names: Dict,
        lookup_id_level_name: str,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        pop: population state table
        query_sims: Series that will be used for a lookup to subset the state table.  This will create one of the
          dataframes we will merge to create our multi-index dataframe
        key_columns: columns to subset pop
        column_names: Dictionary to map columns and their new names to.  These will generally match the key_columns arg
          and wil be of the format KEY_COLUMNS_x or KEY_COLUMN_y and then the new name for that column.
        lookup_id_level_name: Name for index level that will be first level of final dataframe.  This will be the index
          of the simulant for the "left" portion of our dataframe.

        Returns
        -------
        Multi-index dataframe with 2 levels - first being the index (id) of simulants who will be the left portion of
          our dataframe.  These ids are the same as query_sims.  Level 2 will be person_id which will be the index of
          the other members in that household.

        The following example is how we will construct a dataframe for children under 18.
        # Columns will contain data for child alongside each household member
        # This will allow us to do lookups related to both a child and other household members.
        pd.Dataframe = child_age | child_relation_to_household_head | member_age | member_relation_to_household_head
        child_id person_id|
            0        0    |  11              "Biological child               11            "Biological child
            0        1    |  11              "Biological child               35            "Reference person"
            0        2    |  11              "Biological child               7             "Adopted child"
            2        0    |  7               "Adopted child                  11            "Biological child"
            2        1    |  7               "Adopted child                  35            "Reference person"
            2        2    |  7               "Adopted child                  7             "Adopted child"

        Note: For every household with a child under 18, there N * X number of rows per household in the dataframe where
          N = number of simulants under 18 and X is the number of members in that household.  The above example is for a
          three person household with 2 children resulting in 6 rows.  This allows us to (eventually) lookup an the
          index for each child's guardian, which in this case would be [(0, 1), (2, 1)].

        # This function allows us to subset the state table to necessary columns and do more complicated lookups based
          on household structures in a vectorized way to improve performance.  Additional columns to be added to this
          data structure (for example age difference between the lookup (left) member and household member (right) shoud
          be done outside this function.
        """
        lookup_sims = (
            pop.loc[query_sims, key_columns]
            .reset_index()
            .rename(columns={"index": lookup_id_level_name})
            .set_index(["household_id", lookup_id_level_name])
        )
        household_info = (
            pop[key_columns]
            .reset_index()
            .rename(columns={"index": "person_id"})
            .set_index(["household_id", "person_id"])
        )

        household_structure = lookup_sims.merge(
            household_info, left_index=True, right_index=True
        )
        household_structure = household_structure.rename(columns=column_names).droplevel(
            "household_id"
        )

        return household_structure

    def perturb_household_age(self, simulants: pd.DataFrame) -> pd.Series:
        # Takes dataframe of households and returns a series with an applied age shift for each household.
        household_ids = simulants.loc[
            simulants["relation_to_household_head"] == "Reference person", "household_id"
        ]  # Series with index for each reference person and value is household id
        # Flip index and values to map age shift later
        reference_person_ids = pd.Series(data=household_ids.index, index=household_ids)
        age_shift_propensity = self.randomness.get_draw(
            reference_person_ids.index,  # This index is a unique list of household ids
            additional_key="household_age_perturbation",
        )
        # Convert to normal distribution with mean=0 and sd=10
        age_shift = pd.Series(
            data=stats.norm.ppf(age_shift_propensity, loc=0, scale=10),
            index=age_shift_propensity.index,
        )
        # Map age_shift to households so each member's age is perturbed the same amount
        mapped_age_shift = pd.Series(
            data=simulants["household_id"].map(age_shift), index=simulants.index
        )

        simulants["age"] = simulants["age"] + mapped_age_shift

        # Clip ages at 0 and 99
        simulants.loc[simulants["age"] < 0, "age"] = 0
        simulants.loc[simulants["age"] > 99, "age"] = 99

        return simulants["age"]

    def perturb_individual_age(self, pop: pd.DataFrame) -> pd.Series:
        # Takes dataframe containing a column "age" and returns a series of ages shifted with a normal distribution
        #   with mean=0 and sd=10 years.  If a simulant's age shift results in a negative age their perturbed age will
        #   be redrawn from the distribution. pop will be the population for group quarters or immigrants migrating to
        #   the US.
        # todo: Consider changing this function to build a distribution for each simulan instead of resampling.

        # Get age shift and redraw for simulants who get negative ages
        to_shift = pop.index
        max_iterations = 10
        for i in range(max_iterations):
            if to_shift.empty:
                break
            age_shift_propensity = self.randomness.get_draw(
                to_shift,
                additional_key=f"individual_age_perturbation_{i}",
            )
            # Convert to normal distribution with mean=0 and sd=10
            age_shift = pd.Series(
                data=stats.norm.ppf(age_shift_propensity, loc=0, scale=10),
                index=age_shift_propensity.index,
            )

            # Calculate shifted ages and see if any are negative and need to be resampled.
            shifted_ages = pop.loc[to_shift, "age"] + age_shift
            non_negative_ages_idx = shifted_ages.loc[shifted_ages >= 0].index
            pop.loc[non_negative_ages_idx, "age"] = shifted_ages.loc[non_negative_ages_idx]
            to_shift = shifted_ages.index.difference(non_negative_ages_idx)

        # Check if any simulants did not have their age shifted
        if len(to_shift) > 0:
            logger.info(
                f"Maximum iterations for resampling of age perturbation reached.  The number of simulants whose age"
                f"was not perturbed is {len(to_shift)}"
            )

        # Clip ages above 99 at 99
        pop.loc[pop["age"] > 99, "age"] = 99
        return pop["age"]
