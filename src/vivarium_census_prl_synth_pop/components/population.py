import numpy as np
import pandas as pd
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
            "guardian_1",
            "guardian_2",
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

        pop = self.assign_general_population_guardians(pop)
        # todo: Assign GQ-college guardians - MIC-3597

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

        # Generate SSNs for newborns
        # Check for SSN duplicates with existing SSNs
        to_generate = pd.Series(True, index=new_births.index)
        additional_key = 1
        while to_generate.any():
            new_births.loc[to_generate, "ssn"] = self.ssn_generator.generate(
                new_births.loc[to_generate], additional_key
            ).ssn
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
        child_households = self.get_household_structure(gen_population)

        # Children helper index groups
        # Ref_person = "Reference person"
        child_of_ref_person_idx = child_households.loc[
            child_households["child_relation_to_household_head"].isin(children)
        ].index
        child_relative_of_ref_person_idx = child_households.loc[
            child_households["child_relation_to_household_head"].isin(children_relatives)
        ].index
        child_non_relative_of_ref_person_idx = child_households.loc[
            child_households["child_relation_to_household_head"].isin(non_relatives)
        ].index
        child_ref_person_idx = child_households.loc[
            child_households["child_relation_to_household_head"] == "Reference person"
        ].index

        # Potential guardian/household member index groups
        relatives_of_ref_person_idx = child_households.loc[
            ~child_households["member_relation_to_household_head"].isin(
                non_relatives + partners + ["Reference person"]
            )
        ].index
        non_relatives_of_ref_person_idx = child_households.loc[
            child_households["member_relation_to_household_head"].isin(non_relatives)
        ].index
        age_bound_idx = child_households.loc[
            (child_households["age_difference"] >= 18)
            & (child_households["age_difference"] < 46)
        ].index
        parents_of_ref_person_idx = child_households.loc[
            child_households["member_relation_to_household_head"].isin(parents)
        ].index
        over_18_idx = child_households.loc[child_households["member_age"] >= 18].index
        ref_person_idx = child_households.loc[
            child_households["member_relation_to_household_head"] == "Reference person"
        ].index
        partners_of_ref_person_idx = child_households.loc[
            child_households["member_relation_to_household_head"].isin(partners)
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
    def get_household_structure(pop: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        pop: population state table

        Returns
        -------
        pd.DataFrame with 2 level multi-index with levels ("child_id", "person_id")
        # Columns will contain data for child alongside each household member
        # This will allow us to do lookups related to both a child and other household members
        pd.Dataframe = child_age | child_relation_to_household_head | member_age | member_relation_to_household_head | age_difference
        child_id person_id|
            0        0    |  11              "Biological child               11            "Biological child           0
            0        1    |  11              "Biological child               35            "Reference person"          24
            0        2    |  11              "Biological child               7             "Adopted child"             -4
            2        0    |  7               "Adopted child                  11            "Biological child"           4
            2        1    |  7               "Adopted child                  35            "Reference person"          28
            2        2    |  7               "Adopted child                  7             "Adopted child"              0

        Note: For every household with a child under 18, there N * X number of rows per household in the dataframe where
          N = number of simulants under 18 and X is the number of members in that household.  The above example is for a
          three person household with 2 children resulting in 6 rows.  This allows us to (eventually) lookup an the
          index for each child's guardian, which in this case would be [(0, 1), (2, 1)].
        """

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
        household_structure = under_18.merge(
            household_info, left_index=True, right_index=True
        )
        household_structure = household_structure.rename(
            columns={
                "age_x": "child_age",
                "relation_to_household_head_x": "child_relation_to_household_head",
                "age_y": "member_age",
                "relation_to_household_head_y": "member_relation_to_household_head",
            }
        ).droplevel("household_id")
        household_structure["age_difference"] = (
            household_structure["member_age"] - household_structure["child_age"]
        )

        return household_structure

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
        mothers_households = self.get_mothers_household_structure(
            new_births["parent_id"], households
        )

        # Index helpers
        # Mother index groups
        mother_ref_person_idx = mothers_households.loc[
            mothers_households["mother_relation_to_household_head"] == "Reference person"
        ].index
        mother_partner_idx = mothers_households.loc[
            mothers_households["mother_relation_to_household_head"].isin(partners)
        ].index
        # Potential partner index groups
        partners_idx = mothers_households.loc[
            mothers_households["member_relation_to_household_head"].isin(partners)
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

        reference_person_ids = mothers_households.loc[
            mother_partner_idx.intersection(ref_person_idx)
        ]
        if len(reference_person_ids) > 0:
            reference_person_ids = self.choose_random_guardian(
                reference_person_ids, "mother_id"
            )
            new_births.loc[
                new_births["parent_id"].isin(reference_person_ids.index), "guardian_2"
            ] = new_births["parent_id"].map(reference_person_ids)

        return new_births

    @staticmethod
    def get_mothers_household_structure(
        mothers_idx: pd.Series, households: pd.DataFrame
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        mothers_idx: Series of index values corresponding to mothers who gave birth this time step
        households: 2 column dataframe of state table with containing "household_id" and "relation_to_household_head"
          columns.

        Returns
        -------
        Dataframe with 2 index levels "mother_id" and "person_id" and 2 columns "mother_relation_to_household_head" and
          "member_relation_to_household_head".  This is the same schema we used with creating the data structure for
          child households.
        """
        # todo: Future improvement would be to refactor the two data wrangling functions.

        mothers = (
            households.loc[mothers_idx]
            .reset_index()
            .rename(columns={"index": "mother_id"})
            .set_index(["household_id", "mother_id"])
        )
        household_info = (
            households.reset_index()
            .rename(columns={"index": "person_id"})
            .set_index(["household_id", "person_id"])
        )

        mother_households = mothers.merge(household_info, left_index=True, right_index=True)
        mother_households = mother_households.rename(
            columns={
                "relation_to_household_head_x": "mother_relation_to_household_head",
                "relation_to_household_head_y": "member_relation_to_household_head",
            }
        ).droplevel("household_id")

        return mother_households


# Family/household relationships helper lists
non_relatives = ["Roommate", "Other nonrelative"]
children = ["Biological child", "Adopted child", "Foster child", "Stepchild"]
children_relatives = ["Sibling", "Other relative", "Grandchild", "Child-in-law"]
parents = ["Parent", "Parent-in-law"]
partners = [
    "Opp-sex spouse",
    "Opp-sex partner",
    "Same-sex spouse",
    "Same-sex partner",
]
