from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.time import get_time_stamp
from vivarium_public_health.utilities import DAYS_PER_YEAR, to_years

from vivarium_census_prl_synth_pop.constants import data_keys, data_values, metadata
from vivarium_census_prl_synth_pop.utilities import (
    sample_acs_persons,
    sample_acs_standard_households,
    vectorized_choice,
)

# Family/household relationships helper lists
NON_RELATIVES = ["Roommate or housemate", "Other nonrelative"]
CHILDREN = ["Biological child", "Adopted child", "Foster child", "Stepchild"]
CHILDREN_RELATIVES = ["Sibling", "Other relative", "Grandchild", "Child-in-law"]
PARENTS = ["Parent", "Parent-in-law"]
PARTNERS = [
    "Opposite-sex spouse",
    "Opposite-sex unmarried partner",
    "Same-sex spouse",
    "Same-sex unmarried partner",
]


class Population(Component):
    CONFIGURATION_DEFAULTS = {"us_population_size": data_values.US_POPULATION}

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [
            "household_id",
            "relationship_to_reference_person",
            "sex",
            "age",
            "date_of_birth",
            "race_ethnicity",
            "first_name_id",
            "middle_name_id",
            "last_name_id",
            "has_ssn",
            "alive",
            "entrance_time",
            "exit_time",
            "guardian_1",
            "guardian_2",
            "born_in_us",
        ]

    @property
    def colums_required(self) -> List[str]:
        return ["state_id_for_lookup"]

    @property
    def initialization_requires(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": ["state_id_for_lookup"],
        }

    def setup(self, builder: Builder):
        self.config = builder.configuration.population
        self.seed = builder.configuration.randomness.random_seed
        self.randomness = builder.randomness.get_stream("household_sampling")
        self.proportion_with_ssn = builder.lookup.build_table(
            data=data_values.PROPORTION_INITIALIZATION_WITH_SSN
        )
        self.proportion_immigrants_with_ssn = builder.lookup.build_table(
            data=data_values.PROPORTION_IMMIGRANTS_WITH_SSN
        )

        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.step_size_days = builder.configuration.time.step_size
        self.households = builder.components.get_component("households")

        # HACK: ACS data must be loaded in setup, since it comes from the artifact;
        # however, we only use it while creating the initial population.
        self.population_data = self._load_population_data(builder)

        self.updated_relationship_to_reference_person = builder.value.register_value_producer(
            "updated_relationship_to_reference_person",
            source=self.get_updated_relationship_to_reference_person,
            requires_columns=[
                "relationship_to_reference_person",
                "household_id",
                "date_of_birth",
                "guardian_1",
            ],
        )

    def _load_population_data(self, builder: Builder):
        households = builder.data.load(data_keys.POPULATION.HOUSEHOLDS)
        persons = builder.data.load(data_keys.POPULATION.PERSONS)
        return {"households": households, "persons": persons}

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # at start of sim, generate base population
        if pop_data.creation_time < self.start_time:
            self.generate_initial_population(pop_data)
            # HACK: ACS data must be loaded in setup, since it comes from the artifact;
            # however, we don't need it after creating the initial population and want to avoid keeping
            # it in memory.
            self.population_data = None
        elif pop_data.user_data["creation_type"] == "fertility":
            self.initialize_newborns(pop_data)
        elif pop_data.user_data["creation_type"] == "gq_immigrants":
            self.initialize_gq_immigrants(pop_data)
        elif pop_data.user_data["creation_type"] == "non_reference_person_immigrants":
            self.initialize_non_reference_person_immigrants(pop_data)
        elif pop_data.user_data["creation_type"] == "household_immigrants":
            self.initialize_household_immigrants(pop_data)
        else:
            raise ValueError("Unknown simulant creation type")

    def generate_initial_population(self, pop_data: SimulantData) -> None:
        target_gq_pop_size = int(
            self.config.population_size * data_values.PROP_POPULATION_IN_GQ
        )
        target_standard_housing_pop_size = self.config.population_size - target_gq_pop_size

        acs_households = self.population_data["households"]
        is_standard_household = acs_households["household_type"] == "Housing unit"
        chosen_households, chosen_persons = sample_acs_standard_households(
            target_number_sims=target_standard_housing_pop_size,
            acs_households=acs_households[is_standard_household],
            acs_persons=self.population_data["persons"],
            randomness=self.randomness,
            num_households=None,
        )
        non_gq_simulants = self.initialize_standard_households(
            acs_households=chosen_households,
            acs_persons=chosen_persons,
            pop_data=pop_data,
        )

        # Household sampling won't exactly hit its target population size -- we fill
        # in the remainder with GQ
        actual_gq_pop_size = self.config.population_size - len(non_gq_simulants)

        # TODO: If we did a merge when creating the artifact that added the household_type
        # column to the persons data, we would not have to do this quasi-merge
        # (using the households data to determine which persons rows are GQ) at run time.
        gq_household_ids = acs_households[~is_standard_household][
            "census_household_id"
        ].unique()
        gq_persons = self.population_data["persons"][
            self.population_data["persons"]["census_household_id"].isin(gq_household_ids)
        ]

        chosen_persons = sample_acs_persons(
            actual_gq_pop_size,
            acs_persons=gq_persons,
            randomness=self.randomness,
            additional_key="choose_gq_persons",
        )
        gq_simulants = self.initialize_group_quarters(
            acs_persons=chosen_persons,
            pop_data=pop_data,
        )

        pop = pd.concat([non_gq_simulants, gq_simulants]).set_index(pop_data.index)
        pop = self.initialize_simulant_link_columns(
            new_simulants=pop, existing_simulants=None
        )

        self.population_view.update(pop[self.columns_created])

    def initialize_standard_households(
        self,
        acs_households: pd.DataFrame,
        acs_persons: pd.DataFrame,
        pop_data: SimulantData,
    ) -> pd.DataFrame:
        new_simulants = self.initialize_new_simulants_from_acs(acs_persons, pop_data)

        household_ids = self.households.create_households(
            num_households=len(acs_households),
            states_pumas=acs_households[["state", "puma"]].rename(
                columns={"state": "state_id"}
            ),
        )

        household_id_mapping = pd.Series(
            household_ids, index=acs_households["acs_sample_household_id"]
        )
        new_simulants["household_id"] = new_simulants["acs_sample_household_id"].map(
            household_id_mapping
        )

        # NOTE: Must happen after household_ids have been assigned, because it depends
        # on household structure.
        self.perturb_household_age(new_simulants)

        return new_simulants

    def initialize_group_quarters(
        self,
        acs_persons: pd.DataFrame,
        pop_data: SimulantData,
    ) -> pd.DataFrame:
        new_simulants = self.initialize_new_simulants_from_acs(acs_persons, pop_data)
        self.perturb_individual_age(new_simulants)

        noninstitutionalized = new_simulants.loc[
            new_simulants["relationship_to_reference_person"]
            == "Noninstitutionalized group quarters population"
        ].copy()
        institutionalized = new_simulants.loc[
            new_simulants["relationship_to_reference_person"]
            == "Institutionalized group quarters population"
        ].copy()

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

        new_simulants = pd.concat([noninstitutionalized, institutionalized])

        return new_simulants

    def initialize_newborns(self, pop_data: SimulantData) -> None:
        parent_ids_idx = pop_data.user_data["parent_ids"]
        pop_index = pop_data.user_data["current_population_index"]
        mothers = self.population_view.get(parent_ids_idx.unique())
        households = self.population_view.subview(
            ["household_id", "relationship_to_reference_person"]
        ).get(pop_index)
        new_births = pd.DataFrame(data={"parent_id": parent_ids_idx}, index=pop_data.index)
        new_births = self.initialize_new_simulants(new_births, pop_data)

        inherited_traits = [
            "household_id",
            "race_ethnicity",
            "relationship_to_reference_person",
            "last_name_id",
        ]

        # assign babies inherited traits
        new_births = new_births.merge(
            mothers[inherited_traits], left_on="parent_id", right_index=True
        )
        new_births["relationship_to_reference_person"] = (
            new_births["relationship_to_reference_person"]
            .map(metadata.NEWBORNS_RELATIONSHIP_TO_REFERENCE_PERSON_MAP)
            .astype(new_births["relationship_to_reference_person"].dtype)
        )

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
        ).astype(pd.CategoricalDtype(categories=metadata.SEXES))
        new_births["has_ssn"] = True
        new_births["born_in_us"] = True

        # add first and middle names
        new_births["first_name_id"] = new_births.index
        new_births["middle_name_id"] = new_births.index

        # typing
        new_births["household_id"] = new_births["household_id"].astype(int)
        # Assign guardian_1 to birth mother
        new_births["guardian_1"] = new_births["parent_id"]
        # Assign second guardian if one exists
        new_births = self.assign_second_guardian_to_newborns(new_births, households)

        self.population_view.update(new_births[self.columns_created])

    def initialize_gq_immigrants(self, pop_data: SimulantData) -> None:
        """
        Initializes simulants who have newly immigrated into
        the US into group quarters.

        Parameters
        ----------
        pop_data
            The SimulantData on the simulant creation call, which must supply the ACS
            rows the simulants should be based on.
        """
        new_simulants = self.initialize_group_quarters(
            pop_data.user_data["acs_persons"], pop_data
        ).set_index(pop_data.index)

        existing_simulants = self.population_view.get(
            pop_data.user_data["current_population_index"],
            query="alive == 'alive' and in_united_states == True and tracked == True",
        )
        new_simulants = self.initialize_simulant_link_columns(
            new_simulants, existing_simulants
        )

        # We need to do this each time we initialize more simulants, otherwise the
        # addition of NaNs in household_id before it was initialized makes
        # the column a float.
        new_simulants["household_id"] = new_simulants["household_id"].astype(int)

        self.population_view.update(new_simulants[self.columns_created])

    def initialize_non_reference_person_immigrants(self, pop_data: SimulantData) -> None:
        """
        Initializes simulants who have newly immigrated into
        the US into existing, non-GQ households.

        Parameters
        ----------
        pop_data
            The SimulantData on the simulant creation call, which must supply the ACS
            rows the simulants should be based on.
        """
        new_simulants = self.initialize_new_simulants_from_acs(
            pop_data.user_data["acs_persons"], pop_data
        ).set_index(pop_data.index)
        self.perturb_individual_age(new_simulants)

        existing_simulants = self.population_view.get(
            pop_data.user_data["current_population_index"],
            query="alive == 'alive' and in_united_states == True and tracked == True",
        )
        existing_household_ids = existing_simulants[
            ~existing_simulants["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)
        ]["household_id"].unique()

        if len(existing_household_ids) == 0:
            # Extremely rare edge case when there are no existing households to assign to.
            # Should only ever happen with very small population sizes.
            # We just create new households and make the immigrants the reference people.
            household_ids = self.households.create_households(
                num_households=len(new_simulants),
            )
            new_simulants["household_id"] = household_ids
            new_simulants["relationship_to_reference_person"] = "Reference person"
        else:
            # TODO: For now, we do a simple random sample of households.
            # As part of future PUMA perturbation work, we plan to actually use
            # the PUMA value in the ACS row in choosing the household they should join.
            new_simulants["household_id"] = vectorized_choice(
                existing_household_ids,
                n_to_choose=len(new_simulants),
                randomness_stream=self.randomness,
                additional_key="household_id",
            )

        # We avoid non-reference-person immigrants having these relationships, because
        # otherwise they may cause there to be an impossible number of said relationships
        # within a household, i.e. >2 parents or >1 spouse/partner
        new_simulants["relationship_to_reference_person"] = (
            new_simulants["relationship_to_reference_person"]
            .replace(
                {
                    "Parent": "Other relative",
                    "Parent-in-law": "Other relative",
                    "Opposite-sex spouse": "Other relative",
                    "Same-sex spouse": "Other relative",
                    "Opposite-sex unmarried partner": "Other nonrelative",
                    "Same-sex unmarried partner": "Other nonrelative",
                }
            )
            .astype(existing_simulants["relationship_to_reference_person"].dtype)
        )

        new_simulants = self.initialize_simulant_link_columns(
            new_simulants, existing_simulants
        )

        # We need to do this each time we initialize more simulants, otherwise the
        # addition of NaNs in household_id before it was initialized makes
        # the column a float.
        new_simulants["household_id"] = new_simulants["household_id"].astype(int)

        self.population_view.update(new_simulants[self.columns_created])

    def initialize_household_immigrants(self, pop_data: SimulantData) -> None:
        """
        Initializes entire household units of simulants who have newly immigrated into
        the US.

        Parameters
        ----------
        pop_data
            The SimulantData on the simulant creation call, which must supply the ACS
            rows (both household and persons) the simulants should be based on.
        """
        new_simulants = self.initialize_standard_households(
            acs_households=pop_data.user_data["acs_households"],
            acs_persons=pop_data.user_data["acs_persons"],
            pop_data=pop_data,
        ).set_index(pop_data.index)

        existing_simulants = self.population_view.get(
            pop_data.user_data["current_population_index"],
            query="alive == 'alive' and in_united_states == True and tracked == True",
        )
        new_simulants = self.initialize_simulant_link_columns(
            new_simulants, existing_simulants
        )

        # We need to do this each time we initialize more simulants, otherwise the
        # addition of NaNs in household_id before it was initialized makes
        # the column a float.
        new_simulants["household_id"] = new_simulants["household_id"].astype(int)

        self.population_view.update(new_simulants[self.columns_created])

    def on_time_step_cleanup(self, event: Event):
        """Ages simulants each time step.
        Parameters
        ----------
        event : vivarium.framework.event.Event

        """
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population["age"] += to_years(event.step_size)

        self.population_view.update(population)

    @staticmethod
    def initialize_new_simulants(
        new_simulants: pd.DataFrame, pop_data: SimulantData
    ) -> pd.DataFrame:
        """
        Performs basic setup that is the same anytime new simulants are created.

        Parameters
        ----------
        new_simulants
            Dataframe of new simulants. No columns are required.
        pop_data
            The SimulantData of the simulant creation event.

        Returns
        -------
        new_simulants with the bookkeeping columns entrance_time, exit_time, and alive added.
        """
        new_simulants["entrance_time"] = pop_data.creation_time
        new_simulants["exit_time"] = pd.NaT
        new_simulants["alive"] = "alive"

        return new_simulants

    def initialize_new_simulants_from_acs(
        self,
        new_simulants: pd.DataFrame,
        pop_data: SimulantData,
        immigrants: bool = False,
    ) -> pd.DataFrame:
        """
        Initializes simulants that are based on ACS persons rows.

        Parameters
        ----------
        new_simulants
            Dataframe of the new simulants, containing ACS columns.
        pop_data
            The SimulantData of the simulant creation event.
        immigrants
            Whether or not the new_simulants are immigrants, which determines how many will have SSNs.

        Returns
        -------
        new_simulants with added columns: date_of_birth, ssn, entrance_time, exit_time, alive
        """
        # Add basic, non-ACS columns
        new_simulants = self.initialize_new_simulants(new_simulants, pop_data)

        # Age is recorded in ACS as an integer, floored.
        # We shift ages to a random floating point value between the reported age and the next.
        new_simulants["age"] = new_simulants["age"].astype("float64")
        new_simulants["age"] = new_simulants["age"] + self.randomness.get_draw(
            new_simulants.index, "age"
        )
        new_simulants["date_of_birth"] = pop_data.creation_time - pd.to_timedelta(
            np.round(new_simulants["age"] * 365.25), unit="days"
        )

        # Determine if simulants have SSN
        proportion_with_ssn = (
            self.proportion_immigrants_with_ssn if immigrants else self.proportion_with_ssn
        )
        new_simulants["has_ssn"] = False
        # Give simulants born in US a SSN
        native_born_idx = new_simulants.index[new_simulants["born_in_us"]]
        new_simulants.loc[native_born_idx, "has_ssn"] = True
        # Choose which non-native simulants get a SSN
        ssn_idx = self.randomness.filter_for_probability(
            new_simulants.index.difference(native_born_idx),
            proportion_with_ssn(new_simulants.index.difference(native_born_idx)),
            "has_ssn",
        )
        new_simulants.loc[ssn_idx, "has_ssn"] = True

        return new_simulants

    def initialize_simulant_link_columns(
        self, new_simulants: pd.DataFrame, existing_simulants: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Initializes columns that link simulants to one another.
        These are name (currently, only last name actually links) and guardian_1 and guardian_2.
        This depends on all simulants being initialized and having their final indices.

        Note that new simulants may be linked to other new simulants.

        Parameters
        ----------
        new_simulants
            DataFrame of new simulants who should be assigned values for these linking columns,
            with all the other columns already initialized,
            and having the final simulant IDs as the index.
        existing_simulants
            Population state table of simulants already present in the simulation.
            Simulant ID is the index.
            These simulants will *not* be modified in any way.
            Should be None if there is not yet any existing population.

        Returns
        -------
        new_simulants with added columns first_name_id, middle_name_id, last_name_id, guardian_1 and guardian_2
        """
        # Names are a bit of an exception because the link is not to a simulant's ID/index, but
        # to a simulant's *value of the same column*.
        # Note that this is not an academic distinction: a simulant could
        # be linked to a reference person's last_name_id, and later *become* a reference person who a
        # new simulant gets linked to.
        # Due to this exception, all rows in the full_pop dataframe need to have name IDs,
        # which means pre-linking name IDs need to be assigned to new_simulants before
        # creating the full_pop dataframe.

        # Give initial, pre-linking name ids
        new_simulants["first_name_id"] = new_simulants.index
        new_simulants["middle_name_id"] = new_simulants.index
        new_simulants["last_name_id"] = new_simulants.index

        columns_needed_for_linking = [
            "household_id",
            "relationship_to_reference_person",
            "sex",
            "age",
            "race_ethnicity",
            # Due to above exception, last_name_id is needed to link to
            "last_name_id",
        ]
        full_pop = new_simulants[columns_needed_for_linking]
        if existing_simulants is not None:
            full_pop = pd.concat([full_pop, existing_simulants[columns_needed_for_linking]])

        # NOTE: Right now first and middle name don't link to other simulants, but they may in the future
        new_simulants = self.assign_linked_last_name_ids(new_simulants, full_pop)

        # Initialize guardian columns
        new_simulants["guardian_1"] = data_values.UNKNOWN_GUARDIAN_IDX
        new_simulants["guardian_2"] = data_values.UNKNOWN_GUARDIAN_IDX

        new_simulants = self.assign_general_population_guardians(new_simulants, full_pop)
        new_simulants = self.assign_college_simulants_guardians(new_simulants, full_pop)

        return new_simulants

    def assign_general_population_guardians(
        self, simulants_to_assign: pd.DataFrame, pop: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Assign guardians found in pop to simulants_to_assign who are <18 and not in GQ.

        Parameters
        ----------
        simulants_to_assign
            Simulants who may be assigned guardians.
            Should already have guardian_1 and guardian_2 columns, which will not be changed if
            no guardian links are found for that simulant.
        pop
            Population from which to find guardians.
            Indices must be the final simulant IDs.

        Returns
        -------
        simulants_to_assign with updated guardian_1 and guardian_2 columns.
        """

        # Get household structure for population to vectorize choices
        # Non-GQ population
        gen_population = pop.loc[~pop["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP)]
        under_18 = simulants_to_assign.loc[
            (simulants_to_assign["age"] < 18)
            & (~simulants_to_assign["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP))
        ]
        new_column_names = {
            "age_x": "child_age",
            "relationship_to_reference_person_x": "child_relationship_to_reference_person",
            "age_y": "member_age",
            "relationship_to_reference_person_y": "member_relationship_to_reference_person",
        }
        key_cols = ["household_id", "relationship_to_reference_person", "age"]

        child_households = self.get_household_structure(
            gen_population,
            query_sims=under_18,
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
            child_households["child_relationship_to_reference_person"].isin(CHILDREN)
        ].index
        child_relative_of_ref_person_idx = child_households.loc[
            child_households["child_relationship_to_reference_person"].isin(
                CHILDREN_RELATIVES
            )
        ].index
        child_non_relative_of_ref_person_idx = child_households.loc[
            child_households["child_relationship_to_reference_person"].isin(NON_RELATIVES)
        ].index
        child_ref_person_idx = child_households.loc[
            child_households["child_relationship_to_reference_person"] == "Reference person"
        ].index

        # Potential guardian/household member index groups
        relatives_of_ref_person_idx = child_households.loc[
            ~child_households["member_relationship_to_reference_person"].isin(
                NON_RELATIVES + PARTNERS + ["Reference person"]
            )
        ].index
        non_relatives_of_ref_person_idx = child_households.loc[
            child_households["member_relationship_to_reference_person"].isin(NON_RELATIVES)
        ].index
        age_bound_idx = child_households.loc[
            (child_households["age_difference"] >= 18)
            & (child_households["age_difference"] < 46)
        ].index
        parents_of_ref_person_idx = child_households.loc[
            child_households["member_relationship_to_reference_person"].isin(PARENTS)
        ].index
        over_18_idx = child_households.loc[child_households["member_age"] >= 18].index
        ref_person_idx = child_households.loc[
            child_households["member_relationship_to_reference_person"] == "Reference person"
        ].index
        partners_of_ref_person_idx = child_households.loc[
            child_households["member_relationship_to_reference_person"].isin(PARTNERS)
        ].index

        # Assign guardians across groups
        # Children of reference person, assign reference person as guardian - red box
        ref_person_parent_ids = (
            child_households.loc[child_of_ref_person_idx.intersection(ref_person_idx)]
            .reset_index()
            .set_index("child_id")["person_id"]
        )
        simulants_to_assign.loc[
            ref_person_parent_ids.index, "guardian_1"
        ] = ref_person_parent_ids

        # Assign partners of reference person as partners
        partners_of_ref_person_parent_ids = child_households.loc[
            child_of_ref_person_idx.intersection(partners_of_ref_person_idx)
        ]
        if len(partners_of_ref_person_parent_ids) > 0:
            # Select random partner of reference person and assign as second guardian
            partners_of_ref_person_parent_ids = self.choose_random_guardian(
                partners_of_ref_person_parent_ids, "child_id"
            )
            simulants_to_assign.loc[
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
            simulants_to_assign.loc[
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
        simulants_to_assign.loc[relative_ids.index, "guardian_1"] = relative_ids
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
            simulants_to_assign.loc[
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
            simulants_to_assign.loc[
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
            simulants_to_assign.loc[
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
            simulants_to_assign.loc[
                non_relative_guardian_ids.index, "guardian_1"
            ] = non_relative_guardian_ids

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
            simulants_to_assign.loc[
                other_non_relative_guardian_ids.index, "guardian_1"
            ] = other_non_relative_guardian_ids

        return simulants_to_assign

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
          and relationship_to_reference_person.

        Returns
        -------
        new_births with the additional guardian_2 column updated.
        """
        # Setup
        new_births["guardian_2"] = data_values.UNKNOWN_GUARDIAN_IDX
        key_cols = ["household_id", "relationship_to_reference_person"]
        new_column_names = {
            "relationship_to_reference_person_x": "mother_relationship_to_reference_person",
            "relationship_to_reference_person_y": "member_relationship_to_reference_person",
        }
        mothers_households = self.get_household_structure(
            households,
            query_sims=households.loc[new_births["parent_id"]],
            key_columns=key_cols,
            column_names=new_column_names,
            lookup_id_level_name="mother_id",
        )

        # Index helpers
        # Mother index groups
        mother_ref_person_idx = mothers_households.loc[
            mothers_households["mother_relationship_to_reference_person"]
            == "Reference person"
        ].index
        mother_partner_idx = mothers_households.loc[
            mothers_households["mother_relationship_to_reference_person"].isin(PARTNERS)
        ].index
        # Potential partner index groups
        partners_idx = mothers_households.loc[
            mothers_households["member_relationship_to_reference_person"].isin(PARTNERS)
        ].index
        ref_person_idx = mothers_households.loc[
            mothers_households["member_relationship_to_reference_person"]
            == "Reference person"
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

    def assign_college_simulants_guardians(
        self, simulants_to_assign: pd.DataFrame, pop: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Assign guardians, found in pop, to simulants_to_assign who are <24 and in college.

        Parameters
        ----------
        simulants_to_assign
            Simulants to assign guardians (where applicable).
            Should already have guardian_1 and guardian_2 columns, which will not be changed if
            no guardian links are found for that simulant.
        pop
            Population from which to find guardians.
            Indices must be the final simulant IDs.

        Returns
        -------
        simulants_to_assign with updated guardian_1 and guardian_2 columns.
        """

        college_sims = simulants_to_assign.loc[
            (
                simulants_to_assign["household_id"]
                == data_values.NONINSTITUTIONAL_GROUP_QUARTER_IDS["College"]
            )
            & (simulants_to_assign["age"] < 24),
            ["race_ethnicity"],
        ]
        # Get subsets of pop to create data structure with household_id, reference_person_id, partner_id, sex, race
        #  for easier lookups
        reference_persons = (
            pop.loc[
                (pop["relationship_to_reference_person"] == "Reference person")
                & (pop["age"] >= 35)
                & (pop["age"] < 66),
                ["household_id", "sex", "race_ethnicity"],
            ]
            .reset_index()
            .set_index("household_id")
            .rename(columns={"index": "reference_person_id"})
        )
        partners_of_reference_persons = (
            pop.loc[pop["relationship_to_reference_person"].isin(PARTNERS), ["household_id"]]
            .reset_index()
            .rename(columns={"index": "partner_id"})
            .groupby("household_id")["partner_id"]
            .first()
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

            simulants_to_assign.loc[
                other_guardian_ids.index, "guardian_1"
            ] = other_guardian_ids

            if guardian_type == "partnered":
                other_partner_ids = households_with_partners.reset_index().set_index(
                    "reference_person_id"
                )["partner_id"]
                simulants_to_assign.loc[
                    other_guardian_ids.index, "guardian_2"
                ] = simulants_to_assign["guardian_1"].map(other_partner_ids)

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

                # If there is still nobody to assign as the guardian (which should basically never happen),
                # we give up and leave simulants_to_assign without guardians
                if not race_reference_person_ids.empty:
                    race_guardian_ids = self.randomness.choice(
                        race_college_sims_with_guardian_idx,
                        choices=race_reference_person_ids,
                        additional_key=f"{race}_{guardian_type}_guardian_ids",
                    )

                    simulants_to_assign.loc[
                        race_guardian_ids.index, "guardian_1"
                    ] = race_guardian_ids

                    if guardian_type == "partnered":
                        race_partner_ids = households_with_partners.reset_index().set_index(
                            "reference_person_id"
                        )["partner_id"]
                        simulants_to_assign.loc[
                            race_guardian_ids.index, "guardian_2"
                        ] = simulants_to_assign["guardian_1"].map(race_partner_ids)

        return simulants_to_assign

    @staticmethod
    def get_household_structure(
        pop: pd.DataFrame,
        query_sims: pd.DataFrame,
        key_columns: List,
        column_names: Dict,
        lookup_id_level_name: str,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        pop: population state table
        query_sims: The first dataframe we will merge to create our multi-index dataframe
        key_columns: columns to subset pop
        column_names: Dictionary to map columns and their new names to.  These will generally match the key_columns arg
          and wil be of the format KEY_COLUMNS_x or KEY_COLUMN_y and then the new name for that column.
        lookup_id_level_name: Name for index level that will be first level of final dataframe.  This will be the index
          of the simulant for the "left" portion of our dataframe.

        Returns
        -------
        Multi-index dataframe with 2 levels - first being the index (id) of simulants who will be the left portion of
          our dataframe.  These ids are the same as the index of query_sims.  Level 2 will be person_id which will be the index of
          the other members in that household.

        The following example is how we will construct a dataframe for children under 18.
        # Columns will contain data for child alongside each household member
        # This will allow us to do lookups related to both a child and other household members.
        pd.Dataframe = child_age | child_relationship_to_reference_person | member_age | member_relationship_to_reference_person
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
            query_sims[key_columns]
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

    def perturb_household_age(self, simulants: pd.DataFrame) -> None:
        # Takes dataframe of households and returns a series with an applied age shift for each household.
        household_ids = simulants.loc[
            simulants["relationship_to_reference_person"] == "Reference person",
            "household_id",
        ]  # Series with index for each reference person and value is household id
        # Flip index and values to map age shift later
        reference_person_ids = pd.Series(data=household_ids.index, index=household_ids)
        age_shift_propensity = self.randomness.get_draw(
            reference_person_ids.index,  # This index is a unique list of household ids
            additional_key="household_age_perturbation",
        )
        # Convert to normal distribution with mean=0 and sd=10
        age_shift = pd.Series(
            data=stats.norm.ppf(age_shift_propensity, loc=0, scale=1),
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

        # Keep DOB in sync -- should this be a pipeline instead of a column?
        simulants["date_of_birth"] = simulants.entrance_time - pd.to_timedelta(
            np.round(simulants["age"] * 365.25), unit="days"
        )

    def perturb_individual_age(self, pop: pd.DataFrame) -> None:
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
                data=stats.norm.ppf(age_shift_propensity, loc=0, scale=1),
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

        # Keep DOB in sync -- should this be a pipeline instead of a column?
        pop["date_of_birth"] = pop.entrance_time - pd.to_timedelta(
            np.round(pop["age"] * 365.25), unit="days"
        )

    @staticmethod
    def assign_linked_last_name_ids(
        simulants_to_assign: pd.DataFrame,
        pop: pd.DataFrame,
    ) -> pd.Series:
        """
        Sets last_name_ids for simulants_to_assign such that, if a simulant
        is a relative of the reference person in a non-GQ household, their last_name_id matches the reference person's.

        Parameters
        ----------
        simulants_to_assign
            Dataframe of simulants who may be assigned a linked last name ID.
            This dataframe should already include a last_name_id column, which will not be changed if no link is made.
            It should also include household_id and relationship_to_reference_person columns.
        pop
            Dataframe of full population, which includes all reference people the last_name_id may be linked to.

        Returns
        -------
        simulants_to_assign with updated last_name_ids.
        """

        # Match last names for relatives of reference person
        relatives_idx = simulants_to_assign.index[
            (
                ~simulants_to_assign["relationship_to_reference_person"].isin(
                    NON_RELATIVES + ["Reference person"]
                )
            )
            & (~simulants_to_assign["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP))
        ]

        # Get mapping from household_id to current reference person last_name_id
        reference_person_last_name_ids = pop.loc[
            pop["relationship_to_reference_person"] == "Reference person",
            ["household_id", "last_name_id"],
        ].set_index("household_id")["last_name_id"]

        # NOTE: Because we are mutating the very value we are linking to, it would break if
        # a simulants_to_assign row were both linked *to* and linked *from* here!
        # That only one or the other occurs is guaranteed by our relationship_to_reference_person
        # criteria above.
        # We have to set the type because it will have become a float with the addition of
        # NaNs.
        simulants_to_assign.loc[relatives_idx, "last_name_id"] = (
            simulants_to_assign.loc[relatives_idx, "household_id"]
            .map(reference_person_last_name_ids)
            .astype(int)
        )

        return simulants_to_assign

    def get_updated_relationship_to_reference_person(self, idx: pd.Index) -> pd.Series:
        """
        Chooses the oldest member of all households that lack a reference person
        and assigns them as the reference person. Updates all other relationships to
        be relative to this new reference person.
        """
        population = self.population_view.get(
            idx, query="alive == 'alive' and in_united_states == True and tracked == True"
        )

        # Find standard households that do not have a reference person
        household_ids_with_reference_person = population.loc[
            population["relationship_to_reference_person"] == "Reference person",
            "household_id",
        ]
        standard_household_ids = population.loc[
            ~population["household_id"].isin(data_values.GQ_HOUSING_TYPE_MAP), "household_id"
        ].unique()
        household_ids_without_reference_person = set(standard_household_ids) - set(
            household_ids_with_reference_person
        )
        households_to_update_idx = population.index[
            population["household_id"].isin(household_ids_without_reference_person)
        ]

        # Find the oldest member in each household and make them new reference person
        # This is a series with household_id as the index and the new reference person as the value
        new_reference_persons = (
            population.loc[households_to_update_idx].groupby(["household_id"])["age"].idxmax()
        )
        # Preserve old relationship of new reference person before assigning
        # them as new reference persons
        new_reference_person_prev_relationship = population.loc[
            new_reference_persons, ["household_id", "relationship_to_reference_person"]
        ]
        population.loc[
            new_reference_persons, "relationship_to_reference_person"
        ] = "Reference person"

        # Update simulants born in simulation as biological children to their
        # mother if mother is new reference person
        biological_children_idx = population.index[
            (population["date_of_birth"] > self.start_time)
            & (
                population["guardian_1"]
                == population["household_id"].map(
                    new_reference_person_prev_relationship.reset_index().set_index(
                        "household_id"
                    )["index"]
                )
            )
        ].intersection(households_to_update_idx)
        population.loc[
            biological_children_idx, "relationship_to_reference_person"
        ] = "Biological child"

        # Update other household members relationship to new reference person
        for relationship in new_reference_person_prev_relationship[
            "relationship_to_reference_person"
        ].unique():
            relationship_map = data_values.REFERENCE_PERSON_UPDATE_RELATIONSHIPS_MAP.loc[
                data_values.REFERENCE_PERSON_UPDATE_RELATIONSHIPS_MAP[
                    "new_reference_person_relationship_to_old_reference_person"
                ]
                == relationship
            ].set_index("relationship_to_old_reference_person")

            household_ids_to_update = new_reference_person_prev_relationship.loc[
                new_reference_person_prev_relationship["relationship_to_reference_person"]
                == relationship,
                "household_id",
            ]
            simulants_to_update_idx = (
                population.index[population["household_id"].isin(household_ids_to_update)]
                .difference(new_reference_person_prev_relationship.index)
                .difference(biological_children_idx)
            )
            # Update relationships
            population.loc[
                simulants_to_update_idx, "relationship_to_reference_person"
            ] = population["relationship_to_reference_person"].map(
                relationship_map["relationship_to_new_reference_person"]
            )

        # Handle extreme edge cases where there would not be a value to map to.
        population.loc[households_to_update_idx, "relationship_to_reference_person"].fillna(
            "Other nonrelative"
        )

        return population["relationship_to_reference_person"]
