from typing import NamedTuple

import pandas as pd

####################
# Project metadata #
####################


PROJECT_NAME = "vivarium_census_prl_synth_pop"
CLUSTER_PROJECT = "proj_simscience_prod"

CLUSTER_QUEUE = "all.q"
MAKE_ARTIFACT_MEM = "10G"
MAKE_ARTIFACT_CPU = "1"
MAKE_ARTIFACT_RUNTIME = "3:00:00"
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = ["United States of America", "Florida"]
UNITED_STATES_LOCATIONS = [
    # Leave empty for all locations in ACS dataset,
]

ARTIFACT_INDEX_COLUMNS = ["sex", "age_start", "age_end", "year_start", "year_end"]

DRAW_COUNT = 1000
ARTIFACT_COLUMNS = pd.Index([f"draw_{i}" for i in range(DRAW_COUNT)])

HOUSEHOLDS_COLUMN_MAP = {
    "ST": "state",
    "SERIALNO": "census_household_id",
    "PUMA": "puma",
    "WGTP": "household_weight",
    "TYPEHUGQ": "household_type",
}

HOUSEHOLD_TYPE_MAP = {
    1: "Housing unit",
    2: "Institutional group quarters",
    3: "Noninstitutional group quarters",
}
HOUSEHOLD_TYPES = list(HOUSEHOLD_TYPE_MAP.values())

PERSONS_COLUMNS_TO_INITIALIZE = [
    "census_household_id",
    "age",
    "relationship_to_reference_person",
    "sex",
    "race_ethnicity",
    "born_in_us",
]

CENSUS_STATE_IDS = {
    "NA": -1,
    "Alabama": 1,
    "Alaska": 2,
    "Arizona": 4,
    "Arkansas": 5,
    "California": 6,
    "Colorado": 8,
    "Connecticut": 9,
    "Delaware": 10,
    "District of Columbia": 11,
    "Florida": 12,
    "Georgia": 13,
    "Hawaii": 15,
    "Idaho": 16,
    "Illinois": 17,
    "Indiana": 18,
    "Iowa": 19,
    "Kansas": 20,
    "Kentucky": 21,
    "Louisiana": 22,
    "Maine": 23,
    "Maryland": 24,
    "Massachusetts": 25,
    "Michigan": 26,
    "Minnesota": 27,
    "Mississippi": 28,
    "Missouri": 29,
    "Montana": 30,
    "Nebraska": 31,
    "Nevada": 32,
    "New Hampshire": 33,
    "New Jersey": 34,
    "New Mexico": 35,
    "New York": 36,
    "North Carolina": 37,
    "North Dakota": 38,
    "Ohio": 39,
    "Oklahoma": 40,
    "Oregon": 41,
    "Pennsylvania": 42,
    "Rhode Island": 44,
    "South Carolina": 45,
    "South Dakota": 46,
    "Tennessee": 47,
    "Texas": 48,
    "Utah": 49,
    "Vermont": 50,
    "Virginia": 51,
    "Washington": 53,
    "West Virginia": 54,
    "Wisconsin": 55,
    "Wyoming": 56,
    "Puerto Rico": 72,
}

PERSONS_COLUMNS_MAP = {
    "ST": "state",
    "SERIALNO": "census_household_id",
    "AGEP": "age",
    "RELSHIPP": "relationship_to_reference_person",
    "SEX": "sex",
    "HISP": "latino",
    "RAC1P": "race",
    "NATIVITY": "born_in_us",
    "MIG": "immigrated_in_last_year",
    "PWGTP": "person_weight",
}

SUBSET_PERSONS_COLUMNS_MAP = {
    "ST": "state",
    "SERIALNO": "census_household_id",
    "PWGTP": "person_weight",
}

LATINO_VAR_MAP = {i: (1 if i == 1 else 0) for i in range(1, 25)}


RACE_ETHNICITY_VAR_MAP = {
    0: "Latino",
    1: "White",
    2: "Black",
    3: "AIAN",
    4: "AIAN",
    5: "AIAN",
    6: "Asian",
    7: "NHOPI",
    8: "Multiracial or Other",
    9: "Multiracial or Other",
}
RACE_ETHNICITIES = list({race: None for race in RACE_ETHNICITY_VAR_MAP.values()})


SEX_VAR_MAP = {1: "Male", 2: "Female"}
SEXES = list(SEX_VAR_MAP.values())

RELATIONSHIP_TO_REFERENCE_PERSON_MAP = {
    20: "Reference person",
    21: "Opposite-sex spouse",
    22: "Opposite-sex unmarried partner",
    23: "Same-sex spouse",
    24: "Same-sex unmarried partner",
    25: "Biological child",
    26: "Adopted child",
    27: "Stepchild",
    28: "Sibling",
    29: "Parent",
    30: "Grandchild",
    31: "Parent-in-law",
    32: "Child-in-law",
    33: "Other relative",
    34: "Roommate or housemate",
    35: "Foster child",
    36: "Other nonrelative",
    37: "Institutionalized group quarters population",
    38: "Noninstitutionalized group quarters population",
}
RELATIONSHIPS = list(RELATIONSHIP_TO_REFERENCE_PERSON_MAP.values())

NEWBORNS_RELATIONSHIP_TO_REFERENCE_PERSON_MAP = {
    "Reference person": "Biological child",
    "Opposite-sex spouse": "Biological child",
    "Opposite-sex unmarried partner": "Biological child",
    "Same-sex spouse": "Biological child",
    "Same-sex unmarried partner": "Biological child",
    "Biological child": "Grandchild",
    "Adopted child": "Grandchild",
    "Stepchild": "Grandchild",
    "Sibling": "Other relative",
    "Parent": "Sibling",
    "Grandchild": "Other relative",
    "Parent-in-law": "Other relative",
    "Child-in-law": "Grandchild",
    "Other relative": "Other relative",
    "Roommate or housemate": "Other nonrelative",
    "Foster child": "Grandchild",
    "Other nonrelative": "Other nonrelative",
    "Institutionalized group quarters population": "Institutionalized group quarters population",
    "Noninstitutionalized group quarters population": "Noninstitutionalized group quarters population",
}

US_STATE_ABBRV_MAP = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    # "Puerto Rico": "PR",
}

P_GROUP_QUARTERS = 0.03

NATIVITY_MAP = {1: True, 2: False}

MIGRATION_MAP = {1.0: False, 2.0: True, 3.0: False}


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()

PRIORITY_MAP = {
    # When people emigrate, we no longer do anything else with them, so doing
    # this earlier saves pointless computation.
    "person_emigration.on_time_step": 4,
    "household_emigration.on_time_step": 4,
    # Businesses must come after domestic migration
    # components, so that domestic migration can trigger employment change.
    # 5 is the default, but we are explicit here to show ordering.
    "person_migration.on_time_step": 5,
    "household_migration.on_time_step": 5,
    "businesses.on_time_step": 6,
    "immigration.on_time_step": 7,
}


class DatasetNames:
    """Container for Dataset names"""

    ACS = "american_community_survey"
    CENSUS = "decennial_census"
    CPS = "current_population_survey"
    SSA = "social_security"
    TAXES_1040 = "taxes_1040"
    TAXES_W2_1099 = "taxes_w2_and_1099"
    TAXES_DEPENDENTS = "taxes_dependents"
    WIC = "women_infants_and_children"


COPY_HOUSEHOLD_MEMBER_COLS = {
    "age": "copy_age",
    "date_of_birth": "copy_date_of_birth",
    "has_ssn": "copy_ssn",
}


YEAR_AGGREGATION_VALUE = 3000
