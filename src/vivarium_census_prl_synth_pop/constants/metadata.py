from typing import NamedTuple

import pandas as pd

####################
# Project metadata #
####################


PROJECT_NAME = "vivarium_census_prl_synth_pop"
CLUSTER_PROJECT = "proj_cost_effect"

CLUSTER_QUEUE = "all.q"
MAKE_ARTIFACT_MEM = "10G"
MAKE_ARTIFACT_CPU = "1"
MAKE_ARTIFACT_RUNTIME = "3:00:00"
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = ["Florida"]

ARTIFACT_INDEX_COLUMNS = ["sex", "age_start", "age_end", "year_start", "year_end"]

DRAW_COUNT = 1000
ARTIFACT_COLUMNS = pd.Index([f"draw_{i}" for i in range(DRAW_COUNT)])

HOUSEHOLDS_COLUMN_MAP = {
    "ST": "state",
    "SERIALNO": "census_household_id",
    "PUMA": "puma",
    "WGTP": "household_weight",
}

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
    "RELSHIPP": "relation_to_household_head",
    "SEX": "sex",
    "HISP": "latino",
    "RAC1P": "race",
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

SEX_VAR_MAP = {1: "Male", 2: "Female"}

RELATIONSHIP_TO_HOUSEHOLD_HEAD_MAP = {
    20: "Reference person",
    21: "Opp-sex spouse",
    22: "Opp-sex partner",
    23: "Same-sex spouse",
    24: "Same-sex partner",
    25: "Biological child",
    26: "Adopted child",
    27: "Stepchild",
    28: "Sibling",
    29: "Parent",
    30: "Grandchild",
    31: "Parent-in-law",
    32: "Child-in-law",
    33: "Other relative",
    34: "Roommate",
    35: "Foster child",
    36: "Other nonrelative",
    37: "Institutionalized GQ pop",
    38: "Noninstitutionalized GQ pop",
}

NEWBORNS_RELATION_TO_HOUSEHOLD_HEAD_MAP = {
    "Reference person": "Biological child",
    "Opp-sex spouse": "Biological child",
    "Opp-sex partner": "Biological child",
    "Same-sex spouse": "Biological child",
    "Same-sex partner": "Biological child",
    "Biological child": "Grandchild",
    "Adopted child": "Grandchild",
    "Stepchild": "Grandchild",
    "Sibling": "Other relative",
    "Parent": "Sibling",
    "Grandchild": "Other relative",
    "Parent-in-law": "Other relative",
    "Child-in-law": "Grandchild",
    "Other relative": "Other relative",
    "Roommate": "Other nonrelative",
    "Foster child": "Grandchild",
    "Other nonrelative": "Other nonrelative",
    "Institutionalized GQ pop": "Institutionalized GQ pop",
    "Noninstitutionalized GQ pop": "Noninstitutionalized GQ pop",
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
    "Puerto Rico": "PR",
}

P_GROUP_QUARTERS = 0.03


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()
