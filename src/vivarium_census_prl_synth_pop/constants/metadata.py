import pandas as pd

from typing import NamedTuple


####################
# Project metadata #
####################


PROJECT_NAME = 'vivarium_census_prl_synth_pop'
CLUSTER_PROJECT = 'proj_cost_effect'
# # TODO use proj_csu if a csu project
# CLUSTER_PROJECT = 'proj_csu'

CLUSTER_QUEUE = 'all.q'
MAKE_ARTIFACT_MEM = '10G'
MAKE_ARTIFACT_CPU = '1'
MAKE_ARTIFACT_RUNTIME = '3:00:00'
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    "Florida"
]

ARTIFACT_INDEX_COLUMNS = [
    'sex',
    'age_start',
    'age_end',
    'year_start',
    'year_end',
]

DRAW_COUNT = 1000
ARTIFACT_COLUMNS = pd.Index([f'draw_{i}' for i in range(DRAW_COUNT)])

HOUSEHOLDS_COLUMN_MAP = {
    'ST': 'state',
    'SERIALNO': 'hh_id',
    'PUMA': 'puma',
    'WGTP': 'hh_weight',
}

CENSUS_STATE_IDS = {
    'Alabama': 1,
    'Alaska': 2,
    'Arizona': 4,
    'Arkansas': 5,
    'California': 6,
    'Colorado': 8,
    'Connecticut': 9,
    'Delaware': 10,
    'District of Columbia': 11,
    'Florida': 12,
    'Georgia': 13,
    'Hawaii': 15,
    'Idaho': 16,
    'Illinois': 17,
    'Indiana': 18,
    'Iowa': 19,
    'Kansas': 20,
    'Kentucky': 21,
    'Louisiana': 22,
    'Maine': 23,
    'Maryland': 24,
    'Massachusetts': 25,
    'Michigan': 26,
    'Minnesota': 27,
    'Mississippi': 28,
    'Missouri': 29,
    'Montana': 30,
    'Nebraska': 31,
    'Nevada': 32,
    'New Hampshire': 33,
    'New Jersey': 34,
    'New Mexico': 35,
    'New York': 36,
    'North Carolina': 37,
    'North Dakota': 38,
    'Ohio': 39,
    'Oklahoma': 40,
    'Oregon': 41,
    'Pennsylvania': 42,
    'Rhode Island': 44,
    'South Carolina': 45,
    'South Dakota': 46,
    'Tennessee': 47,
    'Texas': 48,
    'Utah': 49,
    'Vermont': 50,
    'Virginia': 51,
    'Washington': 53,
    'West Virginia': 54,
    'Wisconsin': 55,
    'Wyoming': 56,
    'Puerto Rico': 72
}

PERSONS_COLUMNS_MAP = {
    'SERIALNO': 'hh_id',
    'AGEP': 'age',
    'RELSHIPP': 'relation_to_hh_head',
    'SEX': 'sex',
    'HISP': 'latino',
    'RAC1P': 'race',
}

LATINO_VAR_MAP = {
    i: (1 if i == 1 else 0) for i in range(1, 25)
}

RACE_ETH_VAR_MAP = {
    0: 'Latino',
    1: 'White',
    2: 'Black',
    3: 'AIAN',
    4: 'AIAN',
    5: 'AIAN',
    6: 'Asian',
    7: 'NHOPI',
    8: 'Multiracial or Other',
    9: 'Multiracial or Other',
}

SEX_VAR_MAP = {1: 'Male', 2: 'Female'}

RELSHIP_TO_HH_HEAD_MAP = {
    20: 'Reference person',
    21: 'Opp-sex spouse',
    22: 'Opp-sex partner',
    23: 'Same-sex spouse',
    24: 'Same-sex partner',
    25: 'Biological child',
    26: 'Adopted child',
    27: 'Stepchild',
    28: 'Sibling',
    29: 'Parent',
    30: 'Grandchild',
    31: 'Parent-in-law',
    32: 'Child-in-law',
    33: 'Other relative',
    34: 'Roommate',
    35: 'Foster child',
    36: 'Other nonrelative',
    37: 'Institutionalized GQ pop',
    38: 'Noninstitutionalized GQ pop',
}

MAX_HH_SIZE = 17


class __Scenarios(NamedTuple):
    baseline: str = 'baseline'
    # TODO - add scenarios here


SCENARIOS = __Scenarios()
