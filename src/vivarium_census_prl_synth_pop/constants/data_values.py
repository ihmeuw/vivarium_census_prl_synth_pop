from typing import NamedTuple

import pandas as pd
from scipy import stats

from vivarium_census_prl_synth_pop.constants import paths

#########################
# Population parameters #
#########################

# TODO: implement gbd call (vivarium_inputs.get_population_structure("United States"))
US_POPULATION = 333339776
PROPORTION_WORKING_AGE = 0.7756

MAX_HOUSEHOLD_SIZE = 17
PROP_POPULATION_IN_GQ = 0.03
PROBABILITY_OF_TWINS = 0.04
N_GROUP_QUARTER_TYPES = 6

# todo see if these dicts can be converted to lists. are they necessary at all?
INSTITUTIONAL_GROUP_QUARTER_IDS = {"Carceral": 0, "Nursing home": 1, "Other institutional": 2}
NONINSTITUTIONAL_GROUP_QUARTER_IDS = {
    "College": 3,
    "Military": 4,
    "Other noninstitutional": 5,
}
GROUP_QUARTER_IDS = {
    "Institutionalized group quarters population": INSTITUTIONAL_GROUP_QUARTER_IDS,
    "Noninstitutionalized group quarters population": NONINSTITUTIONAL_GROUP_QUARTER_IDS,
}
GQ_HOUSING_TYPE_MAP = {
    0: "Carceral",
    1: "Nursing home",
    2: "Other institutional",
    3: "College",
    4: "Military",
    5: "Other noninstitutional",
}
HOUSING_TYPES = ["Household"] + list(GQ_HOUSING_TYPE_MAP.values())

PROPORTION_INITIALIZATION_WITH_SSN = 0.743
PROPORTION_IMMIGRANTS_WITH_SSN = 0.625

UNKNOWN_GUARDIAN_IDX = -1
PROPORTION_GUARDIAN_TYPES = {"single_female": 0.23, "single_male": 0.05, "partnered": 0.72}

REFERENCE_PERSON_UPDATE_RELATIONSHIPS_MAP = pd.read_csv(
    paths.REFERENCE_PERSON_UPDATE_RELATIONSHIP_DATA_PATH,
)

# 96.50% probability that the mailing address is the same as the physical
#  address and the mailing ZIP code is the same as the physical ZIP code
PROBABILITY_OF_SAME_MAILING_PHYSICAL_ADDRESS = 0.9650
NO_PO_BOX = 0
MIN_PO_BOX = 1
MAX_PO_BOX = 20_000

#########################
# Synthetic Name Inputs #
#########################

# calculated by Abie from North Carolina voter registration data
PROBABILITY_OF_HYPHEN_IN_NAME = {
    "AIAN": 0.00685,
    "Asian": 0.00682,
    "Black": 0.01326,
    "Latino": 0.06842,
    "Multiracial or Other": 0.01791,
    "NHOPI": 0.02064,
    "White": 0.00474,
}

# calculated by Abie from North Carolina voter registration data
PROBABILITY_OF_SPACE_IN_NAME = {
    "AIAN": 0.00408,
    "Asian": 0.0086,
    "Black": 0.0041,
    "Latino": 0.12807,
    "Multiracial or Other": 0.02004,
    "NHOPI": 0.02064,
    "White": 0.00347,
}

#####################
# Businesses Values #
#####################

BUSINESS_NAMES_MAX_TOKENS_LENGTH = 15


WORKING_AGE = 18
EXPECTED_EMPLOYEES_PER_BUSINESS = 90.105203

YEARLY_JOB_CHANGE_RATE = 0.5  # 50 changes per 100 py

BUSINESS_MOVE_RATE_YEARLY = 0.1  # 10 changes per 100 py

PERSONAL_INCOME_PROPENSITY_DISTRIBUTION = stats.norm(loc=0.0, scale=0.812309**0.5)
EMPLOYER_INCOME_PROPENSITY_DISTRIBUTION = stats.norm(loc=0.0, scale=0.187691**0.5)


class KnownEmployer(NamedTuple):
    employer_id: int
    employer_address_id: int
    employer_name: str
    proportion: float


UNEMPLOYED = KnownEmployer(
    employer_id=0,
    employer_address_id=0,
    employer_name="unemployed",
    proportion=1 - 0.6114,
)

MILITARY = KnownEmployer(
    employer_id=1,
    employer_address_id=1,
    employer_name="Military",
    proportion=0.0032,
)

KNOWN_EMPLOYERS = [UNEMPLOYED, MILITARY]


###################
# Observer Values #
###################

COVERAGE_PROBABILITY_WIC = {
    "mothers": dict(Latino=0.993, Black=0.909, White=0.671, Other=0.882),
    0: dict(Latino=0.984, Black=0.984, White=0.7798, Other=0.984),
    1: dict(Latino=0.761, Black=0.696, White=0.514, Other=0.676),
    2: dict(Latino=0.568, Black=0.520, White=0.384, Other=0.505),
    3: dict(Latino=0.512, Black=0.469, White=0.346, Other=0.455),
    4: dict(Latino=0.287, Black=0.263, White=0.194, Other=0.255),
}


class Taxes(NamedTuple):
    PERCENT_W2_RECEIVED = 0.9465
    PERCENT_1099_RECEIVED = 0.0535
    PROBABILITY_OF_JOINT_FILER = 0.95
    PROBABILITY_OF_FILING_TAXES = 0.655


# Needed to allow HDFs to be filtered on these columns
DATA_COLUMNS = [
    # Date columns
    "year",
    "event_date",
    "survey_date",
    "tax_year",
    # State columns
    "state",
    "mailing_address_state",
]
