from typing import NamedTuple

#########################
# Population parameters #
#########################

PROP_POPULATION_IN_GQ = 0.03
PROBABILITY_OF_TWINS = 0.04
N_GROUP_QUARTER_TYPES = 6
INSTITUTIONAL_GROUP_QUARTER_IDS = {"Carceral": 0, "Nursing home": 1, "Other institutional": 2}
NONINSTITUTIONAL_GROUP_QUARTER_IDS = {
    "College": 3,
    "Military": 4,
    "Other non-institutional": 5,
}
GQ_HOUSING_TYPE_MAP = {
    0: "Carceral",
    1: "Nursing home",
    2: "Other institutional",
    3: "College",
    4: "Military",
    5: "Other non-institutional",
}
PROPORTION_INITIALIZATION_WITH_SSN = 0.743
PROPORTION_PERSONS_LEAVING_COUNTRY = 0.05
PROPORTION_HOUSEHOLDS_LEAVING_COUNTRY = 0.05

UNKNOWN_GUARDIAN_IDX = -1
PROPORTION_GUARDIAN_TYPES = {"single_female": 0.23, "single_male": 0.05, "partnered": 0.72}

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

# data from https://www.jec.senate.gov/public/index.cfm/republicans/fl/
BUSINESS_NAMES_MAX_TOKENS_LENGTH = 15
PROPORTION_WORKFORCE_EMPLOYED = {"Florida": 0.576}
WORKING_AGE = 18
EXPECTED_EMPLOYEES_PER_BUSINESS = 90.105203

YEARLY_JOB_CHANGE_RATE = 0.5  # 50 changes per 100 py

UNEMPLOYED_ID = 0
UNEMPLOYED_ADDRESS_ID = 0

BUSINESS_MOVE_RATE_YEARLY = 0.1  # 10 changes per 100 py


class MilitaryEmployer(NamedTuple):
    EMPLOYER_ID = 1
    EMPLOYER_ADDRESS_ID = 1
    EMPLOYER_NAME = "military"
    PROPORTION_WORKFORCE_EMPLOYED = 0.03


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
