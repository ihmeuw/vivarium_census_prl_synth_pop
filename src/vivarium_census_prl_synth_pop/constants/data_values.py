from typing import NamedTuple

from scipy import stats

#########################
# Population parameters #
#########################

# TODO: implement gbd call (vivarium_inputs.get_population_structure("United States"))
US_POPULATION = 333339776

MAX_HOUSEHOLD_SIZE = 17
PROP_POPULATION_IN_GQ = 0.03
PROBABILITY_OF_TWINS = 0.04
N_GROUP_QUARTER_TYPES = 6
INSTITUTIONAL_GROUP_QUARTER_IDS = {"Carceral": 0, "Nursing home": 1, "Other institutional": 2}
NONINSTITUTIONAL_GROUP_QUARTER_IDS = {
    "College": 3,
    "Military": 4,
    "Other non-institutional": 5,
}
GROUP_QUARTER_IDS = {
    "Institutionalized GQ pop": INSTITUTIONAL_GROUP_QUARTER_IDS,
    "Noninstitutionalized GQ pop": NONINSTITUTIONAL_GROUP_QUARTER_IDS,
}
GQ_HOUSING_TYPE_MAP = {
    0: "Carceral",
    1: "Nursing home",
    2: "Other institutional",
    3: "College",
    4: "Military",
    5: "Other non-institutional",
}
HOUSING_TYPES = ["Standard"] + list(GQ_HOUSING_TYPE_MAP.values())

PROPORTION_INITIALIZATION_WITH_SSN = 0.743

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

BUSINESS_NAMES_MAX_TOKENS_LENGTH = 15
PROPORTION_WORKFORCE_EMPLOYED = 0.6114

WORKING_AGE = 18
EXPECTED_EMPLOYEES_PER_BUSINESS = 90.105203

YEARLY_JOB_CHANGE_RATE = 0.5  # 50 changes per 100 py

BUSINESS_MOVE_RATE_YEARLY = 0.1  # 10 changes per 100 py

PERSONAL_INCOME_PROPENSITY_DISTRIBUTION = stats.norm(loc=0.0, scale=0.812309**0.5)
EMPLOYER_INCOME_PROPENSITY_DISTRIBUTION = stats.norm(loc=0.0, scale=0.187691**0.5)


class MilitaryEmployer(NamedTuple):
    EMPLOYER_ID = 1
    EMPLOYER_ADDRESS_ID = 1
    EMPLOYER_NAME = "military"
    PROPORTION_WORKFORCE_EMPLOYED = 0.0032


class Unemployed(NamedTuple):
    EMPLOYER_ID = 0
    EMPLOYER_ADDRESS_ID = 0
    EMPLOYER_NAME = "unemployed"


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
