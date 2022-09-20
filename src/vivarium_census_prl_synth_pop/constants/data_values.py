from typing import NamedTuple


#########################
# Population parameters #
#########################

PROP_POPULATION_IN_GQ = 0.03
PROBABILITY_OF_TWINS = 0.04
N_GROUP_QUARTER_TYPES = 6
INSTITUTIONAL_GROUP_QUARTER_IDS = {
    "Carceral": 0,
    "Nursing home": 1,
    "Other institutional": 2
}
NONINSTITUTIONAL_GROUP_QUARTER_IDS = {
    "College": 3,
    "Military": 4,
    "Other non-institutional": 5,
}

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
PROPORTION_WORKFORCE_EMPLOYED = {"Florida": 0.576}
PROPORTION_INITIALIZATION_NO_SSN = 0.14
PROPORTION_NEWBORNS_NO_SSN = 0.10
PROPORTION_LEAVING_COUNTRY = 0.01
PROPORTION_GQ_PERSONS_LEAVING_COUNTRY = 0.03

WORKING_AGE = 18
EXPECTED_EMPLOYEES_PER_BUSINESS = 90.105203

YEARLY_JOB_CHANGE_RATE = 0.5  # 50 changes per 100 py

UNEMPLOYED_ID = -1

BUSINESS_MOVE_RATE_YEARLY = 0.1  # 10 changes per 100 py


class MilitaryEmployer(NamedTuple):
    EMPLOYER_ID = -3
    EMPLOYER_ADDRESS = "military address"
    EMPLOYER_ZIPCODE = "military zipcode"
    EMPLOYER_NAME = "military"
    PROPORTION_WORKFORCE_EMPLOYED = 0.03
