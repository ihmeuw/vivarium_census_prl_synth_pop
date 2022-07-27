########################
# Migration Parameters #
########################

HOUSEHOLD_MOVE_RATE_YEARLY = 0.15
INDIVIDUAL_MOVE_RATE_YEARLY = 0.15
PROBABILITY_OF_TWINS = 0.04

#########################
# Synthetic Name Inputs #
#########################

# calculated by Abie from North Carolina voter registration data
PROBABILITY_OF_HYPHEN_IN_NAME = {
    'AIAN': 0.00685,
    'Asian': 0.00682,
    'Black': 0.01326,
    'Latino': 0.06842,
    'Multiracial or Other': 0.01791,
    'NHOPI': 0.02064,
    'White': 0.00474
}

# calculated by Abie from North Carolina voter registration data
PROBABILITY_OF_SPACE_IN_NAME = {
    'AIAN': 0.00408,
    'Asian': 0.0086,
    'Black': 0.0041,
    'Latino': 0.12807,
    'Multiracial or Other': 0.02004,
    'NHOPI': 0.02064,
    'White': 0.00347
}

# data from https://www.jec.senate.gov/public/index.cfm/republicans/fl/
PROPORTION_WORKFORCE_EMPLOYED = {
    'Florida': .576
}

EXPECTED_EMPLOYEES_PER_BUSINESS = 90.105203

JOB_CHANGE_RATE = .5 # 50 changes per 100 py?
