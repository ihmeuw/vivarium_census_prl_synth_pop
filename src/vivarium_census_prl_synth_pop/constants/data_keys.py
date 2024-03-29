from typing import NamedTuple

#############
# Data Keys #
#############

METADATA_LOCATIONS = "metadata.locations"


class __Population(NamedTuple):
    HOUSEHOLDS: str = "population.households"
    PERSONS: str = "population.persons"
    ACMR: str = "cause.all_causes.cause_specific_mortality_rate"
    TMRLE: str = "population.theoretical_minimum_risk_life_expectancy"
    LOCATION: str = "population.location"
    ASFR: str = "covariate.age_specific_fertility_rate.estimate"

    @property
    def name(self):
        return "population"

    @property
    def log_name(self):
        return "population"


POPULATION = __Population()


class __SyntheticData(NamedTuple):
    FIRST_NAMES: str = "synthetic_data.first_names"
    LAST_NAMES: str = "synthetic_data.last_names"
    ADDRESSES: str = "synthetic_data.addresses"
    BUSINESS_NAMES: str = "synthetic_data.business_names"
    SSNS: str = "synthetic_data.ssns"
    ITINS: str = "synthetic_data.itins"

    @property
    def name(self):
        return "synthetic_data"

    @property
    def log_name(self):
        return "synthetic_data"


SYNTHETIC_DATA = __SyntheticData()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    SYNTHETIC_DATA,
]


class __MetadataColumns(NamedTuple):
    DATASET = "dataset"
    STATE = "state"
    YEAR = "year"
    NUMBER_OF_ROWS = "number_of_rows"
    COLUMN = "column"
    NOISE_TYPE = "noise_type"
    PROPORTION = "proportion"
    GROUP_ROW_COUNTS = "group_row_counts"


METADATA_COLUMNS = __MetadataColumns()
