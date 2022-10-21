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


class __SyntheticData(NamedTuple):
    FIRST_NAMES: str = "synthetic_data.first_names"
    LAST_NAMES: str = "synthetic_data.last_names"
    ADDRESSES: str = "synthetic_data.addresses"
    BUSINESS_NAMES: str = "synthetic_data.business_names"

    @property
    def name(self):
        return "synthetic_data"

    @property
    def log_name(self):
        return "synthetic_data"


POPULATION = __Population()
SYNTHETIC_DATA = __SyntheticData()


MAKE_ARTIFACT_KEY_GROUPS = [POPULATION, SYNTHETIC_DATA]
