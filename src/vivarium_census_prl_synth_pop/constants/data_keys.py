from typing import NamedTuple


#############
# Data Keys #
#############

METADATA_LOCATIONS = 'metadata.locations'


class __Population(NamedTuple):
    PERSONS: str = 'population.persons'
    HOUSEHOLDS: str = 'population.households'
    ACMR: str = 'cause.all_causes.cause_specific_mortality_rate'
    TMRLE: str = 'population.theoretical_minimum_risk_life_expectancy'

    @property
    def name(self):
        return 'population'

    @property
    def log_name(self):
        return 'population'


POPULATION = __Population()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
]
