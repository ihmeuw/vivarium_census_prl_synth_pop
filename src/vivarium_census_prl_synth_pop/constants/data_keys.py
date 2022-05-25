from typing import NamedTuple


#############
# Data Keys #
#############

METADATA_LOCATIONS = 'metadata.locations'


class __Population(NamedTuple):
    HOUSEHOLDS: str = 'population.households'
    PERSONS: str = 'population.persons'
    ACMR: str = 'cause.all_causes.cause_specific_mortality_rate'

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
