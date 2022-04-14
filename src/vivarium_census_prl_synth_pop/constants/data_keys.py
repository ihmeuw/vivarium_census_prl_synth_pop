from typing import NamedTuple


#############
# Data Keys #
#############

METADATA_LOCATIONS = 'metadata.locations'


class __Population(NamedTuple):
    HOUSEHOLDS: str = 'population.households'
    PERSONS: str = 'population.persons'

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
