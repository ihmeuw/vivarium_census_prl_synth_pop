from typing import NamedTuple

####################
# Project metadata #
####################

PROJECT_NAME = 'vivarium_census_prl_synth_pop'
CLUSTER_PROJECT = 'proj_cost_effect'
# # TODO use proj_csu if a csu project
# CLUSTER_PROJECT = 'proj_csu'

CLUSTER_QUEUE = 'all.q'
MAKE_ARTIFACT_MEM = '10G'
MAKE_ARTIFACT_CPU = '1'
MAKE_ARTIFACT_RUNTIME = '3:00:00'
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    # TODO - project locations here
]

ARTIFACT_INDEX_COLUMNS = [
    'sex',
    'age_start',
    'age_end',
    'year_start',
    'year_end',
]

DRAW_COUNT = 1000
ARTIFACT_COLUMNS = pd.Index([f'draw_{i}' for i in range(DRAW_COUNT)])


class __Scenarios(NamedTuple):
    baseline: str = 'baseline'
    # TODO - add scenarios here


SCENARIOS = __Scenarios()
