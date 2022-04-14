import itertools

import pandas as pd

from vivarium_census_prl_synth_pop.constants import models

#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'

# Columns from parallel runs
INPUT_DRAW_COLUMN = 'input_draw'
RANDOM_SEED_COLUMN = 'random_seed'

OUTPUT_INPUT_DRAW_COLUMN = 'input_data.input_draw_number'
OUTPUT_RANDOM_SEED_COLUMN = 'randomness.random_seed'
OUTPUT_SCENARIO_COLUMN = 'screening_algorithm.scenario'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
}


TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'

COLUMN_TEMPLATES = {
    'population': TOTAL_POPULATION_COLUMN_TEMPLATE,
}

NON_COUNT_TEMPLATES = [
]

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
# TODO - add literals for years in the model
YEARS = ()
# TODO - add literals for ages in the model
AGE_GROUPS = ()
# TODO - add causes of death
CAUSES_OF_DEATH = (
    'other_causes',
    # models.FIRST_STATE_NAME,
)
# TODO - add causes of disability
CAUSES_OF_DISABILITY = (
    # models.FIRST_STATE_NAME,
    # models.SECOND_STATE_NAME,
)

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
}


def RESULT_COLUMNS(kind='all'):
    if kind not in COLUMN_TEMPLATES and kind != 'all':
        raise ValueError(f'Unknown result column type {kind}')
    columns = []
    if kind == 'all':
        for k in COLUMN_TEMPLATES:
            columns += RESULT_COLUMNS(k)
        columns = list(STANDARD_COLUMNS.values()) + columns
    else:
        template = COLUMN_TEMPLATES[kind]
        filtered_field_map = {field: values
                              for field, values in TEMPLATE_FIELD_MAP.items() if f'{{{field}}}' in template}
        fields, value_groups = filtered_field_map.keys(), itertools.product(*filtered_field_map.values())
        for value_group in value_groups:
            columns.append(template.format(**{field: value for field, value in zip(fields, value_group)}))
    return columns


def RESULTS_MAP(kind):
    if kind not in COLUMN_TEMPLATES:
        raise ValueError(f'Unknown result column type {kind}')
    columns = []
    template = COLUMN_TEMPLATES[kind]
    filtered_field_map = {field: values
                          for field, values in TEMPLATE_FIELD_MAP.items() if f'{{{field}}}' in template}
    fields, value_groups = list(filtered_field_map.keys()), list(itertools.product(*filtered_field_map.values()))
    for value_group in value_groups:
        columns.append(template.format(**{field: value for field, value in zip(fields, value_group)}))
    df = pd.DataFrame(value_groups, columns=map(lambda x: x.lower(), fields))
    df['key'] = columns
    df['measure'] = kind  # per researcher feedback, this column is useful, even when it's identical for all rows
    return df.set_index('key').sort_index()

