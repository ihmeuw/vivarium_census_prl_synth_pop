"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""
import pandas as pd

from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import gbd
from vivarium_inputs import globals as vi_globals, interface, utilities as vi_utils, utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_census_prl_synth_pop.constants import data_keys, paths, metadata


def get_data(lookup_key: str, location: str) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        data_keys.POPULATION.HOUSEHOLDS: load_households,
        data_keys.POPULATION.PERSONS: load_persons,
    }
    return mapping[lookup_key](lookup_key, location)


def load_households(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.POPULATION.HOUSEHOLDS:
        raise ValueError(f'Unrecognized key {key}')
    # read in data
    data_dir = paths.HOUSEHOLDS_DATA_DIR
    data = pd.concat(
        [
            pd.read_csv(
                data_dir / file,
                usecols=metadata.HOUSEHOLDS_COLUMN_MAP.keys(),
            ) for file in paths.HOUSEHOLDS_FNAMES
        ])
    data.SERIALNO = data.SERIALNO.astype(str)

    # reshape
    data = data.rename(columns=metadata.HOUSEHOLDS_COLUMN_MAP)
    data = data.set_index(['state', 'puma', 'hh_id', 'hh_weight'])

    if location != "United States":
        data = data.query(f'state == {metadata.CENSUS_STATE_IDS[location]}')

    # return data
    return data


def load_persons(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.POPULATION.PERSONS:
        raise ValueError(f'Unrecognized key {key}')
    # read in data
    location = str.replace(location, ' ', '_')
    data_dir = paths.PERSONS_DATA_DIR / location
    data = pd.read_csv(
        data_dir / paths.PERSONS_FNAME,
        usecols=metadata.PERSONS_COLUMNS_MAP.keys()
    )
    data.SERIALNO = data.SERIALNO.astype(str)

    ## map ACS vars to human-readable ##
    data = data.rename(columns=metadata.PERSONS_COLUMNS_MAP)

    # map race and ethnicity to one var
    data["race_eth"] = data.latino.map(metadata.LATINO_VAR_MAP)
    data.loc[data.race_eth == 1, 'race_eth'] = data.loc[data.race_eth == 1].race

    # label each race/eth
    data.race_eth = data.race_eth.map(metadata.RACE_ETH_VAR_MAP)
    data = data.drop(columns=['latino', 'race'])

    # map sexes
    data.sex = data.sex.map(metadata.SEX_VAR_MAP)

    # map relationship to hh head
    data.relation_to_hh_head = data.relation_to_hh_head.map(metadata.RELSHIP_TO_HH_HEAD_MAP)

    # create person id
    data['person_id'] = range(data.shape[0])

    # reshape
    data = data.set_index(['hh_id', 'person_id', 'age', 'relation_to_hh_head', 'sex', 'race_eth'])

    # return data
    return data


