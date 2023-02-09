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
from pathlib import Path
from typing import Dict, List

import pandas as pd
from vivarium.framework.artifact import EntityKey
from vivarium_inputs import interface

from vivarium_census_prl_synth_pop.constants import data_keys, metadata, paths
from vivarium_census_prl_synth_pop.data.utilities import get_entity
from vivarium_census_prl_synth_pop.utilities import (
    get_norm_from_quantiles,
    get_random_variable_draws_for_location,
)


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
        data_keys.POPULATION.ACMR: load_standard_data,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.LOCATION: load_location,
        data_keys.POPULATION.ASFR: load_asfr,
        data_keys.SYNTHETIC_DATA.LAST_NAMES: load_last_name_data,
        data_keys.SYNTHETIC_DATA.FIRST_NAMES: load_first_name_data,
        data_keys.SYNTHETIC_DATA.ADDRESSES: load_address_data,
    }
    return mapping[lookup_key](lookup_key, location)


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    data = interface.get_measure(entity, key.measure, location).droplevel("location")
    return data


# noinspection PyUnusedLocal
def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_raw_persons_data(column_map: Dict[str, str], location: str) -> pd.DataFrame:
    data = _read_and_format_raw_data(
        data_dir=paths.PERSONS_DATA_DIR,
        filenames=paths.PERSONS_FILENAMES,
        column_map=column_map,
        location=location,
    )
    return data.drop(columns=["state"])


def load_persons(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.POPULATION.PERSONS:
        raise ValueError(f"Unrecognized key {key}")
    # read in data
    data = load_raw_persons_data(metadata.PERSONS_COLUMNS_MAP, location)

    # map race and ethnicity to one var
    data["race_ethnicity"] = data["latino"].map(metadata.LATINO_VAR_MAP)
    data.loc[data["race_ethnicity"] == 1, "race_ethnicity"] = data.loc[
        data["race_ethnicity"] == 1, "race"
    ]

    # label each race/eth
    data["race_ethnicity"] = (
        data["race_ethnicity"]
        .map(metadata.RACE_ETHNICITY_VAR_MAP)
        .astype(pd.CategoricalDtype(categories=metadata.RACE_ETHNICITIES))
    )

    data = data.drop(columns=["latino", "race"])

    # map sexes
    data["sex"] = (
        data["sex"]
        .map(metadata.SEX_VAR_MAP)
        .astype(pd.CategoricalDtype(categories=metadata.SEXES))
    )

    # map relationship to household head
    data["relation_to_household_head"] = (
        data["relation_to_household_head"]
        .map(metadata.RELATIONSHIP_TO_HOUSEHOLD_HEAD_MAP)
        .astype(pd.CategoricalDtype(categories=metadata.RELATIONSHIPS))
    )
    # Map native born persons and if person has migrated in last year
    data["born_in_us"] = data["born_in_us"].map(
        metadata.NATIVITY_MAP
    )  # True for native born persons
    data["immigrated_in_last_year"] = data["immigrated_in_last_year"].map(
        metadata.MIGRATION_MAP
    )
    data.loc[
        data["immigrated_in_last_year"].isnull(), "immigrated_in_last_year"
    ] = False  # Make Nulls map to False

    # put all non-draw columns in the index, else vivarium will drop them
    data = data.set_index(list(data.columns))
    return data


def load_households(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.POPULATION.HOUSEHOLDS:
        raise ValueError(f"Unrecognized key {key}")

    data = _read_and_format_raw_data(
        data_dir=paths.HOUSEHOLDS_DATA_DIR,
        filenames=paths.HOUSEHOLDS_FILENAMES,
        column_map=metadata.HOUSEHOLDS_COLUMN_MAP,
        location=location,
    )

    # map household_type
    data["household_type"] = (
        data["household_type"]
        .map(metadata.HOUSEHOLD_TYPE_MAP)
        .astype(pd.CategoricalDtype(categories=metadata.HOUSEHOLD_TYPES))
    )

    # read in persons file to find which household_ids it contains
    persons = load_raw_persons_data(metadata.SUBSET_PERSONS_COLUMNS_MAP, location)

    # subset data to household ids in person file
    data = data[data["census_household_id"].isin(persons["census_household_id"])]

    # merge in person weights for GQ
    # FIXME -- this is no longer necessary, since person_weights in the household file
    # are not used by the simulation anymore.
    gq_households = data[data["household_type"] != "Housing unit"]
    gq_persons = persons[
        persons["census_household_id"].isin(gq_households["census_household_id"])
    ]
    data = data.merge(
        gq_persons[["census_household_id", "person_weight"]],
        on="census_household_id",
        how="left",
    )

    # put all non-draw columns in the index, else vivarium will drop them
    data = data.set_index(list(data.columns))

    return data


def load_location(key: str, location: str) -> str:
    return location


def load_asfr(key: str, location: str):
    asfr = load_standard_data(key, location)

    # pivot
    asfr = asfr.reset_index()
    # NOTE: 2019 is the latest year available from GBD
    asfr = asfr[(asfr["year_start"] == 2019)]
    asfr_pivot = asfr.pivot(
        index=[col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"],
        columns="parameter",
        values="value",
    )
    asfr_draws = asfr_pivot.apply(create_draws, args=(key, location), axis=1)

    return asfr_draws


def _capitalize_names(name):
    if type(name) == str:
        name = name.lower()
        for connector in [" ", "-"]:
            name = connector.join([i[0].upper() + i[1:] for i in name.split(connector)])
        return name
    else:  # missing names
        return name


def load_last_name_data(key: str, location: str) -> pd.DataFrame:
    df_census_names = pd.read_csv(paths.LAST_NAME_DATA_PATH, na_values=["(S)"])

    df_census_names["name"] = df_census_names["name"].apply(_capitalize_names)

    ## fill missing values with equal amounts of what is left ##
    # per row, count N pct cols that are null
    n_missing = df_census_names.filter(like="pct").isnull().sum(axis=1)

    # per now, sum total pcts that are non-null
    pct_total = df_census_names.filter(like="pct").sum(axis=1)

    # calculate how much each pct to give to each null col
    pct_fill = (100 - pct_total) / n_missing

    for col in df_census_names.filter(like="pct").columns:
        df_census_names[col] = df_census_names[col].fillna(pct_fill)

    # drop non-name
    df_census_names = df_census_names.loc[df_census_names.name != "All Other Names"]

    all_race_name_count = df_census_names["count"].copy()
    for race_eth, pct_specific_race in [
        ["White", "pctwhite"],
        ["Latino", "pcthispanic"],
        ["Black", "pctblack"],
        ["Asian", "pctapi"],
        ["Multiracial or Other", "pct2prace"],
        ["AIAN", "pctaian"],
        ["NHOPI", "pctapi"],
    ]:
        race_specific_name_count = (
            all_race_name_count * df_census_names[pct_specific_race] / 100
        )
        race_specific_name_pct = race_specific_name_count / race_specific_name_count.sum()
        df_census_names[race_eth] = race_specific_name_pct

    # put all non-draw columns in the index, else vivarium will drop them
    df_census_names = df_census_names.set_index(
        [
            "name",
            "rank",
            "count",
            "prop100k",
            "cum_prop100k",
            "pctwhite",
            "pctblack",
            "pctapi",
            "pctaian",
            "pct2prace",
            "pcthispanic",
            "White",
            "Latino",
            "Black",
            "Asian",
            "Multiracial or Other",
            "AIAN",
            "NHOPI",
        ]
    )
    return df_census_names


def load_first_name_data(key: str, location: str) -> pd.DataFrame:
    if location == "United States of America":

        def read_name_csv(csv_path: Path) -> pd.DataFrame:
            return pd.read_csv(csv_path, names=["state", "sex", "yob", "name", "freq"])

        ssn_names = [
            read_name_csv(
                paths.SYNTHETIC_DATA_INPUTS_ROOT / "ssn_names" / f"{state_code}.TXT"
            )
            for state_code in metadata.US_STATE_ABBRV_MAP.values()
        ]
        df_ssn_names = pd.concat(ssn_names)
        df_ssn_names = (
            df_ssn_names.groupby(["sex", "yob", "name"])["freq"].sum().reset_index()
        )
    else:
        state_code = metadata.US_STATE_ABBRV_MAP[location]
        data_path = paths.SYNTHETIC_DATA_INPUTS_ROOT / "ssn_names" / f"{state_code}.TXT"
        df_ssn_names = pd.read_csv(data_path, names=["sex", "yob", "name", "freq"])

    df_ssn_names["sex"] = df_ssn_names["sex"].map({"M": "Male", "F": "Female"})

    # put all non-draw columns in the index, else vivarium will drop them
    df_ssn_names = df_ssn_names.set_index(["sex", "yob", "name", "freq"])
    return df_ssn_names


def load_address_data(key: str, location: str) -> pd.DataFrame:
    df_deepparse_address_data = pd.read_csv(paths.ADDRESS_DATA_PATH)
    df_deepparse_address_data = df_deepparse_address_data.drop(columns="Unnamed: 0")

    # put all non-draw columns in the index, else vivarium will drop them
    df_deepparse_address_data = df_deepparse_address_data.set_index(
        ["StreetNumber", "StreetName", "Municipality", "Province", "PostalCode", "Unit"]
    )
    return df_deepparse_address_data


def create_draws(df: pd.DataFrame, key: str, location: str):
    """
    Parameters
    ----------
    df: Multi-index dataframe with mean, lower, and upper values columns.
    location
    key:
    Returns
    -------
    """
    # location defined in namespace outside of function
    mean = df["mean_value"]
    lower = df["lower_value"]
    upper = df["upper_value"]

    distribution = get_norm_from_quantiles(mean=mean, lower=lower, upper=upper)
    # pull index from constants
    draws = get_random_variable_draws_for_location(
        pd.Index([f"draw_{i}" for i in range(0, 1000)]), location, key, distribution
    )

    return draws


#####################
#  Helper functions #
#####################


def _read_and_format_raw_data(
    data_dir: Path, filenames: List[str], column_map: List[str], location: str
) -> pd.DataFrame:
    data = pd.concat(
        [
            pd.read_csv(data_dir / file, usecols=column_map.keys(), dtype={"SERIALNO": str})
            for file in filenames
        ]
    )

    data = data.rename(columns=column_map)

    if location == "United States of America":
        if metadata.UNITED_STATES_LOCATIONS:
            location_ids = [
                v
                for k, v in metadata.CENSUS_STATE_IDS.items()
                if k in metadata.UNITED_STATES_LOCATIONS
            ]
            data = data.query(f"state in {location_ids}")
    elif location in metadata.CENSUS_STATE_IDS:
        data = data.query(f"state == {metadata.CENSUS_STATE_IDS[location]}")
    else:
        raise RuntimeError(f"location {location} not found in metadata.CENSUS_STATE_IDS")
    return data
