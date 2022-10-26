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
from typing import Dict
from loguru import logger

from collections import defaultdict
import numpy as np
import pandas as pd
from vivarium.framework.artifact import EntityKey
from vivarium_inputs import interface

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import (
    data_keys,
    data_values,
    metadata,
    paths,
)
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
        data_keys.SYNTHETIC_DATA.BUSINESS_NAMES: generate_business_names_data,
    }
    return mapping[lookup_key](lookup_key, location)


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    data = interface.get_measure(entity, key.measure, location).droplevel("location")
    return data


# noinspection PyUnusedLocal
def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_raw_persons_data(column_map: Dict[str, str], location):
    data_dir = paths.PERSONS_DATA_DIR
    data = pd.concat(
        [
            pd.read_csv(data_dir / file, usecols=column_map.keys())
            for file in paths.PERSONS_FILENAMES
        ]
    )
    data.SERIALNO = data.SERIALNO.astype(str)

    # map ACS vars to human-readable
    data = data.rename(columns=column_map)

    if location != "United States":
        data = data.query(f"state == {metadata.CENSUS_STATE_IDS[location]}")
    data = data.drop(columns=["state"])

    return data


def load_persons(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.POPULATION.PERSONS:
        raise ValueError(f"Unrecognized key {key}")
    # read in data
    location = str.replace(location, " ", "_")
    data = load_raw_persons_data(metadata.PERSONS_COLUMNS_MAP, location)

    # map race and ethnicity to one var
    data["race_ethnicity"] = data.latino.map(metadata.LATINO_VAR_MAP)
    data.loc[data.race_ethnicity == 1, "race_ethnicity"] = data.loc[
        data.race_ethnicity == 1
    ].race

    # label each race/eth
    data.race_ethnicity = data.race_ethnicity.map(metadata.RACE_ETHNICITY_VAR_MAP)
    data = data.drop(columns=["latino", "race"])

    # map sexes
    data.sex = data.sex.map(metadata.SEX_VAR_MAP)

    # map relationship to household head
    data.relation_to_household_head = data.relation_to_household_head.map(
        metadata.RELATIONSHIP_TO_HOUSEHOLD_HEAD_MAP
    )

    # put all non-draw columns in the index, else vivarium will drop them
    data = data.set_index(
        ["census_household_id", "age", "relation_to_household_head", "sex", "race_ethnicity"]
    )

    return data


def load_households(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.POPULATION.HOUSEHOLDS:
        raise ValueError(f"Unrecognized key {key}")
    # read in data
    data_dir = paths.HOUSEHOLDS_DATA_DIR
    data = pd.concat(
        [
            pd.read_csv(data_dir / file, usecols=metadata.HOUSEHOLDS_COLUMN_MAP.keys())
            for file in paths.HOUSEHOLDS_FILENAMES
        ]
    )
    data.SERIALNO = data.SERIALNO.astype(str)

    # reshape
    data = data.rename(columns=metadata.HOUSEHOLDS_COLUMN_MAP)

    if location != "United States":
        data = data.query(f"state == {metadata.CENSUS_STATE_IDS[location]}")

    # read in persons file to find which household_ids it contains
    persons = load_raw_persons_data(metadata.SUBSET_PERSONS_COLUMNS_MAP, location)

    # subset data to household ids in person file
    data = data.query(
        f"census_household_id in {list(persons['census_household_id'].unique())}"
    )

    # merge on person weights for GQ
    data = data.merge(
        persons.loc[["GQ" in i for i in persons.census_household_id]],
        on="census_household_id",
        how="left",
    )

    data = data.set_index(
        ["state", "puma", "census_household_id", "household_weight", "person_weight"]
    )

    return data


def load_location(key: str, location: str) -> str:
    return location


def load_asfr(key: str, location: str):
    asfr = load_standard_data(key, location)

    # pivot
    asfr = asfr.reset_index()
    asfr = asfr[(asfr.year_start == 2019)]  # NOTE: this is the latest year available from GBD
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
    STATE_CODE = metadata.US_STATE_ABBRV_MAP[location]
    data_path = paths.SYNTHETIC_DATA_INPUTS_ROOT / "ssn_names" / f"{STATE_CODE}.TXT"
    df_ssn_names = pd.read_csv(data_path, names=["state", "sex", "yob", "name", "freq"])
    df_ssn_names["sex"] = df_ssn_names["sex"].map({"M": "Male", "F": "Female"})

    # put all non-draw columns in the index, else vivarium will drop them
    df_ssn_names = df_ssn_names.set_index(["state", "sex", "yob", "name", "freq"])
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


def generate_business_names_data(key: str, location: str) -> pd.Series:
    # loads csv of business names and generates pandas series with random business names

    business_names = pd.read_csv(paths.BUSINESS_NAMES_DATA)
    bigrams = make_bigrams(business_names)  # bigrams is a pd.Series with multi-index with first_word, second_word

    # Get frequency of business names and find uncommon ones
    s_name_freq = business_names.location_name.value_counts()
    real_but_uncommon_names = set(s_name_freq[s_name_freq < 1_000].index)

    n_total_names = 100_000  # How do we want to choose this and declare it as a constant?

    # Generate random business names.  Drop duplicates and overlapping names with uncommon names
    new_names = pd.Series()

    # Generate additional names until desired number of random business names is met
    while len(new_names) < n_total_names:
        n_needed = n_total_names - len(new_names)
        more_names = sample_names(bigrams, n_needed, data_values.BUSINESS_NAMES_MAX_TOKENS_LENGTH)
        new_names = pd.concat([new_names, more_names]).drop_duplicates()
        new_names = new_names.loc[~new_names.isin(real_but_uncommon_names)]

    new_names.to_hdf(
        paths.BUSINESS_NAMES_DATA_ARTIFACT_INPUT_PATH,
        key="business_names",
        mode="w"
    )


def make_bigrams(df: pd.DataFrame):
    # Makes default dict of business names for map to sample from
    # bigrams will be a Dict[str: Dict[str: int]]
    # Example {"<start>": {keys are all first words for businesses: values are frequence where these pairs happen}}

    def dict_factory():
        return lambda: defaultdict(int)

    bigrams = defaultdict(dict_factory())
    n_rows = len(df)  # expect a few minutes

    for i in range(n_rows):
        if i % 10_000 == 0:
            print('.', end=' ', flush=True)
        names_i = df.iloc[i, 0]

        tokens_i = names_i.split(' ')
        for j in range(len(tokens_i)):
            if j == 0:
                bigrams['<start>'][tokens_i[j]] += 1
            else:
                bigrams[tokens_i[j - 1]][tokens_i[j]] += 1
        bigrams[tokens_i[j]]['<end>'] += 1

    return bigrams


def sample_names(bigrams: defaultdict, n_businesses: int,  n_max_tokens: int) -> pd.Series:
    """

    Parameters
    ----------
    bigrams: Default dict produced from make_bigrams function (see formatting of default dict.
    n_businesses: Int of how many business names to generate
    n_max_tokens: Int of max number of words possible for a business name to contain

    Returns
    -------
    A string that is a randomly generated business name
    """

    columns = [f'word_{i}' for i in range(n_max_tokens)]
    names = pd.DataFrame(columns=columns)
    names['word_0'] = ["<start>"] * n_businesses

    for i in range(1, n_max_tokens):
        # todo: change size based on number of names being created (n_businesses)
        logger.info(
            f"Sampling random business_names.  Creating {i}th of {n_max_tokens} words (column) in business names."
        )
        previous_word = f'word_{i - 1}'
        next_word = f'word_{i}'
        current_words_count_dict = names[previous_word].value_counts().to_dict()
        for word in current_words_count_dict.keys():
            if word != '<end>':
                vals = list(bigrams[word].keys())
                pr = np.array(list(bigrams[word].values()))
                tokens = np.random.choice(vals, p=pr / pr.sum(), size=current_words_count_dict[word])

                names.loc[names[previous_word] == word, next_word] = tokens

    # Process generated names by combining all columns and dropping outer tokens of <start> and <end>
    names = names.replace(np.nan, "", regex=True)
    names["business_names"] = names[columns[1:]].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    names["business_names"] = names["business_names"].str.split(" <").str[0]

    return names["business_names"]
