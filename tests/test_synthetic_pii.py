# Sample Test passing with nose and pytest
import string
from types import MethodType
from typing import List

import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.components import synthetic_pii
from vivarium_census_prl_synth_pop.results_processing.addresses import (
    get_household_address_map,
    get_zipcode_map,
)
from vivarium_census_prl_synth_pop.results_processing.names import (
    get_given_name_map,
    get_last_name_map,
)

key = "test_synthetic_data_generation"
clock = lambda: pd.Timestamp("2020-09-01")
seed = 0
randomness = RandomnessStream(key=key, clock=clock, seed=seed)


def get_draw(self, index, additional_key=None) -> pd.Series:
    # Mock get draw function
    s = np.random.uniform(size=len(index))
    return pd.Series(data=s, index=index)


def test_ssn(mocker):
    g = synthetic_pii.SSNGenerator()

    index = range(10)
    df_in = pd.DataFrame(index=index)

    # Patch randomness stream to a mock object, point randomness.get_draw to a dummy function
    g.randomness = mocker.Mock()
    mocker.patch.object(g.randomness, "get_draw")
    g.randomness.get_draw = MethodType(get_draw, g)
    df = g.generate(df_in)

    assert len(df) == 10, "expect result to be a dataframe with 10 rows"

    ssn1 = df.loc[0, "ssn"]
    area, group, serial = ssn1.split("-")
    assert len(area) == 3
    assert len(group) == 2
    assert len(serial) == 4
    # todo: Add test of uniqueness


@pytest.fixture()
def given_names():
    names = pd.DataFrame(
        data={
            "sex": ["Female", "Female", "Male", "Male"] * 2,
            "yob": [1999, 1999, 1999, 1999, 2000, 2000, 2000, 2000],
            "name": ["Mary", "Annie", "Luis", "John", "Scarlett", "Anna", "Mark", "Brad"],
            "freq": [200, 100, 150, 50, 100, 25, 100, 150],
        }
    )
    names = names.set_index(["yob", "sex", "name", "freq"])
    return names


@pytest.fixture()
def last_names():
    last_names = pd.DataFrame(
        data={
            "name": ["Smith", "Johnson", "Jackson"],
            "rank": [1, 2, 3],
            "count": [2000, 1500, 1000],
            "prop100k": [20.01, 15.01, 10.01],
            "cum_prop100k": [20.01, 15.01, 10.01],
            "pctwhite": [70.01, 60.01, 50.01],
            "pctblack": [22.01, 21.01, 20.01],
            "pctapi": [0.55, 0.54, 0.53],
            "pctasian": [0.62, 0.61, 0.60],
            "pct2prace": [3.03, 3.02, 3.01],
            "pcthispanic": [2.02, 2.01, 2.00],
            "White": [0.34, 0.26, 0.40],
            "Latino": [0.10, 0.35, 0.55],
            "Black": [0.20, 0.40, 0.40],
            "Asian": [0.34, 0.33, 0.33],
            "Multiracial or Other": [0.11, 0.12, 0.13],
            "AIAN": [0.01, 0.02, 0.03],
            "NHOPI": [0.05, 0.04, 0.03],
        }
    )
    cols = list(last_names.columns)
    last_names = last_names.set_index(cols)

    return last_names


@pytest.fixture()
def fake_obs_data():
    size = 100_000
    return {
        "fake_observer": pd.DataFrame(
            {
                "first_name_id": list(range(size)),
                "middle_name_id": list(range(size)),
                "year_of_birth": [1999, 1999, 2000, 2000] * 25_000,
                "sex": ["Male", "Female"] * 50_000,
                "last_name_id": list(range(size)),
                "race_ethnicity": ["White", "Black", "Latino", "Asian"] * 25_000,
                "date_of_birth": [pd.Timestamp("1985-01-01 00:00:00")] * size,
            }
        )
    }


def get_name_frequency_proportions(
    names: pd.Series, population_demographics: pd.DataFrame, groupby_cols: List[str]
) -> pd.Series:
    name_totals = pd.concat([population_demographics, names], axis=1).rename(
        columns={0: "name"}
    )
    names_grouped = name_totals.groupby(groupby_cols)["name"].value_counts()
    grouped_totals = name_totals[groupby_cols].value_counts()
    name_proportions = names_grouped / grouped_totals

    return name_proportions


def test_first_and_middle_names(mocker, given_names, fake_obs_data):
    artifact = mocker.MagicMock()
    artifact.load.return_value = given_names

    # Get proportion of given names for tests
    totals = given_names.reset_index().groupby(["yob", "sex"])["freq"].sum()
    name_freq = given_names.reset_index("freq")
    proportions = name_freq["freq"] / totals
    proportions.index = proportions.index.rename("year_of_birth", "yob")

    # Get name frequencies for comparison
    first_names = get_given_name_map(
        "first_name_id",
        fake_obs_data,
        artifact,
        randomness,
    )
    first_name_proportions = get_name_frequency_proportions(
        first_names["first_name"],
        fake_obs_data["fake_observer"],
        ["year_of_birth", "sex"],
    )

    assert np.isclose(
        first_name_proportions.sort_index(), proportions.sort_index(), atol=1e-02
    ).all()

    middle_names = get_given_name_map(
        "middle_name_id",
        fake_obs_data,
        artifact,
        randomness,
    )
    middle_name_proportions = get_name_frequency_proportions(
        middle_names["middle_name"], fake_obs_data["fake_observer"], ["year_of_birth", "sex"]
    )

    assert np.isclose(
        middle_name_proportions.sort_index(), proportions.sort_index(), atol=1e-02
    ).all()
    assert (first_names["first_name"] != middle_names["middle_name"]).any()


def test_last_names_proportions(mocker, last_names, fake_obs_data):
    # This function tests that the sampling proportions are working as expected
    artifact = mocker.MagicMock()
    artifact.load.return_value = last_names
    # Subset last names data to match fake_obs_data
    last_names = last_names.reset_index()[["name", "White", "Latino", "Black", "Asian"]]

    # Get proportion of given names for tests
    proportions = (
        pd.melt(
            last_names,
            id_vars="name",
            value_vars=["White", "Latino", "Black", "Asian"],
            var_name="race_ethnicity",
            value_name="freq",
        )
        .groupby(["race_ethnicity", "name"])["freq"]
        .sum()
    )

    # Get name frequencies for comparison
    last_names_map = get_last_name_map(
        "last_name_id",
        fake_obs_data,
        artifact,
        randomness,
    )
    last_name_proportions = get_name_frequency_proportions(
        last_names_map["last_name"],
        fake_obs_data["fake_observer"],
        ["race_ethnicity"],
    )

    assert np.isclose(
        last_name_proportions.sort_index(), proportions.sort_index(), atol=1e-02
    ).all()
    assert not (last_names_map["last_name"].isnull().any())


def test_last_name_from_oldest_member(mocker):
    # This tests logic that we are sampling last_name_id for oldest member with that id.
    # Each household has an oldest member of a difference race ethnicity.
    household_1 = pd.DataFrame(
        {
            "race_ethnicity": ["Asian", "Black", "Latino"],
            "date_of_birth": [
                pd.Timestamp("1965-01-01 00:00:00"),
                pd.Timestamp("1975-01-01 00:00:00"),
                pd.Timestamp("1985-01-01 00:00:00"),
            ],
            "last_name_id": [1, 1, 1],
        }
    )
    household_2 = pd.DataFrame(
        {
            "race_ethnicity": ["Asian", "Black", "Latino"],
            "date_of_birth": [
                pd.Timestamp("1975-01-01 00:00:00"),
                pd.Timestamp("1965-01-01 00:00:00"),
                pd.Timestamp("1985-01-01 00:00:00"),
            ],
            "last_name_id": [2, 2, 2],
        }
    )
    household_3 = pd.DataFrame(
        {
            "race_ethnicity": ["Asian", "Black", "Latino"],
            "date_of_birth": [
                pd.Timestamp("1985-01-01 00:00:00"),
                pd.Timestamp("1975-01-01 00:00:00"),
                pd.Timestamp("1965-01-01 00:00:00"),
            ],
            "last_name_id": [3, 3, 3],
        }
    )
    households = pd.concat([household_1, household_2, household_3])
    fake_obs_data = {"last_name_faker": households}

    # Make fake artifact data
    last_names = pd.DataFrame(
        {
            "name": ["Name A", "Name B", "Name C"],
            "Asian": [1.0, 0.0, 0.0],
            "Black": [0.0, 1.0, 0.0],
            "Latino": [0.0, 0.0, 1.0],
        }
    )
    cols = list(last_names.columns)
    last_names = last_names.set_index(cols)

    # Mock artifact
    artifact = mocker.MagicMock()
    artifact.load.return_value = last_names

    # Map last names
    last_names_map = get_last_name_map(
        "last_name_id",
        fake_obs_data,
        artifact,
        randomness,
    )
    expected = pd.Series(data=["Name A", "Name B", "Name C"], index=[1, 2, 3])

    assert (last_names_map["last_name"] == expected).all()


def test_address(mocker):
    # Fake synthetic pii address data
    synthetic_address_data = pd.DataFrame(
        {
            "StreetNumber": list(range(26)),
            "StreetName": list(string.ascii_lowercase),
            "Unit": list(range(26)),
        }
    )
    synthetic_address_data.set_index(["StreetNumber", "StreetName", "Unit"], inplace=True)
    # Mock artifact
    artifact = mocker.MagicMock()
    artifact.load.return_value = synthetic_address_data

    # Address_ids will just be an series of ids so we just need a unique series in a one column dataframe
    address_ids = pd.DataFrame()
    address_ids["address_id"] = list(range(10))
    fake_obs_data = address_ids

    address_map = get_household_address_map(
        "address_id",
        fake_obs_data,
        artifact,
        randomness,
    )
    expected_keys = ["street_number", "street_name", "unit_number"]

    assert all(street_key in expected_keys for street_key in address_map.keys())
    for street_key, series in address_map.items():
        assert (address_map[street_key].index == address_ids["address_id"]).all()
        assert len(address_map[street_key].index.unique()) == len(
            address_map[street_key].index
        )
        assert not (address_map["street_name"].isnull().any())


def test_zipcode_mapping():
    """Tests ZIP code mapping logic.

    Specifically:
    - all address_ids from input should be in output map, and only once
    - For n addresses with state/puma combinations that zip codes match
    expected proportions, n needing to be sufficiently large

    This test relies on one of the few cases of two possible ZIP codes:

    state,puma,zipcode,proportion
    6,3756,90706,0.5884
    6,3756,90723,0.4116

    """
    num_ids = 1000  # the number of simulants
    num_unique_ids = int(num_ids / 2)  # the number of unique address_ids
    expected_proportion_90706 = 0.5884  # from PUMA_TO_ZIP_DATA_PATH
    expected_proportion_90723 = 0.4116  # from PUMA_TO_ZIP_DATA_PATH

    simulation_addresses = pd.DataFrame()
    simulation_addresses["address_id"] = [f"123_{n}" for n in range(num_unique_ids)]
    simulation_addresses["state"] = 6  # from PUMA_TO_ZIP_DATA_PATH
    simulation_addresses["puma"] = 3756  # from PUMA_TO_ZIP_DATA_PATH
    simulation_addresses["silly_column"] = "yada yada yada"
    fake_obs_data = pd.concat(
        [simulation_addresses, simulation_addresses]
    )  # concatenation allows for dupe address_id

    # Function under test
    mapper = get_zipcode_map("address_id", fake_obs_data, randomness)

    # Assert that each address_id is in the index once
    assert (
        len(mapper["zipcode"].reset_index()["address_id"].drop_duplicates()) == num_unique_ids
    )

    # Assert that the `num_ids` simulants get assigned the correct proportion of zip code
    fake_obs_data["zipcode"] = simulation_addresses["address_id"].map(mapper["zipcode"])
    assert np.isclose(
        (fake_obs_data["zipcode"].value_counts()[90723] / len(fake_obs_data["zipcode"])),
        expected_proportion_90723,
        rtol=0.1,
    )
    assert np.isclose(
        (fake_obs_data["zipcode"].value_counts()[90706] / len(fake_obs_data["zipcode"])),
        expected_proportion_90706,
        rtol=0.1,
    )
