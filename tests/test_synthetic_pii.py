# Sample Test passing with nose and pytest
from types import MethodType

import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from vivarium_census_prl_synth_pop.components import synthetic_pii
from vivarium_census_prl_synth_pop.results_processing.names import get_given_name_map

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


def get_last_names():
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
            "White": [0.03, 0.02, 0.01],
            "Latino": [0.003, 0.002, 0.001],
            "Black": [0.01, 0.02, 0.03],
            "Asian": [0.001, 0.002, 0.003],
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
    return {
        "fake_observer": pd.DataFrame(
            {
                "first_name_id": list(range(100_000)),
                "middle_name_id": list(range(100_000)),
                "year_of_birth": [1999, 1999, 2000, 2000] * 25_000,
                "sex": ["Male", "Female"] * 50_000,
            }
        )
    }


def get_name_frequency_proportions(
    names: pd.Series, population_demographics: pd.DataFrame
) -> pd.Series:
    name_totals = pd.concat([population_demographics, names], axis=1).rename(
        columns={0: "name"}
    )
    names_grouped = name_totals.groupby(["year_of_birth", "sex"])["name"].value_counts()
    grouped_totals = name_totals[["year_of_birth", "sex"]].value_counts()
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
        first_names["first_name"], fake_obs_data["fake_observer"]
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
        middle_names["middle_name"], fake_obs_data["fake_observer"]
    )

    assert np.isclose(
        middle_name_proportions.sort_index(), proportions.sort_index(), atol=1e-02
    ).all()
    assert (first_names["first_name"] != middle_names["middle_name"]).any()


@pytest.mark.slow
def test_address():
    g = synthetic_pii.Address()

    index = range(10)
    df_in = pd.DataFrame(index=index)

    df = g.generate(df_in)

    assert len(df) == len(
        index
    ), "expect result to be a dataframe with rows corresponding to `index`"

    assert "zip_code" in df.columns
    assert "address" in df.columns
    assert not np.any(df.address.isnull())
    assert not np.any(df.zip_code.isnull())
    # FIXME: come up with a more robust test of the synthetic content
