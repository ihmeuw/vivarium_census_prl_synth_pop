# Sample Test passing with nose and pytest
from types import MethodType
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from vivarium_census_prl_synth_pop.components import synthetic_pii


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


def get_year():
    class MockClock(NamedTuple):
        day = (12,)
        month = (31,)
        year = 1999

    mock_clock = MockClock()
    return mock_clock


def get_first_names():
    names = pd.DataFrame(
        data={
            "state": ["FL", "FL", "FL", "FL", "FL"],
            "sex": ["Female", "Female", "Male", "Male", "Female"],
            "yob": [1999, 1999, 1999, 2000, 2000],
            "name": ["Mary", "Annie", "Louise", "John", "Jeff"],
            "freq": [200, 199, 198, 10, 11],
        }
    )
    names = names.set_index(["state", "sex", "yob", "name", "freq"])
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


def test_name(mocker):
    # todo: update test
    g = synthetic_pii.NameGenerator()
    # Mock randomness and get_draw
    g.randomness = mocker.Mock()
    mocker.patch.object(g.randomness, "get_draw")
    g.randomness.get_draw = MethodType(get_draw, g)

    # Mock first name data
    g.first_name_data = get_first_names()
    # Mock last name data
    g.last_name_data = get_last_names()

    all_race_eth_values = [
        "White",
        "Latino",
        "Black",
        "Asian",
        "Multiracial or Other",
        "AIAN",
        "NHOPI",
    ]
    index = range(len(all_race_eth_values))
    df_in = pd.DataFrame(index=index)
    df_in["race_ethnicity"] = all_race_eth_values
    df_in["year_of_birth"] = 1985
    df_in["sex"] = "Male"

    series1 = g.generate_first_and_middle_names(df_in, "dummy_key")
    # This will be updated when last names are processed
    # df2 = g.generate_last_names(df_in)

    assert series1.index.is_unique


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
