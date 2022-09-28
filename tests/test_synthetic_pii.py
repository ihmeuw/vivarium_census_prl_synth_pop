# Sample Test passing with nose and pytest
import pytest
from types import MethodType

import numpy as np
import pandas as pd
from typing import NamedTuple

from vivarium_census_prl_synth_pop.components import synthetic_pii


def test_generic():
    g = synthetic_pii.GenericGenerator()

    index = [1, 2, 4, 8]
    df_in = pd.DataFrame(index=index)
    df = g.generate(df_in)

    assert len(df) == 4, "expect result to be a dataframe with 4 rows"
    assert np.all(df.index == index), "expect index of result to match initial index"

    df2 = g.noise(df)
    assert np.all(df.index == df2.index) and np.all(
        df.columns == df2.columns
    ), "expect noise to leave dataframe index and columns unchanged"


# This is outdated and not a random generator anymore
# def test_dob():
#     g = synthetic_pii.DOBGenerator(1234)
#
#     index = range(10_000)
#     df_in = pd.DataFrame(index=index)
#     df_in["age"] = np.random.uniform(0, 125, len(index))
#     df = g.generate(df_in)
#
#     assert np.all(df.month <= 12)
#     assert np.all(df.day <= 31)
#     assert np.all(df.year >= 2019 - 125 - 1)
#
#     df2 = g.noise(df)
#     assert np.all(df.index == df2.index) and np.all(
#         df.columns == df2.columns
#     ), "expect noise to leave dataframe index and columns unchanged"
#
#     assert np.all((df2.month >= 1) | df2.month.isnull()) and np.all(
#         (df2.month <= 12)
#         | (df2.day <= 12)  # noise can swap day and month, resulting in a month > 12
#         | df2.month.isnull()
#     )
#     assert np.all((df2.day >= 1) | df2.day.isnull()) and np.all(
#         (df2.day <= 31) | df2.day.isnull()
#     )
#     assert np.all((df2.year >= 2019 - 150) | df2.year.isnull()) and np.all(
#         (df2.year < 2022) | df2.year.isnull()
#     )
#
#     assert not np.all(df.day == df2.day)
#     assert not np.all(df.month == df2.month)
#     assert not np.all(df.year == df2.year)
#     assert not np.all(df.dob == df2.dob)


def get_draw(self, index, additional_key=None) -> pd.Series:
    # Mock get draw function
    return pd.Series(0, index=index)


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

    df2 = g.noise(df)
    assert np.all(df.index == df2.index) and np.all(
        df.columns == df2.columns
    ), "expect noise to leave dataframe index and columns unchanged"


def get_year():
    class MockClock(NamedTuple):
        day = 12,
        month = 31,
        year = 1999

    mock_clock = MockClock()
    return mock_clock



def test_name(mocker):
    g = synthetic_pii.NameGenerator()
    # Get year from clock
    g.clock = mocker.Mock()
    mocker.patch.object(g.clock, "year")
    g.clock.year = MethodType(get_year, g)
    g.clock.return_value = get_year()

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
    df_in["age"] = 0
    df_in["sex"] = "Male"

    df = g.generate_first_and_middle_names(df_in)
    df = g.generate_last_names(df_in)

    assert len(df) == len(all_race_eth_values), "expect result to be a dataframe with 7 rows"

    assert "first_name" in df.columns
    assert "middle_name" in df.columns
    assert "last_name" in df.columns
    assert (
        "AIAN" not in df.last_name.values
    )  # FIXME: come up with a more robust test of the synthetic content


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
