# Sample Test passing with nose and pytest
import pytest

import numpy as np, pandas as pd

from vivarium_census_prl_synth_pop import synthetic_pii_deprecated

def test_generic():
    g = synthetic_pii.GenericGenerator(1234)

    index = [1, 2, 4, 8]
    df_in = pd.DataFrame(index=index)
    df = g.generate(df_in)

    assert len(df) == 4, "expect result to be a dataframe with 4 rows"
    assert np.all(df.index == index), "expect index of result to match initial index"

    df2 = g.noise(df)
    assert np.all(df.index == df2.index) and np.all(df.columns == df2.columns), "expect noise to leave dataframe index and columns unchanged"

def test_dob():
    g = synthetic_pii.DOBGenerator(1234)

    index = range(10_000)
    df_in = pd.DataFrame(index=index)
    df_in['age'] = np.random.uniform(0, 125, len(index))
    df = g.generate(df_in)

    assert np.all(df.month <= 12)
    assert np.all(df.day <= 31)
    assert np.all(df.year >= 2019 - 125 - 1)

    df2 = g.noise(df)
    assert np.all(df.index == df2.index) and np.all(df.columns == df2.columns), "expect noise to leave dataframe index and columns unchanged"

    assert (np.all((df2.month >= 1) | df2.month.isnull())
            and np.all((df2.month <= 12)
                       | (df2.day <= 12) # noise can swap day and month, resulting in a month > 12
                       | df2.month.isnull()
                      )
           )
    assert np.all((df2.day >= 1) | df2.day.isnull()) and np.all((df2.day <= 31) | df2.day.isnull())
    assert np.all((df2.year >= 2019 - 150) | df2.year.isnull()) and np.all((df2.year < 2022) | df2.year.isnull())
    
    assert not np.all(df.day == df2.day)
    assert not np.all(df.month == df2.month)
    assert not np.all(df.year == df2.year)
    assert not np.all(df.dob == df2.dob)

def test_ssn():
    g = synthetic_pii.SSNGenerator(1234)

    index = range(10)
    df_in = pd.DataFrame(index=index)
    df = g.generate(df_in)

    assert len(df) == 10, "expect result to be a dataframe with 10 rows"

    ssn1 = df.loc[0, 'ssn']
    area, group, serial = ssn1.split('-')
    assert len(area) == 3
    assert len(group) == 2
    assert len(serial) == 4

    df2 = g.noise(df)
    assert np.all(df.index == df2.index) and np.all(df.columns == df2.columns), "expect noise to leave dataframe index and columns unchanged"


def test_name():
    g = synthetic_pii.NameGenerator(1234)

    all_race_eth_values = ['White', 'Latino', 'Black', 'Asian', 'Multiracial or Other', 'AIAN', 'NHOPI']
    index = range(len(all_race_eth_values))
    df_in = pd.DataFrame(index=index)
    df_in['race_ethnicity'] = all_race_eth_values
    df_in['age'] = 0
    df_in['sex'] = 'Male'

    df = g.generate(df_in)

    assert len(df) == len(all_race_eth_values), "expect result to be a dataframe with 7 rows"

    assert 'first_name' in df.columns
    assert 'middle_name' in df.columns
    assert 'last_name' in df.columns
    assert 'AIAN' not in df.last_name.values # FIXME: come up with a more robust test of the synthetic content

    
@pytest.mark.slow
def test_address():
    g = synthetic_pii.AddressGenerator(1234)

    index = range(10)
    df_in = pd.DataFrame(index=index)

    df = g.generate(df_in)

    assert len(df) == len(index), "expect result to be a dataframe with rows corresponding to `index`"

    assert 'zip_code' in df.columns
    assert 'address' in df.columns
    assert not np.any(df.address.isnull())
    assert not np.any(df.zip_code.isnull())
    # FIXME: come up with a more robust test of the synthetic content
