# Sample Test passing with nose and pytest
import numpy as np, pandas as pd

from vivarium_census_prl_synth_pop import synthetic_pii

def test_generic():
    g = synthetic_pii.generic_generator(1234)

    index = [1, 2, 4, 8]
    df_in = pd.DataFrame(index=index)
    df = g.generate(df_in)

    assert len(df) == 4, "expect result to be a dataframe with 4 rows"
    assert np.all(df.index == index), "expect index of result to match initial index"

    df2 = g.noise(df)
    assert np.all(df.index == df2.index) and np.all(df.columns == df2.columns), "expect noise to leave dataframe index and columns unchanged"

def test_dob():
    g = synthetic_pii.dob_generator(1234)

    index = range(10)
    df_in = pd.DataFrame(index=index)
    df_in['age'] = np.arange(0, 50, 5)
    df = g.generate(df_in)

    assert np.all(df.month <= 12)
    assert np.all(df.day <= 31)
    assert np.all(df.year >= 1950)

    df2 = g.noise(df)
    assert np.all(df.index == df2.index) and np.all(df.columns == df2.columns), "expect noise to leave dataframe index and columns unchanged"

def test_ssn():
    g = synthetic_pii.ssn_generator(1234)

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
    g = synthetic_pii.name_generator(1234)

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
