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
