"""collection of classes for generating sensitive data
synthetically, e.g. name, address, social-security number
"""
import pandas as pd
import numpy as np

class generic_generator:
    def __init__(self, seed : int):
        self._rng = np.random.default_rng(seed)

    def generate(self, df_in : pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df_in.index)

    def noise(self, df : pd.DataFrame) -> pd.DataFrame:
        return df


class ssn_generator(generic_generator):
    def generate(self, df_in : pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic Social Security Numbers

        Parameters
        ----------
        df_in : pd.DataFrame

        Results
        -------
        returns pd.DataFrame with SSN information, encoded in three
        numeric columns `ssn_area`, `ssn_group`, `ssn_serial`, and one
        str column that puts these together with dashes called `ssn`

        Notes
        -----
        See https://www.ssa.gov/kc/SSAFactSheet--IssuingSSNs.pdf for
        details on the format of SSNs.

        """

        df = pd.DataFrame(index=df_in.index)
        
        n = len(df)

        area = self._rng.integers(1, 899, size=n)
        area = np.where(area == 666, 667, area)
        df['ssn_area'] = area

        group = self._rng.integers(1, 99, size=n)
        df['ssn_group'] = group

        serial = self._rng.integers(1, 9999, size=n)
        df['ssn_serial'] = serial

        df['ssn'] = ''
        df['ssn'] += df.ssn_area.astype(str).str.zfill(3)
        df['ssn'] += '-'
        df['ssn'] += df.ssn_group.astype(str).str.zfill(2)
        df['ssn'] += '-'
        df['ssn'] += df.ssn_serial.astype(str).str.zfill(4)
        return df

    def noise(self, df):
        df = df.copy()

        # TODO: add some errors in digits
        # typically just getting one digit wrong

        n_to_blank = len(df.index) // 10  # TODO: make this an optional parameter to this method and/or inform it with some evidence
        if n_to_blank > 0:
            blank_rows = self._rng.choice(df.index,
                                          size=n_to_blank,
                                          replace=False)
            df.loc[blank_rows, 'ssn'] = ''

        return df

_g_ssn_names = None # global data on first names, to avoid loading repeatedly
def random_first_names(rng, age, sex, size):
    global _g_ssn_names

    if _g_ssn_names is None:
        df_ssn_names = pd.read_csv('/home/j/Project/simulation_science/prl/data/ssn_names/FL.TXT',
                                    names=['state', 'sex', 'yob', 'name', 'freq'])
        df_ssn_names['age'] = 2020 - df_ssn_names.yob
        df_ssn_names['sex'] = df_ssn_names.sex.map({'M':'Male', 'F':'Female'})
        _g_ssn_names = df_ssn_names.groupby(['age', 'sex'])

    t = _g_ssn_names.get_group((age, sex))
    p = t.freq / t.freq.sum()
    return rng.choice(t.name, size=size, replace=True, p=p)

class name_generator(generic_generator):
    def generate(self, df_in : pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic names for individuals

        Parameters
        ----------

        df_in : pd.DataFrame, with columns race_ethnicity, sex, age,
        relationship_to_household_head, and household_id

        Results
        -------
        returns pd.DataFrame with name data, stored in three
        string columns `first_name`, `middle_name`, `last_name`

        """

        df = self._generate_first_and_middle_names(df_in)
        df['last_name'] = self._generate_last_names(df_in)
        return df

    def _generate_first_and_middle_names(self, df_in : pd.DataFrame) -> pd.DataFrame:
        # first and middle names
        # strategy: calculate year of birth based on age, use it with sex and state to find a representative name
        df = pd.DataFrame(index=df_in.index)

        for (age,sex), df_age in df_in.groupby(['age', 'sex']):
            df.loc[df_age.index, 'first_name'] = random_first_names(self._rng, age, sex, len(df_age))
            df.loc[df_age.index, 'middle_name'] = random_first_names(self._rng, age, sex, len(df_age))

        return df

    def _generate_last_names(self, df_in : pd.DataFrame) -> pd.DataFrame:
        return df_in.race_ethnicity.copy()

    def noise(self, df):
        df = df.copy()

        # TODO: add some errors

        n_to_blank = len(df.index) // 10  # TODO: make this an optional parameter to this method and/or inform it with some evidence
        if n_to_blank > 0:
            blank_rows = self._rng.choice(df.index,
                                          size=n_to_blank,
                                          replace=False)
            df.loc[blank_rows, 'first_name'] = ''
            df.loc[blank_rows, 'middle_name'] = ''
            # TODO: include common substitutes for first names

        return df
