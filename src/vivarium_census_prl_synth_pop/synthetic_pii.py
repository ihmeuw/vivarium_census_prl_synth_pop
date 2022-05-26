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
