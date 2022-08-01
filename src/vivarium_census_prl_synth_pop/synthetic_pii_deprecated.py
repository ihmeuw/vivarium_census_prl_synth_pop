"""collection of classes for generating sensitive data
synthetically, e.g. name, address, social-security number
"""
import pandas as pd
import numpy as np

class GenericGenerator:
    def __init__(self, seed : int):
        self._rng = np.random.default_rng(seed)

    def generate(self, df_in : pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df_in.index)

    def noise(self, df : pd.DataFrame) -> pd.DataFrame:
        return df

def make_dob_col(df):
    """Append year-month-day in a column called 'dob'
    """
    df['dob'] = (df.year.fillna('').astype(str) + '-'
                 + df.month.fillna('').astype(str).str.zfill(2) + '-'
                 + df.day.fillna('').astype(str).str.zfill(2))

class DOBGenerator(GenericGenerator):
    def generate(self, df_in : pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic Date of Birth

        Parameters
        ----------
        df_in : pd.DataFrame, containing column `age`

        Results
        -------
        returns pd.DataFrame with date-of-birth information, encoded in three
        numeric columns `year`, `month`, `day`, and one
        pd.Timestamp column that puts these together with dashes called `dob`

        """
        df = pd.DataFrame(index=df_in.index)

        data_date = pd.Timestamp('2019-06-01')  # TODO: decide if this is the right date

        age = 365.25 * df_in.age
        age += self._rng.uniform(low=0, high=365, size=len(df))
        dob = data_date - pd.to_timedelta(np.round(age), unit='days')

        df['year'] = dob.dt.year
        df['month'] = dob.dt.month
        df['day'] = dob.dt.day
        make_dob_col(df)

        return df

    def noise(self, df, pr_field_error=0.0106, pr_full_error=0.0026,
              pr_missing=0.0024, pr_month_day_swap=0.0018):
        """Add noise to synthetic Date of Birth

        Parameters
        ----------
        df : pd.DataFrame
        pr_field_error : float, optional
        pr_full_error : float, optional
        pr_missing : float, optional
        pr_month_day_swap : float, optional

        Notes
        -----
        Default values based on Buzz's experience.
        """

        df = df.copy()

        # make a small error in each column with probability (around
        # 1%), that sums up to probability of a single field error
        for col in ['year', 'month', 'day']:
            rows = (self._rng.uniform(size=len(df)) < pr_field_error)
            df.loc[rows, col] = df.loc[rows, col] + np.random.choice([-2, -1, 1, 2], size=np.sum(rows, dtype=int))  # TODO: investigate error distribution for these errors (current approach is not evidence-based)
        df.month = np.clip(df.month, 1, 12, dtype=int)
        df.day = np.clip(df.day, 1, 31, dtype=int)  # NOTE: it is acceptible to have an error that has a day of 31 in a month with less days (because it is an erroneous DOB)

        # get the whole thing wrong sometimes
        rows = (self._rng.uniform(size=len(df)) < pr_full_error)
        swap_rows = self._rng.choice(df.index, sum(rows))
        df.loc[rows, 'day'] = df.loc[swap_rows, 'day']
        df.loc[rows, 'month'] = df.loc[swap_rows, 'month']
        df.loc[rows, 'year'] = df.loc[swap_rows, 'year']

        # leave dob blank occasionally
        rows = (self._rng.uniform(size=len(df)) < pr_missing)
        df.loc[rows, 'day'] = np.nan
        df.loc[rows, 'month'] = np.nan
        df.loc[rows, 'year'] = np.nan

        # transpose day and month occasionally
        rows = (self._rng.uniform(size=len(df)) < pr_month_day_swap)
        s_day = df.loc[rows, 'day']
        df.loc[rows, 'day'] = df.loc[rows, 'month'].values
        df.loc[rows, 'month'] = s_day.values
                    
        make_dob_col(df)
        
        return df
