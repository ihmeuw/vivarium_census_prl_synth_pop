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


class SSNGenerator(GenericGenerator):
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
def load_first_name_data():
    global _g_ssn_names

    df_ssn_names = pd.read_csv('/snfs1/Project/simulation_science/prl/data/ssn_names/FL.TXT',
                                names=['state', 'sex', 'yob', 'name', 'freq'])
    df_ssn_names['age'] = 2020 - df_ssn_names.yob
    df_ssn_names['sex'] = df_ssn_names.sex.map({'M':'Male', 'F':'Female'})
    _g_ssn_names = df_ssn_names.groupby(['age', 'sex'])


def random_first_names(rng, age, sex, size):
    global _g_ssn_names

    if _g_ssn_names is None:
        load_first_name_data()

    t = _g_ssn_names.get_group((age, sex))
    p = t.freq / t.freq.sum()
    return rng.choice(t.name, size=size, replace=True, p=p) # TODO: include spaces and hyphens


_df_census_names = None # global data on last names, to avoid loading repeatedly
def load_last_name_data():
    global _df_census_names

    _df_census_names = pd.read_csv('/snfs1/Project/simulation_science/prl/data/Names_2010Census.csv', na_values=['(S)'])

    _df_census_names.name = _df_census_names.name.str.capitalize()

    # fill missing values with equal amounts of what is left
    n_missing = _df_census_names.filter(like='pct').isnull().sum(axis=1)
    pct_total = _df_census_names.filter(like='pct').sum(axis=1)
    pct_fill = (100 - pct_total) / n_missing

    for col in _df_census_names.filter(like='pct').columns:
        _df_census_names[col] = _df_census_names[col].fillna(pct_fill)

    # drop final row
    _df_census_names = _df_census_names.iloc[:-1]


    n = _df_census_names['count'].copy()
    for race_eth, col in [['White', 'pctwhite'],
                          ['Latino', 'pcthispanic'],
                          ['Black', 'pctblack'],
                          ['Asian', 'pctapi'],
                          ['Multiracial or Other', 'pct2prace'],
                          ['AIAN', 'pctaian'],
                          ['NHOPI', 'pctapi']]:
        p = n * _df_census_names[col] / 100
        p /= p.sum()
        _df_census_names[race_eth] = p


def random_last_names(rng, race_eth, size):
    global _df_census_names

    if _df_census_names is None:
        load_last_name_data()

    s_last = rng.choice(_df_census_names.name, p=_df_census_names[race_eth], size=size)

    # Last names sometimes also include spaces or hyphens, and I have
    # come up with race/ethnicity specific space and hyphen
    # probabilities from an analysis of voter registration data (from
    # publicly available data from North Carolina, filename
    # VR_Snapshot_20220101.txt; see
    # 2022_06_02b_prl_code_for_probs_of_spaces_and_hyphens_in_last_and_first_names.ipynb
    # for computation details.)

    # add hyphens to some names
    p_hyphen = {
             'AIAN': 0.00685,
             'Asian': 0.00682,
             'Black': 0.01326,
             'Latino': 0.06842,
             'Multiracial or Other': 0.01791,
             'NHOPI': 0.02064,
             'White': 0.00474}[race_eth]

    hyphen_rows = (rng.uniform(0, 1, size=len(s_last)) < p_hyphen)
    s_last[hyphen_rows] += '-' + rng.choice(_df_census_names.name,
                                            p=_df_census_names[race_eth],
                                            size=hyphen_rows.sum())

    # add spaces to some names
    p_space = {
             'AIAN': 0.00408,
             'Asian': 0.0086,
             'Black': 0.0041,
             'Latino': 0.12807,
             'Multiracial or Other': 0.02004,
             'NHOPI': 0.02064,
             'White': 0.00347}[race_eth]

    space_rows = (rng.uniform(0, 1, size=len(s_last)) < p_space*(1-hyphen_rows))  # HACK: don't put spaces in names that are already hyphenated
    s_last[space_rows] += ' ' + rng.choice(_df_census_names.name,
                                           p=_df_census_names[race_eth],
                                           size=space_rows.sum())

    return s_last


class NameGenerator(GenericGenerator):
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
        s = pd.Series(index=df_in.index, dtype=str)
        for race_eth, df_race_eth in df_in.groupby('race_ethnicity'):
            s.loc[df_race_eth.index] = random_last_names(self._rng, race_eth, len(df_race_eth))
        # TODO: include household structure
        return s

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
        substitute_first_name_list = 'Girl, Mom, A, Goh, Mother, Adult, Grandchild, Mr, Adult male, Granddaughter, Mrs, B, Grandson, Ms, Baby, H, N, Boy, Hija, Nephew, Brother, Hijo, Nino, C, House, O, Child, Husband, Oldest, Child f, Inmate, One, Coh, J, P, D, K, Person, Dad, Kid, R, Dau, L, Resident, Daughter, Lady, Respondent, Daughter of, Lady in the, S, Doh, Lady of, Senor, E, Lady of house, Senora, F, Lady of the, Sister, Father, Loh, Soh, Female, M, Son, Female child, Male, Son of, Friend, Male child, T, G, Man, V, Gent, Man in the, W, Gentelman, Man of, Wife, Gentle, Man of the, Woman, Gentleman, Minor, Youngest, Gentleman of, Miss, Gentlemen, Moh'.split(', ')
        substitute_last_name_list = 'Hh, Of the house, A, Hhm, One, Adult, Home, Owner, Anon, House, P, Anonymous, Household, Parent, Apellido, Householder, Person, B, Husband, R, Boy, J, Ref, C, K, Refuse, Casa, L, Resident, Child, Lady, Resp, Coh, Lady of house, Respondant, D, Lady of the house, Respondent, Daughter, Last name, S, De casa, Loh, Soh, De la casa, M, Son, Declined, Male, T, Doe, Man, The house, Doh, Man of the house, Three, Dont know, Moh, Two, E, N, Unk, F, Na, Unknown, Female, No, W, Four, No last name, Wife, Friend, No name, X, G, None, Xxx, Girl, O, Y, Goh, Occupant, Younger, H, Of house, H age, Of the home'.split(', ')


        return df


class AddressGenerator(GenericGenerator):
    def load_address_data(self):
        if not hasattr(self, '_df_deepparse_address_data'):
            self._df_deepparse_address_data = pd.read_csv('/snfs1/Project/simulation_science/prl/data/deepparse_address_data_usa.csv.bz2')
        return self._df_deepparse_address_data

    def generate(self, df_in : pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic addresses for individuals

        Parameters
        ----------

        df_in : pd.DataFrame

        Results
        -------
        returns pd.DataFrame with address data, stored in two
        string columns `address` and `zip_code`

        """
        df_address = self.load_address_data()
        
        df = pd.DataFrame(index=df_in.index)
        N = len(df)

        s_address = pd.Series('', index=df.index, name='address')

        for col in ['StreetNumber', 'StreetName', 'Unit']:
            i = self._rng.choice(df_address.index, size=(N,))
            s_address += df_address.loc[i, col].fillna('').values
            s_address += ' '

        # handle Municipality, Province, PostalCode separately
        # to keep them perfectly correlated
        i = self._rng.choice(df_address[df_address.Province == 'tx'].index, size=(N,))
        s_address += df_address.loc[i, 'Municipality'].fillna('').values
        s_address += ', '
        s_address += df_address.loc[i, 'Province'].fillna('').values

        df['address'] = s_address
        df['zip_code'] = df_address.loc[i, 'PostalCode'].fillna('').values
        return df
