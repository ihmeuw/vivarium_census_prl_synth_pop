"""collection of classes for generating sensitive data
synthetically, e.g. name, address, social-security number
"""
import pandas as pd
import numpy as np
from vivarium.framework.engine import Builder

from vivarium_census_prl_synth_pop.constants import data_keys, data_values


class GenericGenerator:
    def setup(self, builder: Builder):
        self._rng = np.random.default_rng(builder.configuration.randomness.random_seed)

    def generate(self, df_in: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df_in.index)

    def noise(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class NameGenerator(GenericGenerator):
    @property
    def name(self):
        return "NameGenerator"

    def setup(self, builder: Builder):
        super().setup(builder)
        self.first_name_data = builder.data.load(data_keys.SYNTHETIC_DATA.FIRST_NAMES)
        #TODO: maybe update artifact
        self.first_name_data['age'] = 2020 - self.first_name_data['yob']
        self.last_name_data = builder.data.load(data_keys.SYNTHETIC_DATA.LAST_NAMES)

    def random_first_names(self, rng, age, sex, size):
        g_ssn_names = self.first_name_data.groupby(['age', 'sex'])
        # TODO: UNDERSTAND / RENAME
        t = g_ssn_names.get_group((age, sex))
        p = t.freq / t.freq.sum()
        return rng.choice(t.name, size=size, replace=True, p=p)  # TODO: include spaces and hyphens

    def random_last_names(self, rng, race_eth, size):
        df_census_names = self.last_name_data

        # TODO: UNDERSTAND, MAYBE CLARIFY
        s_last = rng.choice(df_census_names.name, p=df_census_names[race_eth], size=size)

        # TODO: MOVE THIS INTO CONSTANTS
        # add hyphens to some names
        p_hyphen = data_values.PROBABILITY_OF_HYPHEN_IN_NAME[race_eth]

        hyphen_rows = (rng.uniform(0, 1, size=len(s_last)) < p_hyphen)
        s_last[hyphen_rows] += '-' + rng.choice(df_census_names.name,
                                                p=df_census_names[race_eth],
                                                size=hyphen_rows.sum())

        # add spaces to some names
        p_space = data_values.PROBABILITY_OF_SPACE_IN_NAME[race_eth]

        space_rows = (rng.uniform(0, 1, size=len(s_last)) < p_space * (
                    1 - hyphen_rows))  # HACK: don't put spaces in names that are already hyphenated
        s_last[space_rows] += ' ' + rng.choice(df_census_names.name,
                                               p=df_census_names[race_eth],
                                               size=space_rows.sum())

        return s_last

    def generate(self, df_in: pd.DataFrame) -> pd.DataFrame:
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

    def _generate_first_and_middle_names(self, df_in: pd.DataFrame) -> pd.DataFrame:
        # first and middle names
        # strategy: calculate year of birth based on age, use it with sex and state to find a representative name
        df = pd.DataFrame(index=df_in.index)

        for (age, sex), df_age in df_in.groupby(['age', 'sex']):
            df.loc[df_age.index, 'first_name'] = self.random_first_names(self._rng, age, sex, len(df_age))
            df.loc[df_age.index, 'middle_name'] = self.random_first_names(self._rng, age, sex, len(df_age))

        return df

    def _generate_last_names(self, df_in: pd.DataFrame) -> pd.DataFrame:
        s = pd.Series(index=df_in.index, dtype=str)
        for race_eth, df_race_eth in df_in.groupby('race_ethnicity'):
            s.loc[df_race_eth.index] = self.random_last_names(self._rng, race_eth, len(df_race_eth))
        # TODO: include household structure
        return s

    def noise(self, df):
        df = df.copy()

        # TODO: add some errors

        n_to_blank = len(
            df.index) // 10  # TODO: make this an optional parameter to this method and/or inform it with some evidence
        if n_to_blank > 0:
            blank_rows = self._rng.choice(df.index,
                                          size=n_to_blank,
                                          replace=False)
            df.loc[blank_rows, 'first_name'] = ''
            df.loc[blank_rows, 'middle_name'] = ''
            # TODO: include common substitutes for first names
        substitute_first_name_list = 'Girl, Mom, A, Goh, Mother, Adult, Grandchild, Mr, Adult male, Granddaughter, Mrs, B, Grandson, Ms, Baby, H, N, Boy, Hija, Nephew, Brother, Hijo, Nino, C, House, O, Child, Husband, Oldest, Child f, Inmate, One, Coh, J, P, D, K, Person, Dad, Kid, R, Dau, L, Resident, Daughter, Lady, Respondent, Daughter of, Lady in the, S, Doh, Lady of, Senor, E, Lady of house, Senora, F, Lady of the, Sister, Father, Loh, Soh, Female, M, Son, Female child, Male, Son of, Friend, Male child, T, G, Man, V, Gent, Man in the, W, Gentelman, Man of, Wife, Gentle, Man of the, Woman, Gentleman, Minor, Youngest, Gentleman of, Miss, Gentlemen, Moh'.split(
            ', ')
        substitute_last_name_list = 'Hh, Of the house, A, Hhm, One, Adult, Home, Owner, Anon, House, P, Anonymous, Household, Parent, Apellido, Householder, Person, B, Husband, R, Boy, J, Ref, C, K, Refuse, Casa, L, Resident, Child, Lady, Resp, Coh, Lady of house, Respondant, D, Lady of the house, Respondent, Daughter, Last name, S, De casa, Loh, Soh, De la casa, M, Son, Declined, Male, T, Doe, Man, The house, Doh, Man of the house, Three, Dont know, Moh, Two, E, N, Unk, F, Na, Unknown, Female, No, W, Four, No last name, Wife, Friend, No name, X, G, None, Xxx, Girl, O, Y, Goh, Occupant, Younger, H, Of house, H age, Of the home'.split(
            ', ')

        return df


class AddressGenerator(GenericGenerator):

    @property
    def name(self):
        return "AddressGenerator"

    def setup(self, builder: Builder):
        super().setup(builder)
        self.address_data = builder.data.load(data_keys.SYNTHETIC_DATA.ADDRESSES)

    def generate(self, idx: pd.Index, state: str) -> pd.DataFrame:
        """Generate synthetic addresses for individuals

        Parameters
        ----------
        idx : pd.Index

        Results
        -------
        returns pd.DataFrame with address data, stored in two
        string columns `address` and `zip_code`

        Caution
        -------
        there is a (likely very small) chance this function could return two non-unique addresses:
        for example, by sampling:
        212 E 18th St, Seattle, WA 98765
        536 Garfield Pl, Brooklyn, NY 11215
        212 Prospect Park West, Brooklyn, NY 11215

        we could, in two different ways, get the address:
        212 Garfield Pl, Seattle, WA 98765
        """
        df = pd.DataFrame(index=idx)
        N = len(df)

        synthetic_address = pd.Series('', index=df.index, name='address')

        for col in ['StreetNumber', 'StreetName', 'Unit']:
            chosen_indices = self._rng.choice(self.address_data.index, size=(N,))
            synthetic_address += self.address_data.loc[chosen_indices, col].fillna('').values
            synthetic_address += ' '

        # handle Municipality, Province, PostalCode separately
        # to keep them perfectly correlated
        chosen_indices = self._rng.choice(self.address_data[self.address_data.Province == state].index, size=(N,))
        synthetic_address += self.address_data.loc[chosen_indices, 'Municipality'].fillna('').values
        synthetic_address += ', '
        synthetic_address += self.address_data.loc[chosen_indices, 'Province'].fillna('').values

        df['address'] = synthetic_address
        df['zipcode'] = self.address_data.loc[chosen_indices, 'PostalCode'].fillna('').values
        return df
