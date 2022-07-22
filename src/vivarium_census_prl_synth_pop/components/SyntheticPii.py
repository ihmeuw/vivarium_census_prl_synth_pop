"""collection of classes for generating sensitive data
synthetically, e.g. name, address, social-security number
"""
import pandas as pd
import numpy as np
from vivarium.framework.engine import Builder

from vivarium_census_prl_synth_pop.constants import data_keys


class GenericGenerator:
    def setup(self, builder: Builder):
        self._rng = np.random.default_rng(builder.configuration.randomness.random_seed)

    def generate(self, df_in: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df_in.index)

    def noise(self, df: pd.DataFrame) -> pd.DataFrame:
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
