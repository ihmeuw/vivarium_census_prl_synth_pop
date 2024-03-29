"""collection of classes for generating sensitive data
synthetically, e.g. name, address, social-security number
"""
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable

from vivarium_census_prl_synth_pop.constants import data_keys, data_values, metadata
from vivarium_census_prl_synth_pop.utilities import random_integers, vectorized_choice

Array = Union[List, Tuple, np.ndarray, pd.Series]


class GenericGenerator:
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

    def generate(self, df_in: pd.DataFrame, additional_key: Any = None) -> pd.DataFrame:
        return pd.DataFrame(index=df_in.index)

    def noise(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class NameGenerator(GenericGenerator):
    @property
    def name(self):
        return "NameGenerator"

    def setup(self, builder: Builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.first_name_data = builder.data.load(data_keys.SYNTHETIC_DATA.FIRST_NAMES)
        self.last_name_data = builder.data.load(data_keys.SYNTHETIC_DATA.LAST_NAMES)

    def random_first_names(
        self, yob: int, sex: str, size: int, additional_key: Any = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        yob: the year of birth of the sims for whom to sample a name
        sex: the sex of the sims for whom to sample a name
        size: the number of sample to return
        additional_key: additional randomness key to pass vivarium.randomness

        Returns
        -------
        if [yob] <= 2020:
            nd.ndarray of [size] names sampled from the first names of people of sex [sex], born in the year [yob]
        if [yob] > 2020:
            nd.ndarray of [size] names sampled from the first names of people of sex [sex], born in the year 2020
        """
        # we only have data up to 2020; for younger children, sample from 2020 names.
        if yob > 2020:
            yob = 2020
        grouped_name_data = self.first_name_data.reset_index().groupby(["yob", "sex"])
        age_sex_specific_names = grouped_name_data.get_group((yob, sex))
        name_probabilities = (
            age_sex_specific_names["freq"] / age_sex_specific_names["freq"].sum()
        )
        return vectorized_choice(
            options=age_sex_specific_names.name,
            n_to_choose=size,
            randomness_stream=self.randomness,
            weights=name_probabilities,
            additional_key=additional_key,
        ).to_numpy()  # TODO: include spaces and hyphens

    def random_last_names(
        self,
        race_eth: str,
        size: int,
        additional_key: Any = None,
    ) -> np.ndarray:
        """

        Parameters
        ----------
        randomness: randomness stream
        race_eth: the race_ethnicity category (string) of the sims for whom to sample a name
        size: the number of samples to return
        additional_key: additional randomness key to pass vivarium.randomness

        Returns
        -------
        nd.ndarray of [size] last names sampled from people of race and ethnicity [race_eth]
        """
        df_census_names = self.last_name_data.reset_index()

        # randomly sample last names
        last_names = vectorized_choice(
            options=df_census_names.name,
            n_to_choose=size,
            randomness_stream=self.randomness,
            weights=df_census_names[race_eth],
            additional_key=additional_key,
        )

        # Last names sometimes also include spaces or hyphens, and abie has
        # come up with race/ethnicity specific space and hyphen
        # probabilities from an analysis of voter registration data (from
        # publicly available data from North Carolina, filename
        # VR_Snapshot_20220101.txt; see
        # 2022_06_02b_prl_code_for_probs_of_spaces_and_hyphens_in_last_and_first_names.ipynb
        # for computation details.)

        # for some names, add a hyphen between two randomly samples last names
        probability_of_hyphen = data_values.PROBABILITY_OF_HYPHEN_IN_NAME[race_eth]
        hyphen_rows = (
            self.randomness.get_draw(last_names.index, "choose_hyphen_sims")
            < probability_of_hyphen
        )
        if hyphen_rows.sum() > 0:
            last_names[hyphen_rows] += (
                "-"
                + vectorized_choice(
                    options=df_census_names.name,
                    n_to_choose=hyphen_rows.sum(),
                    randomness_stream=self.randomness,
                    weights=df_census_names[race_eth],
                    additional_key="hyphen_last_names",
                ).to_numpy()
            )

        # add spaces to some names
        probability_of_space = data_values.PROBABILITY_OF_SPACE_IN_NAME[race_eth]
        space_rows = self.randomness.get_draw(
            last_names.index, "choose_space_sims"
        ) < probability_of_space * (
            1 - hyphen_rows
        )  # HACK: don't put spaces in names that are already hyphenated
        if space_rows.sum() > 0:
            last_names[space_rows] += (
                " "
                + vectorized_choice(
                    options=df_census_names.name,
                    n_to_choose=space_rows.sum(),
                    randomness_stream=self.randomness,
                    weights=df_census_names[race_eth],
                    additional_key="space_last_names",
                ).to_numpy()
            )

        return last_names.to_numpy()

    def generate_first_and_middle_names(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic names for individuals
        Parameters
        ----------
        df_in : pd.DataFrame, with columns sex, age
        Results
        -------
        returns pd.DataFrame with name data, stored in
        string columns `first_name`, `middle_name`,
        """
        # first and middle names
        # strategy: calculate year of birth based on age, use it with sex and state to find a representative name
        first_and_middle = pd.DataFrame(index=df_in.index)
        current_year = self.clock().year
        for (age, sex), df_age in df_in.groupby(["age", "sex"]):
            n = len(df_age)
            first_and_middle.loc[df_age.index, "first_name"] = self.random_first_names(
                current_year - np.floor(age), sex, n, "first"
            )
            first_and_middle.loc[df_age.index, "middle_name"] = self.random_first_names(
                current_year - np.floor(age), sex, n, "middle"
            )

        return first_and_middle

    def generate_last_names(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic names for individuals

        Parameters
        ----------

        df_in : pd.DataFrame, with column race_ethnicity

        Results
        -------
        returns pd.DataFrame with name data, stored in
        string column `last_name`

        """
        last_names = pd.Series(index=df_in.index, dtype=str)
        for race_eth, df_race_eth in df_in.groupby("race_ethnicity"):
            last_names.loc[df_race_eth.index] = self.random_last_names(
                race_eth, len(df_race_eth), "last_name"
            )
        # TODO: include household structure
        return pd.DataFrame(last_names, columns=["last_name"])

    def noise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # TODO: add some errors

        n_to_blank = (
            len(df.index) // 10
        )  # TODO: make this an optional parameter to this method and/or inform it with some evidence
        if n_to_blank > 0:
            blank_rows = vectorized_choice(
                options=df.index, n_to_choose=n_to_blank, randomness_stream=self.randomness
            )
            df.loc[blank_rows, "first_name"] = ""
            df.loc[blank_rows, "middle_name"] = ""
            # TODO: include common substitutes for first names
        substitute_first_name_list = "Girl, Mom, A, Goh, Mother, Adult, Grandchild, Mr, Adult male, Granddaughter, Mrs, B, Grandson, Ms, Baby, H, N, Boy, Hija, Nephew, Brother, Hijo, Nino, C, House, O, Child, Husband, Oldest, Child f, Inmate, One, Coh, J, P, D, K, Person, Dad, Kid, R, Dau, L, Resident, Daughter, Lady, Respondent, Daughter of, Lady in the, S, Doh, Lady of, Senor, E, Lady of house, Senora, F, Lady of the, Sister, Father, Loh, Soh, Female, M, Son, Female child, Male, Son of, Friend, Male child, T, G, Man, V, Gent, Man in the, W, Gentelman, Man of, Wife, Gentle, Man of the, Woman, Gentleman, Minor, Youngest, Gentleman of, Miss, Gentlemen, Moh".split(
            ", "
        )
        substitute_last_name_list = "Hh, Of the house, A, Hhm, One, Adult, Home, Owner, Anon, House, P, Anonymous, Household, Parent, Apellido, Householder, Person, B, Husband, R, Boy, J, Ref, C, K, Refuse, Casa, L, Resident, Child, Lady, Resp, Coh, Lady of house, Respondant, D, Lady of the house, Respondent, Daughter, Last name, S, De casa, Loh, Soh, De la casa, M, Son, Declined, Male, T, Doe, Man, The house, Doh, Man of the house, Three, Dont know, Moh, Two, E, N, Unk, F, Na, Unknown, Female, No, W, Four, No last name, Wife, Friend, No name, X, G, None, Xxx, Girl, O, Y, Goh, Occupant, Younger, H, Of house, H age, Of the home".split(
            ", "
        )

        return df


class Address(GenericGenerator):
    @property
    def name(self):
        return "Address"

    def setup(self, builder: Builder):
        super().setup(builder)
        self.address_data = builder.data.load(data_keys.SYNTHETIC_DATA.ADDRESSES)

    def generate(self, idx: pd.Series, state: str) -> pd.DataFrame:
        """Generate synthetic addresses for individuals

        Parameters
        ----------
        idx : pd.Series that is a list of ids to be used as an index

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

        synthetic_address = pd.Series("", index=df.index, name="address")

        for col in ["StreetNumber", "StreetName", "Unit"]:
            chosen_indices = vectorized_choice(
                options=self.address_data.index,
                n_to_choose=N,
                randomness_stream=self.randomness,
            )
            synthetic_address += self.address_data.loc[chosen_indices, col].fillna("").values
            synthetic_address += " "

        # handle Municipality, Province, PostalCode separately
        # to keep them perfectly correlated
        chosen_indices = vectorized_choice(
            options=self.address_data[self.address_data.Province == state].index,
            n_to_choose=N,
            randomness_stream=self.randomness,
        )
        synthetic_address += (
            self.address_data.loc[chosen_indices, "Municipality"].fillna("").values
        )
        synthetic_address += ", "
        synthetic_address += (
            self.address_data.loc[chosen_indices, "Province"].fillna("").values
        )

        df["address"] = synthetic_address
        df["zipcode"] = self.address_data.loc[chosen_indices, "PostalCode"].fillna("").values
        return df

    def get_new_addresses_and_zipcodes(
        self, those_that_move: pd.Series, state: str
    ) -> Tuple[Dict, Dict]:
        """
        Parameters
        ----------
        those_that_move: a pd.Series of ids (e.g. employer_id or household_id) who will move
        state: US state. e.g., Florida

        Returns
        -------
        ({those_that_move: addresses}, {those_that_move: zipcodes})
        """
        new_addresses = self.generate(those_that_move, state=state)
        return (new_addresses["address"].to_dict(), new_addresses["zipcode"].to_dict())


def update_address_and_zipcode(
    df: pd.DataFrame,
    rows_to_update: pd.Index,
    id_key: pd.Series,
    address_map: Dict,
    zipcode_map: Dict,
    address_col_name: str = "address",
    zipcode_col_name: str = "zipcode",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df: the pd.DataFrame to update
    rows_to_update: a pd.Index of the rows of df to update
    id_key: the id values corresponding to the rows_to_update.
            should be the key for address_map and zipcode_map
    address_map: a Dict from id_key values to addresses
    zipcode_map: a Dict from id_key values to zipcodes
    address_col_name: a string. the name of the column in df to hold addresses.
    zipcode_col_name: a string. the name of the column in df to hold zipcodes.

    Returns
    -------
    df with appropriately updated addresses and zipcodes
    """
    df.loc[rows_to_update, address_col_name] = id_key.map(address_map)
    df.loc[rows_to_update, zipcode_col_name] = id_key.map(zipcode_map)
    return df
