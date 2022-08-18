"""collection of classes for generating sensitive data
synthetically, e.g. name, address, social-security number
"""
from typing import Union, List, Tuple, Dict

import pandas as pd
import numpy as np
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline

from vivarium_census_prl_synth_pop.utilities import random_integers

from vivarium_census_prl_synth_pop.constants import data_keys, data_values, metadata
from vivarium_census_prl_synth_pop.utilities import vectorized_choice

Array = Union[List, Tuple, np.ndarray, pd.Series]


class GenericGenerator:
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

    def generate(self, df_in: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=df_in.index)

    def noise(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class SSNGenerator(GenericGenerator):
    @property
    def name(self):
        return "SSNGenerator"

    def generate(self, df_in: pd.DataFrame) -> pd.DataFrame:
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

        area = random_integers(
            min_val=1, max_val=899, index=df.index, randomness=self.randomness
        )
        area = np.where(area == 666, 667, area)
        df["ssn_area"] = area

        group = random_integers(
            min_val=1, max_val=99, index=df.index, randomness=self.randomness
        )
        df["ssn_group"] = group

        serial = random_integers(
            min_val=1, max_val=9999, index=df.index, randomness=self.randomness
        )
        df["ssn_serial"] = serial

        df["ssn"] = ""
        df["ssn"] += df.ssn_area.astype(str).str.zfill(3)
        df["ssn"] += "-"
        df["ssn"] += df.ssn_group.astype(str).str.zfill(2)
        df["ssn"] += "-"
        df["ssn"] += df.ssn_serial.astype(str).str.zfill(4)
        return df

    def noise(self, df):
        df = df.copy()

        # TODO: add some errors in digits
        # typically just getting one digit wrong

        n_to_blank = (
            len(df.index) // 10
        )  # TODO: make this an optional parameter to this method and/or inform it with some evidence
        if n_to_blank > 0:
            blank_rows = vectorized_choice(
                options=df.index, n_to_choose=n_to_blank, randomness_stream=self.randomness
            )
            df.loc[blank_rows, "ssn"] = ""

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

    def random_first_names(self, randomness, yob, sex, size) -> np.ndarray:
        # we only have data up to 2020; for younger children, sample from 2020 names.
        if yob > 2020:
            yob = 2020
        grouped_name_data = self.first_name_data.groupby(["yob", "sex"])
        age_sex_specific_names = grouped_name_data.get_group((yob, sex))
        name_probabilities = (
            age_sex_specific_names["freq"] / age_sex_specific_names["freq"].sum()
        )
        return vectorized_choice(
            options=age_sex_specific_names.name,
            n_to_choose=size,
            randomness_stream=self.randomness,
            weights=name_probabilities,
        ).to_numpy()  # TODO: include spaces and hyphens

    def random_last_names(self, randomness, race_eth, size) -> np.ndarray:
        df_census_names = self.last_name_data

        # randomly sample last names
        last_names = vectorized_choice(
            options=df_census_names.name,
            n_to_choose=size,
            randomness_stream=self.randomness,
            weights=df_census_names[race_eth],
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
        hyphen_rows = randomness.get_draw(last_names.index) < probability_of_hyphen
        if hyphen_rows.sum() > 0:
            last_names[hyphen_rows] += (
                "-"
                + vectorized_choice(
                    options=df_census_names.name,
                    n_to_choose=hyphen_rows.sum(),
                    randomness_stream=self.randomness,
                    weights=df_census_names[race_eth],
                ).to_numpy()
            )

        # add spaces to some names
        probability_of_space = data_values.PROBABILITY_OF_SPACE_IN_NAME[race_eth]
        space_rows = randomness.get_draw(last_names.index) < probability_of_space * (
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
                self.randomness, current_year - age, sex, n
            )
            first_and_middle.loc[df_age.index, "middle_name"] = self.random_first_names(
                self.randomness, current_year - age, sex, n
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
                self.randomness, race_eth, len(df_race_eth)
            )
        # TODO: include household structure
        return pd.DataFrame(last_names, columns=["last_name"])

    def noise(self, df):
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


class Addresses(GenericGenerator):
    @property
    def name(self):
        return "Addresses"

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

    def determine_if_moving(self, options: pd.Series, move_rate_producer: Pipeline) -> pd.Series:
        options = options.drop_duplicates()
        those_that_move = self.randomness.filter_for_rate(
            options, move_rate_producer(options.index)
        )
        return those_that_move

    def get_new_addresses_and_zipcodes(self, those_that_move: pd.Index, state: str):
        """

        Parameters
        ----------
        those_that_move
        state

        Returns
        -------
        {those_that_move: addresses}, {those_that_move: zipcodes}
        """
        new_addresses = self.generate(
            those_that_move, state=state
        )
        return new_addresses["address"].to_dict(), new_addresses["zipcode"].to_dict()

    def update_address_and_zipcode(
            self,
            df: pd.DataFrame,
            rows_to_update: pd.Index,
            address_map: Dict,
            zipcode_map: Dict,
            address_col_name: str = "address",
            zipcode_col_name: str = "zipcode",
    ) -> pd.DataFrame:
        df.loc[rows_to_update, address_col_name] = rows_to_update.map(address_map)
        df.loc[rows_to_update, zipcode_col_name] = rows_to_update.map(zipcode_map)
        return df
