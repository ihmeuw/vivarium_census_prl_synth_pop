import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

from vivarium_census_prl_synth_pop.constants import data_values, paths


def generate_business_names_data(n_total_names: str):
    # loads csv of business names and generates pandas series with random business names

    business_names = pd.read_csv(paths.BUSINESS_NAMES_DATA)
    bigrams = make_bigrams(business_names)  # bigrams is a pd.Series with multi-index with first_word, second_word

    # Get frequency of business names and find uncommon ones
    s_name_freq = business_names.location_name.value_counts()
    real_but_uncommon_names = set(s_name_freq[s_name_freq < 1_000].index)

    # Generate random business names.  Drop duplicates and overlapping names with uncommon names
    n_total_names = int(n_total_names)  # Make int because of sys args
    new_names = pd.Series()

    # Generate additional names until desired number of random business names is met
    while len(new_names) < n_total_names:
        n_needed = n_total_names - len(new_names)
        more_names = sample_names(
            bigrams, n_needed, data_values.BUSINESS_NAMES_MAX_TOKENS_LENGTH
        )
        new_names = pd.concat([new_names, more_names]).drop_duplicates()
        new_names = new_names.loc[~new_names.isin(real_but_uncommon_names)]

    new_names = pd.Series(data=new_names, name="business_names")
    new_names.to_csv(paths.BUSINESS_NAMES_DATA_ARTIFACT_INPUT_PATH, compression="bz2", index=False)


def make_bigrams(df: pd.DataFrame):
    # Makes default dict of business names for map to sample from
    # bigrams will be a Dict[str: Dict[str: int]]
    # Example {"<start>": {keys are all first words for businesses: values are frequence where these pairs happen}}

    def dict_factory():
        return lambda: defaultdict(int)

    bigrams = defaultdict(dict_factory())
    n_rows = len(df)  # expect a few minutes

    for i in range(n_rows):
        if i % 10_000 == 0:
            print(".", end=" ", flush=True)
        names_i = df.iloc[i, 0]

        tokens_i = names_i.split(" ")
        for j in range(len(tokens_i)):
            if j == 0:
                bigrams["<start>"][tokens_i[j]] += 1
            else:
                bigrams[tokens_i[j - 1]][tokens_i[j]] += 1
        bigrams[tokens_i[j]]["<end>"] += 1

    return bigrams


def sample_names(bigrams: defaultdict, n_businesses: int, n_max_tokens: int) -> pd.Series:
    """

    Parameters
    ----------
    bigrams: Default dict produced from make_bigrams function (see formatting of default dict.
    n_businesses: Int of how many business names to generate
    n_max_tokens: Int of max number of words possible for a business name to contain

    Returns
    -------
    A string that is a randomly generated business name
    """

    logger.info("")
    columns = [f"word_{i}" for i in range(n_max_tokens)]
    names = pd.DataFrame(columns=columns)
    names["word_0"] = ["<start>"] * n_businesses

    for i in range(1, n_max_tokens):
        if i == 1:
            logger.info("Initializing random name sampling for business name generator")
        previous_word = f"word_{i - 1}"
        next_word = f"word_{i}"
        current_words_count_dict = names[previous_word].value_counts().to_dict()
        for word in current_words_count_dict.keys():
            logger.info(
                f"Generating {len(list(current_words_count_dict.keys()))} words "
                f"in {i}th of {n_max_tokens} columns for business names data."
            )
            if word != "<end>":
                vals = list(bigrams[word].keys())
                pr = np.array(list(bigrams[word].values()))
                tokens = np.random.choice(
                    vals, p=pr / pr.sum(), size=current_words_count_dict[word]
                )

                names.loc[names[previous_word] == word, next_word] = tokens

    # Process generated names by combining all columns and dropping outer tokens of <start> and <end>
    names = names.replace(np.nan, "", regex=True)
    names["business_names"] = names[columns[1:]].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )
    names["business_names"] = names["business_names"].str.split(" <").str[0]

    return names["business_names"]


if __name__ == "__main__":
    generate_business_names_data(sys.argv[1])
