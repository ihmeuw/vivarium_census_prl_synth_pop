import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

from vivarium_census_prl_synth_pop.constants import data_values, paths


def generate_business_names_data(n_total_names: str, batch_sets: str):
    # loads csv of business names and generates pandas series with random business names

    business_names = pd.read_csv(paths.BUSINESS_NAMES_DATA)
    bigrams = make_bigrams(business_names)

    # Get frequency of business names and find uncommon ones
    business_names_freq = business_names.location_name.value_counts()
    real_but_uncommon_names = set(business_names_freq[business_names_freq < 1_000].index)

    # Generate random business names.  Drop duplicates and overlapping names with uncommon names
    n_total_names = int(n_total_names)  # Make int because of sys args
    batch_sets = int(batch_sets)
    new_names = pd.Series()

    # Generate additional names until desired number of random business names is met
    cycles = 1
    sets = 0
    logger.info("Initializing random name sampling for business name generator")
    while sets < batch_sets:
        while len(new_names) < n_total_names:
            n_needed = n_total_names - len(new_names)
            if cycles > 1:
                logger.info(f"{n_needed} duplicates found.  Generating additional names for the {cycles} time.")
            start_time = time.time()
            more_names = sample_names(
                bigrams, n_needed, data_values.BUSINESS_NAMES_MAX_TOKENS_LENGTH
            )
            generation_time = time.time() - start_time
            logger.info(f"Total time to sample {n_needed} names for {cycles} iteration of name sample was {generation_time}.")

            # Check for duplicates
            duplicate_processing_start_time = time.time()
            new_names = pd.concat([new_names, more_names]).drop_duplicates()
            new_names = new_names.loc[~new_names.isin(real_but_uncommon_names)]
            duplicate_processing_time = time.time() - duplicate_processing_start_time
            logger.info(f"Total time used to check for duplicates was {duplicate_processing_time}.")
            cycles += 1

        new_names.to_csv(
            f"{paths.BUSINESS_NAMES_DATA_ARTIFACT_INPUT_PATH}/business_names_list_{sets}.csv.bz2",
            header=["business_names"],
            index=False,
            compression="bz2"
        )
        sets += 1


def make_bigrams(df: pd.DataFrame) -> defaultdict:
    """

    Parameters
    ----------
    df: A one column dataframe from a CSV containing a list of real business names.

    Makes default dict of business names for map to sample from
    Bigrams will be a Dict[str: Dict[str: int]]

    Example:  A series of business names to be turned into a Dict map
            location_name
        0 	Cottage Inn Pizza
        1 	Union St Barber Shop
        2 	Larkfield Auto Center
        3 	Randall Guns
        4 	Sport Clips

    Create bigrams to map a word and the frequence at which the next choice happens.

        {'<start>': {'Cottage': 1,
                     'Union': 1,
                     'Larkfield': 1,
                     'Randall': 1,
                     'Sport': 1
                     }
         'Cottage': {'Inn': 1
                    }
         'Inn':     {'Pizza': 1
                    }
         'Pizza':   {'<end>': 1
                    }
        }

    Go through each word in each name and get the frequency of each name pair in the entire series.  The default dict
        allows us to add to that frequency when we get a repeat and create a new key if that key does not exist.
    Returns
    -------
    defaultdict
    """

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
    bigrams: Default dict of format Dict{str: Dict{str: int}} capturing all word pairs and their frequencies from
        business names data..
    n_businesses: Int of how many business names to generate
    n_max_tokens: Int of max number of words possible for a business name to contain

    Returns
    -------
    A string that is a randomly generated business name
    """

    columns = [f"word_{i}" for i in range(n_max_tokens)]
    names = pd.DataFrame(columns=columns)
    names["word_0"] = ["<start>"] * n_businesses

    for i in range(1, n_max_tokens):
        previous_word = f"word_{i - 1}"
        next_word = f"word_{i}"
        current_words_count_dict = names[previous_word].value_counts().to_dict()
        for word in current_words_count_dict.keys():
            if word == "<end>":
                logger.info(f"Generated {current_words_count_dict[word]} business names containing {i} words.")
            else:
                vals = list(bigrams[word].keys())
                freq = np.array(list(bigrams[word].values()))
                tokens = np.random.choice(
                    vals, p=freq / freq.sum(), size=current_words_count_dict[word]
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
    generate_business_names_data(sys.argv[1:3])
