from typing import Dict, List

import pandas as pd


def get_year_of_birth(data: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(data["date_of_birth"]).dt.year


def get_first_name_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["first_name_id"].astype(str)


def get_middle_name_id(data: pd.DataFrame) -> pd.Series:
    return data["random_seed"].astype(str) + "_" + data["middle_name_id"].astype(str)


# Fixme: Add formatting functions as necessary
COLUMN_FORMATTERS = {
    "year_of_birth": get_year_of_birth,
    "first_name_id": get_first_name_id,
    "middle_name_id": get_middle_name_id,
}


def format_data_for_mapping(
    index_name: str,
    raw_results: Dict[str, pd.DataFrame],
    columns_required: List[str],
    output_columns: List[str],
) -> pd.DataFrame:

    data_to_map = [obs_data[columns_required] for obs_data in raw_results.values()]
    data = pd.concat(data_to_map).drop_duplicates()
    for column in output_columns:
        if column in COLUMN_FORMATTERS:
            data[column] = COLUMN_FORMATTERS[column](data)

    data = data[output_columns].set_index(index_name)

    return data
