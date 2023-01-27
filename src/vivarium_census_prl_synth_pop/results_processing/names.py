from typing import Dict

import pandas as pd
from vivarium import Artifact

from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)


def get_data_for_first_middle_name_mapping(
    index_name: str, raw_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:

    input_cols = [index_name, "date_of_birth", "sex", "random_seed"]
    output_cols = [index_name, "year_of_birth", "sex"]

    name_data = format_data_for_mapping(
        index_name=index_name,
        raw_results=raw_data,
        columns_required=input_cols,
        output_columns=output_cols,
    )

    return name_data


def get_first_name_map(
    raw_data: Dict[str, pd.DataFrame], artifact: Artifact
) -> Dict[str, pd.DataFrame]:
    first_name_data = get_data_for_first_middle_name_mapping(
        "first_name_id",
        raw_data,
    )

    return {}


def get_middle_name_map(
    raw_data: Dict[str, pd.DataFrame], artifact: Artifact
) -> Dict[str, pd.DataFrame]:
    middle_name_data = get_data_for_first_middle_name_mapping(
        "middle_name_id",
        raw_data,
    )

    return {}
