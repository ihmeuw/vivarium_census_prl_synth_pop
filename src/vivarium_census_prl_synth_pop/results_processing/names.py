from typing import Dict

import pandas as pd
from vivarium import Artifact

from vivarium_census_prl_synth_pop.results_processing.formatter import (
    format_data_for_mapping,
)


def get_first_name_map(
    index_name: str, raw_data: Dict[str, pd.DataFrame], artifact: Artifact
) -> Dict[str, pd.Series]:

    input_cols = [index_name, "date_of_birth", "sex", "random_seed"]
    output_cols = [index_name, "year_of_birth", "sex"]

    name_data = format_data_for_mapping(
        index_name=index_name,
        raw_results=raw_data,
        columns_required=input_cols,
        output_columns=output_cols,
    )

    # Note this is returning nothing right now since it is not part of this ticket.
    return {}
