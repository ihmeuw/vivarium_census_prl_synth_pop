import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import paths


def build_results(results_dir: str) -> None:
    logger.info("Creating final results directory.")
    final_output_dir = utilities.build_output_dir(
        Path(results_dir), subdir=paths.FINAL_RESULTS_DIR_NAME
    ) / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    logger.info("Copying raw results to final location.")
    raw_output_dir = Path(results_dir) / paths.RAW_RESULTS_DIR_NAME
    shutil.copytree(raw_output_dir, final_output_dir)
