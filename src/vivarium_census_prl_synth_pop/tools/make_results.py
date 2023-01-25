import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger

from vivarium_census_prl_synth_pop import utilities
from vivarium_census_prl_synth_pop.constants import paths
from vivarium_census_prl_synth_pop.results_processing.generate_names import (
    generate_names,
)


def build_results(results_dir: str, mark_best: bool, test_run: bool) -> None:
    if mark_best and test_run:
        logger.error(
            "A test run can't be marked best. "
            "Please remove either the mark best or the test run flag."
        )
        return

    logger.info("Creating final results directory.")
    final_output_root_dir = utilities.build_output_dir(
        Path(results_dir), subdir=paths.FINAL_RESULTS_DIR_NAME
    )
    final_output_dir = final_output_root_dir / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    logger.info("Copying raw results to final location.")
    raw_output_dir = Path(results_dir) / paths.RAW_RESULTS_DIR_NAME
    # Perform post-processing
    perform_post_processing(raw_output_dir, final_output_dir)

    if test_run:
        logger.info("Test run - not marking results as latest.")
    else:
        create_results_link(final_output_dir, paths.LATEST_DIR_NAME)

    if mark_best:
        create_results_link(final_output_dir, paths.BEST_DIR_NAME)


def create_results_link(output_dir: Path, link_name: Path) -> None:
    logger.info(f"Marking results as {link_name}.")
    output_root_dir = output_dir.parent
    link_dir = output_root_dir / link_name
    link_dir.unlink(missing_ok=True)
    link_dir.symlink_to(output_dir, target_is_directory=True)


def perform_post_processing(raw_output_dir: Path, final_output_dir: Path) -> None:
    shutil.copytree(raw_output_dir, final_output_dir)
    generate_names(raw_output_dir, final_output_dir)
